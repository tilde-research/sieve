import torch
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Union, Any
import json
import time
import sys
import os
import logging
from jaxtyping import Float
from torch import Tensor

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.wrapper import InterventionWrapper
from src.eval_config import EvalConfig, InterventionType
import src.utils as utils
import src.caa as caa


def get_feature_acts_with_generations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    code_id: str,
    target_layer: int,
    encoder_vectors_dict: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Track feature activations during model generation.

    This function monitors how strongly the model activates specific features
    during text generation, accounting for various thresholds and biases.

    Args:
        model: The language model to analyze
        tokenizer: Tokenizer for processing text
        code_id: Identifier for the code block
        target_layer: Which transformer layer to monitor
        encoder_vectors_dict: Dict mapping intervention types to feature directions
        inputs: Input token ids
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        Dict mapping intervention types to feature activation tensors
    """

    if "question" in code_id or "without_regex" in code_id:
        filter_regex = True
    else:
        filter_regex = False

    acts_BLD, tokens_BL = caa.get_layer_activations_with_generation(
        model, target_layer, inputs, max_new_tokens, temperature
    )

    for i in range(tokens_BL.size(0)):
        single_prompt = tokens_BL[i]
        if not filter_regex:
            break
        decoded_prompt = tokenizer.decode(single_prompt)

        if (
            "regex" in decoded_prompt
            or "regular expression" in decoded_prompt
            or utils.check_for_re_usage(decoded_prompt)
        ):
            print(f"Skipping regex prompt: {decoded_prompt}")
            tokens_BL[i, :] = torch.tensor([tokenizer.pad_token_id])

    tokens_BL = tokens_BL[:, :-1]  # There are no activations for the last generated token
    tokens_L = tokens_BL.flatten()

    retain_mask = (
        (tokens_L != tokenizer.pad_token_id)
        & (tokens_L != tokenizer.eos_token_id)
        & (tokens_L != tokenizer.bos_token_id)
    )

    # tokens_BL = tokens_BL[:, inputs.size(1) :]  # Remove input tokens
    # acts_BLD = acts_BLD[:, inputs.size(1) :, :]

    feature_acts_dict = {}
    for intervention_type, encoder_vector_D in encoder_vectors_dict.items():
        feature_acts_BL = torch.einsum("BLD,D->BL", acts_BLD, encoder_vector_D.to(acts_BLD.device))
        feature_acts_L = feature_acts_BL.flatten()

        feature_acts_dict[intervention_type] = feature_acts_L[retain_mask]

    return feature_acts_dict


def test_single_prompt(
    wrapper: InterventionWrapper,
    base_prompt: str,
    code_id: str,
    code_example: str,
    config: EvalConfig,
    model_params: dict,
) -> Dict[str, torch.Tensor]:
    """Test activation patterns for a single prompt.

    Args:
        wrapper: Model wrapper instance
        base_prompt: Base prompt template
        code_example: Code example to test
        config: Evaluation configuration
        model_params: Model-specific parameters

    Returns:
        Dictionary mapping intervention types to activation tensors
    """
    if "question" in code_id:
        prompt = code_example
    else:
        prompt = base_prompt.replace("{code}", code_example)

    formatted_prompt = utils.format_llm_prompt(prompt, wrapper.tokenizer)
    formatted_prompt += config.prefill

    batched_prompts = [formatted_prompt] * config.batch_size

    num_batches = config.total_generations // config.batch_size

    input_tokens = wrapper.tokenizer(
        batched_prompts, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(wrapper.model.device)

    # Get encoder vectors for all intervention types
    encoder_vectors_dict = {}
    for intervention_type in config.intervention_types:
        if intervention_type == InterventionType.PROBE_SAE.value:
            encoder_vectors_dict[intervention_type] = wrapper.probe_vector
            print(f"Probe bias: {wrapper.probe_bias}")
        elif intervention_type == InterventionType.CONDITIONAL_PER_TOKEN.value:
            encoder_vector = wrapper.sae.W_enc[:, [model_params["feature_idx"]]].squeeze()
            encoder_vectors_dict[intervention_type] = encoder_vector
            bias = wrapper.sae.b_enc[model_params["feature_idx"]]
            # threshold = wrapper.sae.threshold[model_params["feature_idx"]]
            # print(f"Threshold: {threshold}, Bias: {bias}")
        elif intervention_type == InterventionType.CONDITIONAL_STEERING_VECTOR.value:
            encoder_vectors_dict[intervention_type] = wrapper.caa_steering_vector
        else:
            raise ValueError(f"Invalid intervention type: {intervention_type}")

    feature_acts_by_type = {itype: [] for itype in config.intervention_types}

    for _ in tqdm(range(num_batches), desc="Generating responses"):
        feature_acts_dict = get_feature_acts_with_generations(
            wrapper.model,
            wrapper.tokenizer,
            code_id,
            model_params["targ_layer"],
            encoder_vectors_dict,
            input_tokens,
            max_new_tokens=config.max_new_tokens,
        )
        for itype in config.intervention_types:
            feature_acts_by_type[itype].append(feature_acts_dict[itype])

    return {itype: torch.cat(acts_list, dim=0) for itype, acts_list in feature_acts_by_type.items()}


def count_classifier_activations() -> dict:
    """Run comprehensive activation analysis across multiple code examples.

    This function:
    1. Sets up the model and configuration
    2. Loads necessary components (SAE, prompts)
    3. Runs activation analysis for different intervention types
    4. Saves results periodically

    Returns:
        Dictionary containing:
        - Configuration settings
        - Activation measurements per intervention type
        - Results for each code example

    Note:
        Results are saved to disk after each code block to prevent data loss
    """
    config = EvalConfig()

    config.intervention_types = [
        InterventionType.CONDITIONAL_PER_TOKEN.value,
        InterventionType.PROBE_SAE.value,
        InterventionType.CONDITIONAL_STEERING_VECTOR.value,
    ]

    config.save_path = "activation_values.pt"
    config.prompt_filename = "prompt_no_regex.txt"

    results = {"config": asdict(config)}

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = utils.get_model_params(config.model_name)

    # Initialize wrapper
    wrapper = InterventionWrapper(model_name=config.model_name, device=device, dtype=torch.bfloat16)

    # Load SAE
    wrapper.load_sae(
        release=model_params["sae_release"],
        sae_id=model_params["sae_id"],
        layer_idx=model_params["targ_layer"],
    )

    # Load and format prompt
    base_prompt = utils.load_prompt_files(config)
    # initialize steering vectors
    for intervention_type in config.intervention_types:
        _ = wrapper.get_hook(intervention_type, model_params, 1, config)

    with open(f"{config.prompt_folder}/activation_counting_code.json", "r") as f:
        code_blocks = json.load(f)

    print(f"Evaluating {len(code_blocks)} code blocks")
    print(f"Evaluating the following interventions: {config.intervention_types}")

    for code_block_key, single_code_block in code_blocks.items():
        activations_by_type = test_single_prompt(
            wrapper,
            base_prompt,
            code_block_key,
            single_code_block,
            config,
            model_params,
        )

        for intervention_type in config.intervention_types:
            if intervention_type not in results:
                results[intervention_type] = {}
            results[intervention_type][code_block_key] = activations_by_type[intervention_type]

        torch.save(results, config.save_path)

    return results


if __name__ == "__main__":
    """
    Main entry point for activation analysis.
    
    Environment variables:
        PYTORCH_CUDA_ALLOC_CONF: Set to "expandable_segments:True"
    """
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.set_grad_enabled(False)

    start_time = time.time()
    run_results = count_classifier_activations()
    print(f"Total time: {time.time() - start_time:.2f} seconds")
