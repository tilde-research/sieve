"""
Regex Interventions Module

This module implements testing and evaluation of model interventions specifically 
focused on regex pattern usage in generated code. It provides functionality to:
1. Run controlled experiments with different intervention types
2. Measure regex usage in generated code
3. Evaluate code quality and correctness
4. Compare baseline and intervention results
"""

import torch
from transformers import AutoTokenizer
from dataclasses import asdict
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Union
import json
import time
import asyncio
import re
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wrapper import InterventionWrapper
from src.eval_config import EvalConfig, InterventionType
import src.utils as utils
import src.caa as caa
from src.agent_eval import batch_evaluate_generations, summarize_evaluations


def measure_regex_usage(
    responses: list[str], 
    prompt: str, 
    prefill: str
) -> tuple[int, int, int, int]:
    """Analyze generated code for regex usage and validity.
    
    Args:
        responses: List of generated text responses
        prompt: Original prompt used for generation
        prefill: Prefix text to remove from responses
        
    Returns:
        Tuple containing counts of:
        - Valid Python code snippets
        - Snippets using regex
        - Syntactically valid snippets
        - Semantically valid snippets
    """
    valid_python_count = 0
    syntactically_valid_python_count = 0
    semantically_valid_python_count = 0
    regex_usage_count = 0

    for i, response in tqdm(enumerate(responses), desc="Measuring regex usage"):
        response = re.sub(r'(?<!\s)!=', ' !=', response)
        if prompt in response:
            response = response[len(prompt) - len(prefill) :]
        python = utils.extract_python(response)
        if python:
            valid_python_count += 1
            if utils.check_for_re_usage(python):
                regex_usage_count += 1
            try:
                if utils.is_syntactically_valid_python(python)[0]:
                    syntactically_valid_python_count += 1
                valid_code = utils.validate_single_llm_response(prompt, response, verbose=False)
                if valid_code:
                    semantically_valid_python_count += 1
            except:
                print(f"Error in response {i}: {response}")
                continue

    return (
        valid_python_count,
        regex_usage_count,
        syntactically_valid_python_count,
        semantically_valid_python_count,
    )


def extract_response(responses: list[str], prompt: str, prefill: str) -> list[str]:
    """Clean and extract actual responses from model outputs.
    
    Args:
        responses: Raw model outputs
        prompt: Original prompt to remove
        prefill: Prefix text to remove
        
    Returns:
        List of cleaned response texts
    """
    extracted_responses = []
    for i, response in enumerate(responses):
        response = re.sub(r'(?<!\s)!=', ' !=', response)
        if prompt in response:
            response = response[len(prompt) - len(prefill) :]
        extracted_responses.append(response)
    return extracted_responses


def run_generation(
    wrapper: InterventionWrapper,
    prompt: str,
    batch_size: int,
    total_generations: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate multiple responses without interventions.
    
    Args:
        wrapper: Model wrapper instance
        prompt: Input prompt
        batch_size: Number of generations per batch
        total_generations: Total number of responses to generate
        max_new_tokens: Maximum new tokens per generation
        
    Returns:
        List of generated responses
    """
    batched_prompts = [prompt] * batch_size
    num_batches = total_generations // batch_size
    generations = []

    for _ in tqdm(range(num_batches), desc="Generating responses"):
        response = wrapper.generate(batched_prompts, max_new_tokens=max_new_tokens)
        generations.extend(response)

    return generations


def run_intervention(
    wrapper: InterventionWrapper,
    prompt: str,
    batch_size: int,
    total_generations: int,
    max_new_tokens: int,
    intervention_type: str,
    model_params: dict,
    scale: int,
    config: EvalConfig,
) -> list[str]:
    """Generate responses with specified intervention.
    
    Args:
        wrapper: Model wrapper instance
        prompt: Input prompt
        batch_size: Number of generations per batch
        total_generations: Total number of responses to generate
        max_new_tokens: Maximum new tokens per generation
        intervention_type: Type of intervention to apply
        model_params: Parameters for the intervention
        scale: Scale factor for intervention
        config: Evaluation configuration
        
    Returns:
        List of generated responses with intervention applied
    """
    batched_prompts = [prompt] * batch_size
    num_batches = total_generations // batch_size
    generations = []

    module_and_hook_fn = wrapper.get_hook(intervention_type, model_params, scale, config)

    for _ in tqdm(range(num_batches), desc="Generating responses"):
        response = wrapper.generate(
            batched_prompts, max_new_tokens=max_new_tokens, module_and_hook_fn=module_and_hook_fn
        )
        generations.extend(response)

    return generations


def test_single_prompt(
    wrapper: InterventionWrapper,
    base_prompt: str,
    code_example: str,
    config: EvalConfig,
    model_params: dict,
    intervention_type: str,
    api_key: str,
) -> dict:
    """Test a single prompt with and without interventions.
    
    Args:
        wrapper: Model wrapper instance
        base_prompt: Base prompt template
        code_example: Code example to insert in prompt
        config: Evaluation configuration
        model_params: Model-specific parameters
        intervention_type: Type of intervention to test
        api_key: API key for LLM judge (if used)
        
    Returns:
        Dictionary containing:
        - Original and intervention generations
        - Evaluation results
        - Agent evaluations (if enabled)
        - Result summaries
    """
    results = {"generations": {}, "eval_results": {}, "agent_evals": {}, "agent_summaries": {}}

    # Format prompt for this code example
    prompt = base_prompt.replace("{code}", code_example)
    formatted_prompt = utils.format_llm_prompt(prompt, wrapper.tokenizer)
    formatted_prompt += config.prefill

    # Generate without interventions
    original_texts = run_generation(
        wrapper,
        formatted_prompt,
        config.batch_size,
        config.total_generations,
        config.max_new_tokens,
    )
    original_texts = extract_response(original_texts, formatted_prompt, config.prefill)
    results["generations"]["original"] = original_texts
    results["eval_results"]["original"] = measure_regex_usage(
        original_texts, formatted_prompt, config.prefill
    )
    logging.info(f"Original eval results: {results['eval_results']['original']}")
    if config.use_llm_judge:
        # Add agent evaluations for original texts
        agent_evals = batch_evaluate_generations(original_texts, prompt, api_key)
        results["agent_evals"]["original"] = [asdict(eval) for eval in agent_evals]
        results["agent_summaries"]["original"] = summarize_evaluations(agent_evals)

    # Generate with different intervention scales
    for scale in tqdm(config.scales, desc="Interventions"):
        modified_texts = run_intervention(
            wrapper,
            formatted_prompt,
            config.batch_size,
            config.total_generations,
            config.max_new_tokens,
            intervention_type,
            model_params,
            scale,
            config,
        )

        modified_texts = extract_response(modified_texts, formatted_prompt, config.prefill)
        results["generations"][f"intervention_{scale}"] = modified_texts
        results["eval_results"][f"intervention_{scale}"] = measure_regex_usage(
            modified_texts, formatted_prompt, config.prefill
        )
        logging.info(f"Intervention {scale} eval results: {results['eval_results'][f'intervention_{scale}']}")
        if config.use_llm_judge:
            # Add agent evaluations for interventions
            agent_evals = batch_evaluate_generations(modified_texts, prompt, api_key)
            results["agent_evals"][f"intervention_{scale}"] = [asdict(eval) for eval in agent_evals]
            results["agent_summaries"][f"intervention_{scale}"] = summarize_evaluations(agent_evals)

    results["prompt"] = formatted_prompt
    return results


def test_sae_interventions(api_key: str) -> dict:
    """Run comprehensive intervention tests across multiple code examples.
    
    Args:
        api_key: API key for LLM judge evaluations
        
    Returns:
        Dictionary containing all test results and configurations
        
    Note:
        Results are saved after each code block to prevent data loss
    """
    config = EvalConfig()
    results = {"config": asdict(config)}

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = utils.get_model_params(config.model_name)

    # Initialize wrapper
    wrapper = InterventionWrapper(model_name=config.model_name, device=device, dtype=torch.bfloat16)

    # Load SAE
    wrapper.load_sae(release=model_params["sae_release"], sae_id=model_params["sae_id"], layer_idx=model_params["targ_layer"])

    # Load and format prompt
    base_prompt = utils.load_prompt_files(config)

    with open(f"{config.prompt_folder}/{config.code_filename}", "r") as f:
        code_blocks = json.load(f)

    print(f"Evaluating {len(code_blocks)} code blocks")

    print(f"Evaluating the following interventions: {config.intervention_types}")

    for intervention_type in config.intervention_types:
        print(f"Evaluating {intervention_type}!")
        results[intervention_type] = {"code_results": {}}
        for code_block_key, single_code_block in code_blocks.items():
            results[intervention_type]["code_results"][code_block_key] = test_single_prompt(
                wrapper,
                base_prompt,
                single_code_block,
                config,
                model_params,
                intervention_type,
                api_key,
            )

            # Save results after each code block
            with open(config.save_path, "w") as f:
                json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    """
    Main entry point for running intervention tests.
    
    Environment variables:
        PYTORCH_CUDA_ALLOC_CONF: Set to "expandable_segments:True"
        TOKENIZERS_PARALLELISM: Set to "false" for process safety
        OPENAI_API_KEY: Optional API key for LLM judge
    """
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # We disable this because we launch additional processes when checking for valid code
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import argparse

    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", type=str, help="OpenAI API key", default=os.environ.get("OPENAI_API_KEY")
    )
    args = parser.parse_args()

    start_time = time.time()
    run_results = test_sae_interventions(args.api_key)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

    run_filename = "run_results.json"
    with open(run_filename, "w") as f:
        json.dump(run_results, f, indent=4)
