from dataclasses import dataclass, field
from enum import Enum


class InterventionType(Enum):
    CONSTANT_SAE = "constant_sae"
    CONSTANT_STEERING_VECTOR = "constant_steering_vector"
    CONDITIONAL_PER_INPUT = "conditional_per_input"
    CONDITIONAL_PER_TOKEN = "conditional_per_token"
    CONDITIONAL_STEERING_VECTOR = "conditional_steering_vector"
    CLAMPING = "clamping"
    CONDITIONAL_CLAMPING = "conditional_clamping"
    PROBE_STEERING_VECTOR = "probe_steering_vector"
    PROBE_SAE = "probe_sae"
    PROBE_SAE_CLAMPING = "probe_sae_clamping"
    PROBE_STEERING_VECTOR_CLAMPING = "probe_steering_vector_clamping"
    SAE_STEERING_VECTOR = "sae_steering_vector"


@dataclass
class EvalConfig:
    random_seed: int = 42

    # Enum isn't serializable to JSON, so we use the value attribute
    intervention_types: list[str] = field(
        default_factory=lambda: [
            InterventionType.CLAMPING.value,
            InterventionType.CONDITIONAL_CLAMPING.value,
            InterventionType.CONSTANT_SAE.value,
            InterventionType.CONSTANT_STEERING_VECTOR.value,
            # InterventionType.CONDITIONAL_PER_INPUT.value,
            InterventionType.CONDITIONAL_PER_TOKEN.value,
            InterventionType.CONDITIONAL_STEERING_VECTOR.value,
            InterventionType.SAE_STEERING_VECTOR.value,
            InterventionType.PROBE_STEERING_VECTOR.value,
            InterventionType.PROBE_SAE.value,
        ]
    )

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct" # "google/gemma-2-9b-it" 
    # scales: list[int] = field(default_factory=lambda: [-10, -20, -40, -80, -160])
    scales: list[int] = field(default_factory=lambda: [ -2, -3, -4, -5, -6, -7, -8, -10, -20, -40])
    batch_size: int = 20
    total_generations: int = (200 // batch_size) * batch_size
    max_new_tokens: int = (
        400  # This needs to be high enough that we reach the end of the code block
    )

    prompt_filename: str = "prompt.txt"  # prompt.txt
    docs_filename: str = "pytest_docs.txt"
    code_filename: str = "code.json"
    contrastive_prompts_filename: str = "contrastive_prompts.json"
    probe_prompts_filename: str = "probe_prompts.json"

    prefill: str = "```python\n"
    prompt_folder: str = "src/prompts"
    prompt_type: str = "regex"

    encoder_threshold_bias: float = 0.0
    steering_vector_threshold_bias: float = -150.0

    use_llm_judge: bool = False  # True

    save_path: str = "gemma.json"
