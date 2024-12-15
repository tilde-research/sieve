import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from typing import Callable, Optional, Union, List, cast
from jaxtyping import Float
from torch import Tensor
import contextlib

from src.eval_config import EvalConfig, InterventionType
import src.caa as caa
from sae.sae import Sae

try:
    import flash_attn
    USE_FA = True
    print("Flash attention installed")
except ImportError:
    print("Flash attention not installed, using regular attention")
    USE_FA = False


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_addition_output_hook(
    vectors: list[Float[Tensor, "d_model"]], coeffs: list[float]
) -> Callable:
    """Creates a hook function that adds scaled vectors to layer activations.

    This hook performs a simple activation steering by adding scaled vectors
    to the layer's output activations. This is the most basic form of intervention.

    Args:
        vectors: List of vectors to add, each of shape (d_model,)
        coeffs: List of scaling coefficients for each vector

    Returns:
        Hook function that modifies layer activations

    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            resid_BLD = output[0]
            rest = output[1:]
        else:
            resid_BLD = output
            rest = ()

        for vector, coeff in zip(vectors, coeffs):
            vector = vector.to(resid_BLD.device)
            resid_BLD = resid_BLD + coeff * vector

        if rest:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def get_conditional_per_input_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally applies interventions based on input-level activation.

    This hook checks if any token in the input sequence triggers the encoder vector
    above threshold. If triggered, applies the intervention to the entire sequence.

    Args:
        encoder_vectors: List of vectors used to detect activation patterns
        decoder_vectors: List of vectors to add when conditions are met
        scales: Scaling factors for decoder vectors
        encoder_thresholds: Threshold values for each encoder vector

    Returns:
        Hook function that conditionally modifies activations

    Note:
        - Zeros out BOS token activations to prevent false triggers
        - Intervention applies to entire sequence if any token triggers
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]
            rest = output[1:]
        else:
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output
            rest = ()

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            encoder_vector_D = encoder_vector_D.to(resid_BLD.device)
            decoder_vector_D = decoder_vector_D.to(resid_BLD.device)

            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)
            feature_acts_BL[:, 0] = 0  # zero out the BOS token
            intervention_threshold_B11 = ((feature_acts_BL > encoder_threshold).any(dim=1).float())[
                :, None, None
            ]
            decoder_BLD = einops.repeat(decoder_vector_D * coeff, "D -> B L D", B=B, L=L).to(
                dtype=resid_BLD.dtype
            )

            resid_BLD += decoder_BLD * intervention_threshold_B11

        if rest:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def get_conditional_per_token_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally applies interventions per token.

    Unlike the per-input hook, this applies interventions independently to each token
    based on whether it exceeds the encoder threshold.

    Args:
        encoder_vectors: List of vectors used to detect activation patterns
        decoder_vectors: List of vectors to add when conditions are met
        scales: Scaling factors for decoder vectors
        encoder_thresholds: Threshold values for each encoder vector

    Returns:
        Hook function that modifies activations on a per-token basis

    Note:
        More granular than per-input hook as it can selectively modify
        specific tokens in the sequence
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]
            rest = output[1:]
        else:
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output
            rest = ()

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            encoder_vector_D = encoder_vector_D.to(resid_BLD.device)
            decoder_vector_D = decoder_vector_D.to(resid_BLD.device)

            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)
            intervention_mask_BL = feature_acts_BL > encoder_threshold
            decoder_BLD = einops.repeat(decoder_vector_D * coeff, "D -> B L D", B=B, L=L).to(
                dtype=resid_BLD.dtype
            )

            resid_BLD = torch.where(
                intervention_mask_BL.unsqueeze(-1),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        if rest:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def get_clamping_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
) -> Callable:
    """Creates a hook function that clamps activations using decoder vectors.

    This hook fixes the activations to a target value in the decoder vector direction.

    Args:
        encoder_vectors: List of vectors defining directions to clamp
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping (acts as offset after zeroing)

    Returns:
        Hook function that clamps and redirects activations

    Note:
        Useful for always having a final activation value in the decoder vector direction.
    """

    # coeff = -feature_acts_BL
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]
            rest = output[1:]
        else:
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output
            rest = ()
        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff in zip(
            encoder_vectors, decoder_vectors, scales
        ):
            encoder_vector_D = encoder_vector_D.to(resid_BLD.device)
            decoder_vector_D = decoder_vector_D.to(resid_BLD.device)
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)
            decoder_BLD = (-feature_acts_BL[:, :, None] + coeff) * decoder_vector_D[None, None, :]
            resid_BLD = torch.where(
                feature_acts_BL[:, :, None] > 0,
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        if rest:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def get_conditional_clamping_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally clamps activations.

    Combines conditional intervention with clamping - only clamps activations
    when they exceed the encoder threshold with the decoder intervention.

    Args:
        encoder_vectors: List of vectors defining directions to monitor
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping
        encoder_thresholds: Threshold values that trigger clamping

    Returns:
        Hook function that conditionally clamps and modifies activations

    Note:
        Most sophisticated intervention type, combining benefits of
        conditional application and activation clamping
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]
            rest = output[1:]
        else:
            resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output
            rest = ()

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            encoder_vector_D = encoder_vector_D.to(resid_BLD.device)
            decoder_vector_D = decoder_vector_D.to(resid_BLD.device)

            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = (-feature_acts_BL[:, :, None] + coeff) * decoder_vector_D[None, None, :]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = torch.where(
                (intervention_mask_BL[:, :, None] & (feature_acts_BL[:, :, None] > 0)),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        if rest:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


class InterventionWrapper:
    """Wrapper class for applying interventions to language models.

    This class manages model loading, intervention application, and generation
    with various steering techniques.

    Attributes:
        model: The underlying language model
        tokenizer: Tokenizer for the model
        sae: Optional Sparse Autoencoder for intervention
        device: Device to run computations on
        caa_steering_vector: Cached steering vector for interventions
        probe_vector: Cached probe vector for guided steering
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Initialize the wrapper with a specified model.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device ('cuda' or 'cpu')
            dtype: Data type for model weights
        """

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", # Load model on first visible GPU
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if USE_FA else "eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.interventions = {}
        self.sae: Optional[Union[SAE, Sae]] = None
        self.device = device
        self.caa_steering_vector: Optional[Tensor] = None
        self.probe_vector: Optional[Tensor] = None
        self.probe_bias: Optional[Tensor] = None

    def generate(
        self,
        batched_prompts: list[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        module_and_hook_fn: Optional[tuple[torch.nn.Module, Callable]] = None,
    ) -> list[str]:
        """Generate text with optional interventions.

        Args:
            batched_prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            module_and_hook_fn: Optional tuple of (module, hook_function) for intervention

        Returns:
            List of generated text strings

        Note:
            Prompts must contain the model's BOS token
        """
        assert all(
            self.tokenizer.bos_token in prompt for prompt in batched_prompts
        ), "All prompts must contain the BOS token."
        batched_tokens = self.tokenizer(
            batched_prompts, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        batched_tokens = batched_tokens["input_ids"]

        if module_and_hook_fn:
            module, hook_fn = module_and_hook_fn
            context_manager = add_hook(
                module=module,
                hook=hook_fn,
            )
            with context_manager:
                generated_toks = self.model.generate(
                    batched_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
        else:
            generated_toks = self.model.generate(
                batched_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        response = [self.tokenizer.decode(tokens) for tokens in generated_toks]

        return response

    def get_hook(
        self,
        intervention_type: str,
        model_params: dict,
        scale: Union[int, float],
        config: EvalConfig,
    ) -> tuple[torch.nn.Module, Callable]:
        """Create a hook function for the specified intervention type.

        Args:
            intervention_type: Type of intervention to apply
            model_params: Parameters for the intervention including:
                - targ_layer: Target layer index
                - feature_idx: Feature index for SAE
            scale: Scaling factor for the intervention
            config: Configuration for evaluation and steering

        Returns:
            Tuple of (target_module, hook_function)

        Raises:
            AttributeError: If SAE is required but not loaded
            ValueError: If intervention type is not supported

        Note:
            Different intervention types require different preconditions:
            - SAE interventions require loaded SAE
            - Steering vector interventions calculate vectors on first use
            - Probe interventions train probes on first use
        """
        module = self.model.model.layers[model_params["targ_layer"]]

        # Convert scale to float for type compatibility
        scales: List[float] = [float(scale)]

        if self.sae is None:
            raise AttributeError("SAE must be loaded before getting hook")

        # Get encoder/decoder vectors with proper null checks
        encoder_vectors = [cast(Tensor, self.sae.W_enc[:, [model_params["feature_idx"]]].squeeze())]
        decoder_vectors = [cast(Tensor, self.sae.W_dec[[model_params["feature_idx"]]].squeeze())]

        # Normalize decoder vectors
        decoder_vectors = [v / v.norm() for v in decoder_vectors]
        encoder_thresholds = [2.0]  if "gemma" in self.model_name else [5.0] # TODO make this an arg, llama 8B features use a higher threshold
        print(f"Encoder thresholds: {encoder_thresholds}")
        if hasattr(self.sae, "threshold"): # Check for jumprelu
            threshold_val = float(self.sae.threshold[model_params["feature_idx"]])
            encoder_thresholds = [threshold_val]

        # for i, threshold in enumerate(encoder_thresholds):
        #     encoder_thresholds[i] = threshold + config.encoder_threshold_bias

        # Initialize steering vectors if needed
        steering_vectors: List[Tensor] = []
        if intervention_type in [
            InterventionType.CONSTANT_STEERING_VECTOR.value,
            InterventionType.CONDITIONAL_STEERING_VECTOR.value,
            InterventionType.SAE_STEERING_VECTOR.value,
            InterventionType.PROBE_STEERING_VECTOR.value,
            InterventionType.PROBE_STEERING_VECTOR_CLAMPING.value,
        ]:
            if self.caa_steering_vector is None:
                self.caa_steering_vector = caa.calculate_steering_vector(
                    config.prompt_folder,
                    config.contrastive_prompts_filename,
                    config.code_filename,
                    self.model,
                    self.tokenizer,
                    config.prompt_type,
                    model_params["targ_layer"],
                )

            # TODO: Support multiple steering vectors
            steering_vector_threshold = (
                caa.get_threshold(
                    config,
                    model_params,
                    self,
                    self.caa_steering_vector,
                    encoder_vectors[0],
                    encoder_thresholds[0],
                )
                + config.steering_vector_threshold_bias
            )

            steering_vector_thresholds = [steering_vector_threshold]
            encoder_steering_vectors = [self.caa_steering_vector]
            steering_vectors = [self.caa_steering_vector]
            steering_vectors = [v / v.norm() for v in steering_vectors]

        if intervention_type in [
            InterventionType.PROBE_SAE.value,
            InterventionType.PROBE_SAE_CLAMPING.value,
            InterventionType.PROBE_STEERING_VECTOR.value,
            InterventionType.PROBE_STEERING_VECTOR_CLAMPING.value,
        ]:
            if self.probe_vector is None:
                self.probe_vector, self.probe_bias = caa.calculate_probe_vector(
                    config.prompt_folder,
                    config.probe_prompts_filename,
                    config.code_filename,
                    self.model,
                    self.tokenizer,
                    config.prompt_type,
                    model_params["targ_layer"],
                )
            probe_vector_threshold = [self.probe_bias]

        if intervention_type == InterventionType.CONSTANT_SAE.value:
            hook_fn = get_activation_addition_output_hook(decoder_vectors, scales)
        elif intervention_type == InterventionType.CONSTANT_STEERING_VECTOR.value:
            hook_fn = get_activation_addition_output_hook(steering_vectors, scales)
        elif intervention_type == InterventionType.PROBE_STEERING_VECTOR.value:
            hook_fn = get_conditional_per_token_hook(
                [self.probe_vector], steering_vectors, scales, probe_vector_threshold
            )
        elif intervention_type == InterventionType.PROBE_SAE.value:
            hook_fn = get_conditional_per_token_hook(
                [self.probe_vector], decoder_vectors, scales, probe_vector_threshold
            )
        elif intervention_type == InterventionType.PROBE_SAE_CLAMPING.value:
            hook_fn = get_conditional_clamping_hook(
                [self.probe_vector], decoder_vectors, scales, probe_vector_threshold
            )
        elif intervention_type == InterventionType.PROBE_STEERING_VECTOR_CLAMPING.value:
            hook_fn = get_conditional_clamping_hook(
                [self.probe_vector], steering_vectors, scales, probe_vector_threshold
            )
        elif intervention_type == InterventionType.SAE_STEERING_VECTOR.value:
            hook_fn = get_conditional_per_token_hook(
                encoder_vectors, steering_vectors, scales, encoder_thresholds
            )
        elif intervention_type == InterventionType.CONDITIONAL_PER_INPUT.value:
            hook_fn = get_conditional_per_input_hook(
                encoder_vectors, decoder_vectors, scales, encoder_thresholds
            )
        elif intervention_type == InterventionType.CONDITIONAL_PER_TOKEN.value:
            hook_fn = get_conditional_per_token_hook(
                encoder_vectors, decoder_vectors, scales, encoder_thresholds
            )
        elif intervention_type == InterventionType.CONDITIONAL_STEERING_VECTOR.value:
            hook_fn = get_conditional_per_token_hook(
                encoder_steering_vectors, steering_vectors, scales, steering_vector_thresholds
            )
        elif intervention_type == InterventionType.CLAMPING.value:
            hook_fn = get_clamping_hook(encoder_vectors, decoder_vectors, scales)
        elif intervention_type == InterventionType.CONDITIONAL_CLAMPING.value:
            hook_fn = get_conditional_clamping_hook(
                encoder_vectors, decoder_vectors, scales, encoder_thresholds
            )
        else:
            raise ValueError(f"Unsupported intervention type: {intervention_type}")
        return module, hook_fn

    def load_sae(self, release: str, sae_id: str, layer_idx: int):
        """Load a Sparse Autoencoder for interventions.

        Args:
            release: Release identifier for the SAE
            sae_id: Specific SAE identifier
            layer_idx: Layer index the SAE was trained on

        Note:
            Supports both tilde-research and standard SAE formats
        """
        if "tilde-research" in release:
            self.sae = Sae.from_pretrained(release, layer_idx=layer_idx)
            self.sae = self.sae.to(dtype=self.model.dtype, device=self.model.device)
            return
        self.sae, _, _ = SAE.from_pretrained(
            release=release, sae_id=sae_id, device=str(self.model.device)
        )
        self.sae = self.sae.to(dtype=self.model.dtype, device=self.model.device)
