import torch
from src.wrapper import InterventionWrapper
from src.eval_config import EvalConfig, InterventionType

# Step 1: Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 2: Define model parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 12 
FEATURE_IDX =  9853
SCALE = -8.0

# Step 3: Create the InterventionWrapper
wrapper = InterventionWrapper(MODEL_NAME, device=device)

# Step 4: Load the SAE
wrapper.load_sae(f"tilde-research/SIEVE_8B_coding", sae_id=None, layer_idx=LAYER)

# Step 5: Set up the intervention
config = EvalConfig()
model_params = {
    "targ_layer": LAYER,
    "feature_idx": FEATURE_IDX
}



# Step 6: Format input text using chat template
input_text = "Write a python function using the re module to match a numerical substring in a string."
chat = [{"role": "user", "content": input_text}]
formatted_text = wrapper.tokenizer.apply_chat_template(
    chat, 
    tokenize=False,
    add_generation_prompt=True
)

# Step 7: Generate text without intervention
print("Generating without intervention...")
generated_text_original = wrapper.generate(
    [formatted_text], 
    max_new_tokens=200,
    temperature=0.2,
    module_and_hook_fn=None
)[0]

# Step 8: Generate text with intervention
print("\nGenerating with intervention...")
module_and_hook_fn = wrapper.get_hook(
    intervention_type=InterventionType.CONDITIONAL_PER_TOKEN.value,
    model_params=model_params,
    scale=SCALE,
    config=config
)

generated_text_intervened = wrapper.generate(
    [formatted_text], 
    max_new_tokens=800,
    temperature=0.2,
    # repetition_penalty=1.15,
    module_and_hook_fn=module_and_hook_fn
)[0]

# Step 8: Print and compare the results
print("Original generated text:")
print(generated_text_original)
print("\nGenerated text with intervention:")
print(generated_text_intervened)

# Optional: Calculate and print the difference in token length
# original_tokens = len(wrapper.model.to_tokens(generated_text_original)[0])
# intervened_tokens = len(wrapper.model.to_tokens(generated_text_intervened)[0])
# print(f"\nOriginal generation length: {original_tokens} tokens")
# print(f"Intervened generation length: {intervened_tokens} tokens")
# print(f"Difference: {intervened_tokens - original_tokens} tokens")
