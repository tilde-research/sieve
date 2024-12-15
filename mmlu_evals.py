import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
import pickle
from dataclasses import asdict

from src.eval_config import EvalConfig
from src.wrapper import InterventionWrapper
import src.utils as utils


if __name__ == "__main__":
    # Step 1: Set up the device
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = EvalConfig()

    model_params = utils.get_model_params(config.model_name)

    tasks = [
        "mmlu_high_school_statistics",
        "mmlu_high_school_computer_science",
        "mmlu_high_school_mathematics",
        "mmlu_high_school_physics",
        "mmlu_high_school_biology",
    ]

    # Step 3: Create the InterventionWrapper
    wrapper = InterventionWrapper(config.model_name, device=device, dtype=torch.bfloat16)

    # Step 4: Load the SAE
    wrapper.load_sae(release=model_params["sae_release"], sae_id=model_params["sae_id"], layer_idx=model_params["targ_layer"])

    results = {"config": asdict(config)}

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.

    lm_obj = HFLM(pretrained=wrapper.model)

    task_manager = lm_eval.tasks.TaskManager()

    for intervention_type in config.intervention_types:
        results[intervention_type] = {}

        results[intervention_type][0] = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            num_fewshot=0,
            task_manager=task_manager,
        )

        for scale in config.scales:
            module, hook_fn = wrapper.get_hook(intervention_type, model_params, scale, config)

            # NOTE: Make sure to remove the hook after using it.
            handle = module.register_forward_hook(hook_fn)

            # Setting `task_manager` to the one above is optional and should generally be done
            # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
            # `simple_evaluate` will instantiate its own task_manager if it is set to None here.

            results[intervention_type][scale] = lm_eval.simple_evaluate(  # call simple_evaluate
                model=lm_obj,
                tasks=tasks,
                num_fewshot=0,
                task_manager=task_manager,
            )

            with open("results_mmlu_evals.pkl", "wb") as f:
                pickle.dump(results, f)

            handle.remove()
