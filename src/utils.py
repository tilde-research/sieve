import re
import ast
from typing import Optional, Tuple
import builtins
from contextlib import contextmanager
import sys
from io import StringIO
from transformers import AutoTokenizer
import random
from multiprocessing import Process, Manager


from src.eval_config import EvalConfig


def get_model_params(model_name: str) -> dict[str, str | int]:
    """Get model-specific parameters for interventions.

    Args:
        model_name: Name/path of the model to get parameters for

    Returns:
        Dictionary containing:
        - sae_release: SAE model release identifier
        - sae_id: Specific SAE identifier (if applicable)
        - targ_layer: Target layer for intervention
        - feature_idx: Feature index in SAE

    Raises:
        ValueError: If model is not supported
    """
    if model_name == "google/gemma-2-9b-it":
        return {
            "sae_release": "gemma-scope-9b-it-res",
            "sae_id": "layer_9/width_16k/average_l0_88",
            "targ_layer": 9,
            "feature_idx": 3585,
            "secondary_feature_idx": 12650,
        }
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return {
            "sae_release": "tilde-research/sieve_coding",
            "sae_id": None,
            "targ_layer": 12, # 8
            "feature_idx": 9853, # 9699
        }
    elif model_name == "google/gemma-2-2b-it":
        return {
            "sae_release": "gemma-scope-2b-pt-res",
            "sae_id": "layer_8/width_16k/average_l0_71",
            "targ_layer": 8,
            "feature_idx": 931,
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def format_llm_prompt(prompt: str, tokenizer: AutoTokenizer) -> str:
    """Format the prompt according to Transformers instruction format."""
    chat = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def load_prompt_files(config: EvalConfig) -> str:
    """Load and process prompt files."""
    files: dict[str, str] = {
        "prompt": f"{config.prompt_folder}/{config.prompt_filename}",
        "docs": f"{config.prompt_folder}/{config.docs_filename}",
    }

    content: dict[str, str] = {}
    for key, filename in files.items():
        with open(filename, "r") as f:
            content[key] = f.read()

    # Format the prompt
    prompt = content["prompt"].replace("{documentation}", content["docs"])

    return prompt


def extract_python(response: str, verbose: bool = True) -> Optional[str]:
    # Regex pattern to match python block
    pattern = r"```python\s*(.*?)\s*```"

    # Search for the pattern
    match = re.search(pattern, response, re.DOTALL)

    if match:
        python_str = match.group(1)
        return python_str
    else:
        if verbose:
            print("WARNING: No python block found")
        return None


def check_for_re_usage(code_snippet: str) -> bool:
    """
    Checks if any re module function is used in a given code snippet.

    This is pretty basic and may not catch all cases.
    """
    # Define a pattern that matches common re module functions
    pattern = r"\bre\.(match|search|sub|findall|finditer|split|compile|fullmatch|escape|subn)\b"

    # Search for any of these patterns in the code snippet
    return bool(re.search(pattern, code_snippet))


def print_generations(generations: list[str], prompt: str, prefill: str) -> None:
    """Print the generated texts, removing the prompt prefix."""
    for i, generation in enumerate(generations):
        if prompt in generation:
            generation = generation[len(prompt) - len(prefill) :]
        print(f"Generation {i}:")
        print(generation)
        print()


def is_syntactically_valid_python(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a string contains syntactically valid Python code.

    Args:
        code: String containing Python code to validate

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Parsing error: {str(e)}"


@contextmanager
def restricted_compile_environment():
    """
    Context manager that provides a restricted environment for code compilation.
    Temporarily replaces stdout/stderr and restricts builtins to common safe operations.
    """
    # Save original stdout/stderr and builtins
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_builtins = dict(builtins.__dict__)

    # Create string buffers for capturing output
    temp_stdout = StringIO()
    temp_stderr = StringIO()

    # Define safe exception types
    safe_exceptions = {
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "AttributeError": AttributeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "AssertionError": AssertionError,
        "NotImplementedError": NotImplementedError,
        "ZeroDivisionError": ZeroDivisionError,
    }

    # Expanded set of safe builtins
    safe_builtins = {
        # Constants
        "None": None,
        "False": False,
        "True": True,
        # Basic types and operations
        "abs": abs,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "len": len,
        "type": type,
        "repr": repr,
        # Collections
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "frozenset": frozenset,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "reversed": reversed,
        # Type checking
        "isinstance": isinstance,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "getattr": getattr,
        # Math operations
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "pow": pow,
        # String operations
        "chr": chr,
        "ord": ord,
        "format": format,
        # Itertools functions
        "filter": filter,
        "map": map,
        # Other safe operations
        "print": print,  # Captured by StringIO
        "sorted": sorted,
        "any": any,
        "all": all,
        "iter": iter,
        "next": next,
        "slice": slice,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        # Exception handling
        "try": "try",
        "except": "except",
        "finally": "finally",
        **safe_exceptions,  # Add all safe exception types
    }

    try:
        # Replace stdout/stderr
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

        # Restrict builtins
        for key in list(builtins.__dict__.keys()):
            if key not in safe_builtins:
                del builtins.__dict__[key]

        # Add exception types to the builtins
        builtins.__dict__.update(safe_exceptions)

        yield temp_stdout, temp_stderr

    finally:
        # Restore original environment
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        builtins.__dict__.clear()
        builtins.__dict__.update(original_builtins)


def is_semantically_valid_python(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a string contains semantically valid Python code by:
    1. Checking syntax
    2. Verifying it contains actual code structure
    3. Attempting to compile and validate basic execution

    Args:
        code: String containing Python code to validate

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    # First check syntax
    syntax_valid, syntax_error = is_syntactically_valid_python(code)
    if not syntax_valid:
        return False, syntax_error

    # Basic content validation
    code = code.strip()
    if not code:
        return False, "Empty code string"

    # Check for basic code structure (must have at least one function or class definition)
    if not any(keyword in code for keyword in ["def ", "class "]):
        return False, "No function or class definitions found"

    # Check for excessive non-ASCII characters that aren't in strings/comments
    code_lines = code.split("\n")
    invalid_lines = 0
    total_lines = len(code_lines)

    for line in code_lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            total_lines -= 1
            continue

        # Count lines with too many non-ASCII characters
        non_ascii_count = len([c for c in line if ord(c) > 127])
        if non_ascii_count / len(line) > 0.3:  # More than 30% non-ASCII
            invalid_lines += 1

    # If more than 20% of non-empty, non-comment lines are invalid
    if total_lines > 0 and (invalid_lines / total_lines) > 0.2:
        return False, "Code contains too many non-ASCII characters"

    try:
        # Try to compile the code
        try:
            compiled_code = compile(code, "<string>", "exec")
        except Exception as e:
            return False, f"Compilation error: {str(e)}"

        # Create a restricted globals dict with common built-ins
        # restricted_globals = {
        #     '__builtins__': {
        #         name: getattr(builtins, name)
        #         for name in [
        #             'len', 'int', 'str', 'list', 'dict', 'set', 'tuple',
        #             'min', 'max', 'True', 'False', 'None', 'type',
        #             'isinstance', 'print', 'range', 'compile', 'exec',
        #             "import",
        #         ]
        #     }
        # }

        # Try to execute in the restricted environment
        try:
            exec(compiled_code, {}, {})
        except Exception as e:
            # Some errors are acceptable for valid code
            error_str = str(e)
            acceptable_errors = [
                "name 'pytest' is not defined",
                "name 're' is not defined",
                "name 'random' is not defined",
                "name 'time' is not defined",
                "name 'asyncio' is not defined",
                "name 'typing' is not defined",
                "name 'Optional' is not defined",
                "name 'List' is not defined",
                "name 'Dict' is not defined",
                "name 'Any' is not defined",
                "name 'Union' is not defined",
            ]
            if not any(err in error_str for err in acceptable_errors):
                return False, f"Execution error: {error_str}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_func_name(func_code: str) -> str:
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", func_code)
    if match:
        func_name = match.group(1)
    else:
        return None
    return func_name


def validate_llm_response(
    func_code: str, llm_code: str, timeout: float = 3.0, verbose: bool = False
) -> bool:
    """
    Validates whether the LLM-generated code runs, calls the specified function,
    and doesn't raise errors within the given timeout.
    """

    func_name = get_func_name(func_code)

    def exec_code(function_called):
        # List of dangerous modules to block
        dangerous_modules = {
            "os",
            "subprocess",
            "sys",
            "socket",
            "requests",
            "urllib",
            "ftplib",
            "telnetlib",
            "smtplib",
            "pathlib",
            "shutil",
        }

        def safe_import(name, *args, **kwargs):
            if name in dangerous_modules:
                raise ImportError(f"Import of {name} is not allowed for security reasons")
            return __import__(name, *args, **kwargs)

        # Create safe globals with all built-ins
        safe_globals = {}
        for name in dir(builtins):
            safe_globals[name] = getattr(builtins, name)

        safe_globals["__import__"] = safe_import
        safe_globals["__builtins__"] = safe_globals

        # Add commonly needed modules
        safe_globals.update(
            {
                "re": re,
                "random": random,
                "__name__": "__main__",
            }
        )

        # Execute the function definition
        try:
            exec(func_code, safe_globals)
            if func_name not in safe_globals:
                function_called["error"] = f"Function {func_name} was not properly defined."
                return
        except Exception as e:
            function_called["error"] = f"Error in function definition: {str(e)}"
            return

        # Store the original function and create wrapper
        original_func = safe_globals[func_name]

        def wrapper(*args, **kwargs):
            function_called["called"] = True
            return original_func(*args, **kwargs)

        safe_globals[func_name] = wrapper

        # Execute the test code
        try:
            exec(llm_code, safe_globals)
        except Exception as e:
            function_called["error"] = f"Error in test execution: {str(e)}"
            return

    # Shared dictionary for results
    manager = Manager()
    function_called = manager.dict()
    function_called["called"] = False

    # Run in separate process with timeout
    p = Process(target=exec_code, args=(function_called,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print("Execution timed out.")
    else:
        if "error" in function_called:
            if verbose:
                print(f"An error occurred: {function_called['error']}")
        elif not function_called["called"]:
            if verbose:
                print(f"{func_name}() was not called.")
        else:
            if verbose:
                print(f"{func_name}() was successfully called.")
            return True
    return False


def validate_single_llm_response(prompt: str, response: str, verbose: bool = True) -> bool:
    # Regex pattern to match python block
    pattern = r"```python\s*(.*?)\s*```"

    # Search for the pattern
    matches = re.findall(pattern, prompt, re.DOTALL)

    original_code = matches[-1]

    llm_python = extract_python(response, verbose=False)

    if llm_python is None:
        return False

    func_name = get_func_name(llm_python)

    if func_name is None:
        return False

    llm_python = llm_python + f"\n\n{func_name}()"
    valid_code = validate_llm_response(original_code, llm_python)

    return valid_code


def validate_all_llm_responses(
    data: dict, intervention_method: str, code_id: str, scale: int
) -> tuple[float, float]:
    prompt = data[intervention_method]["code_results"][code_id]["prompt"]

    # Regex pattern to match python block
    pattern = r"```python\s*(.*?)\s*```"

    # Search for the pattern
    matches = re.findall(pattern, prompt, re.DOTALL)

    original_code = matches[-1]

    total = 0
    valid = 0
    syntactically_valid = 0

    for response in data[intervention_method]["code_results"][code_id]["generations"][scale]:
        total += 1

        llm_python = extract_python(response, verbose=False)

        if llm_python is None:
            continue

        func_name = get_func_name(llm_python)

        if func_name is None:
            continue

        syntactically_valid += is_syntactically_valid_python(llm_python)[0]

        llm_python = llm_python + f"\n\n{func_name}()"
        valid_code = validate_llm_response(original_code, llm_python)

        if valid_code:
            valid += 1

    return (valid / total), (syntactically_valid / total)
