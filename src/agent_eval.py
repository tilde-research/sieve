from typing import Optional, Tuple, Dict, Any, List
from openai import OpenAI
from dataclasses import dataclass
import src.utils as utils
import json
from tqdm import tqdm
import time
import os

@dataclass
class CodeEvaluation:
    """Stores the evaluation results for a code snippet"""
    is_syntactically_valid: bool
    is_semantically_valid: bool
    uses_regex: bool
    executes_successfully: bool
    follows_prompt: bool
    explanation: str
    properties: Dict[str, Any]

def evaluate_code_with_gpt4(code: str, prompt: str, client: OpenAI) -> Tuple[bool, bool, str, Dict[str, Any]]:
    """
    Evaluate code using GPT-4 to determine if it successfully executes the prompt
    and analyze its properties.
    """
    system_prompt = """You are an expert Python programmer evaluating code solutions.
    Analyze the given code and determine:
    1. Whether it successfully implements the requirements from the prompt
    2. Whether it would execute successfully
    3. Key properties of the implementation
    
    Provide your response in JSON format with the following fields:
    {
        "executes_successfully": bool,
        "follows_prompt": bool,
        "explanation": str,
        "properties": {
            "uses_list_comprehension": bool,
            "uses_error_handling": bool,
            "is_efficient": bool,
            "is_readable": bool,
        }
    }
    """
    
    user_message = f"""
    Original Prompt:
    {prompt}
    
    Code to Evaluate:    ```python
    {code}    ```
    
    Evaluate this code and provide your analysis in the requested JSON format.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        if result is None:
            return False, False, "No response from GPT-4", {}
            
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            try:
                parsed_result = eval(result)
            except Exception:
                return False, False, f"Failed to parse response: {result}", {}
        
        return (
            parsed_result["executes_successfully"],
            parsed_result["follows_prompt"],
            parsed_result["explanation"],
            parsed_result["properties"]
        )
        
    except Exception as e:
        return False, False, f"GPT-4 evaluation failed: {str(e)}", {}

def evaluate_code(
    code: str,
    prompt: str,
    client: OpenAI
) -> CodeEvaluation:
    """
    Comprehensive evaluation of a code snippet using both automated checks
    and GPT-4 analysis.
    """
    # Run automated checks
    is_syntactically_valid, _ = utils.is_syntactically_valid_python(code)
    is_semantically_valid, _ = utils.is_semantically_valid_python(code)
    uses_regex = utils.check_for_re_usage(code)
    
    # Get GPT-4 evaluation
    executes_successfully, follows_prompt, explanation, properties = (
        evaluate_code_with_gpt4(code, prompt, client)
    )
    
    return CodeEvaluation(
        is_syntactically_valid=is_syntactically_valid,
        is_semantically_valid=is_semantically_valid,
        uses_regex=uses_regex,
        executes_successfully=executes_successfully,
        follows_prompt=follows_prompt,
        explanation=explanation,
        properties=properties
    )

def batch_evaluate_generations(
    generations: list[str],
    prompt: str,
    api_key: Optional[str] = None,
    batch_size: int = 10,
    retry_delay: float = 1.0,
    max_retries: int = 3
) -> list[CodeEvaluation]:
    """
    Evaluate multiple code generations in batches with retry logic.
    
    Args:
        generations: List of generated code snippets
        prompt: The original prompt that requested the code
        api_key: Optional OpenAI API key
        batch_size: Number of evaluations to process in parallel
        retry_delay: Delay between retries in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of CodeEvaluation objects
    """
    client = OpenAI(
        api_key=api_key,
    ) if api_key else OpenAI()
    
    results = []
    valid_codes = []
    
    # First extract all valid Python code
    for code in generations:
        python_code = utils.extract_python(code)
        if python_code is not None:
            valid_codes.append(python_code)
    
    # Process in batches
    for i, code in tqdm(enumerate(valid_codes), total=len(valid_codes), desc="Evaluating code batches"):

        retries = 0
        while retries < max_retries:
            try:
                evaluation = evaluate_code(code, prompt, client)
                results.append(evaluation)
                break
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    print(f"Failed to evaluate code after {max_retries} attempts: {e}")
                    continue
                time.sleep(retry_delay)
    
        
        # Small delay between batches to avoid rate limits
        time.sleep(0.01)
    
    return results

def summarize_evaluations(evaluations: list[CodeEvaluation]) -> Dict[str, Any]:
    """
    Summarize the results of multiple code evaluations.
    
    Args:
        evaluations: List of CodeEvaluation objects
        
    Returns:
        Dictionary containing summary statistics
    """
    total = len(evaluations)
    if total == 0:
        return {"error": "No evaluations to summarize"}
        
    summary = {
        "total_samples": total,
        "syntactically_valid": sum(1 for e in evaluations if e.is_syntactically_valid) / total,
        "semantically_valid": sum(1 for e in evaluations if e.is_semantically_valid) / total,
        "uses_regex": sum(1 for e in evaluations if e.uses_regex) / total,
        "executes_successfully": sum(1 for e in evaluations if e.executes_successfully) / total,
        "follows_prompt": sum(1 for e in evaluations if e.follows_prompt) / total,
        "property_stats": {
            "uses_list_comprehension": sum(1 for e in evaluations if e.properties.get("uses_list_comprehension", False)) / total,
            "uses_error_handling": sum(1 for e in evaluations if e.properties.get("uses_error_handling", False)) / total,
            "is_efficient": sum(1 for e in evaluations if e.properties.get("is_efficient", False)) / total,
            "is_readable": sum(1 for e in evaluations if e.properties.get("is_readable", False)) / total,
        },
        "complexity_distribution": {}
    }
    
    # Count complexity distributions
    complexity_counts = {}
    for eval in evaluations:
        complexity = eval.properties.get("complexity", "unknown")
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    summary["complexity_distribution"] = {
        k: v/total for k, v in complexity_counts.items()
    }
    
    return summary

# Example usage
if __name__ == "__main__":
    code = """
    def extract_numbers(text):
        import re
        return [int(num) for num in re.findall(r'\d+', text)]
    """
    
    prompt = "Write a function that extracts all numbers from a text string using regex"
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize client once
    client = OpenAI(api_key=api_key)
    
    # Single evaluation
    evaluation = evaluate_code(code, prompt, client)
    print(f"Evaluation results:\n{evaluation}")
    
    # Batch evaluation
    generations = [code, code]  # Example with duplicate code
    evaluations = batch_evaluate_generations(generations, prompt, api_key)
    summary = summarize_evaluations(evaluations)
    print(f"\nSummary of evaluations:\n{summary}") 