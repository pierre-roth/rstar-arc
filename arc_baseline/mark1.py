import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from global_config import *
from vllm import LLM

import re
import json
import subprocess
import argparse
import platform
import numpy as np
import warnings

# Suppress PyTorch/TF warnings that aren't helpful
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_MAX_ITERATIONS = 3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple ARC Task Solver')
    parser.add_argument('--model', type=str, default=DEFAULT_POLICY_LLM,
                        help=f'LLM model to use for code generation (default: {DEFAULT_POLICY_LLM})')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f'Maximum number of fix iterations allowed (default: {DEFAULT_MAX_ITERATIONS})')
    parser.add_argument('--eval', action='store_true', default=False,
                        help=f'Evaluation tasks instead of training? (default: {False})')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                        help=f'Print detailed progress information (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--task-index', type=int, default=1,
                        help='Index of the task to test (1-based) (default: 1)')
    parser.add_argument('--task-file', type=str, default=None,
                        help='Specific task file to use (overrides task-index)')
    parser.add_argument('--hint', type=str, default='',
                        help='Hint to provide to the LLM')
    parser.add_argument('--all-tasks', action='store_true', default=False,
                        help='Process all tasks in the data_sample/[training or evaluation] directory')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use for the LLM')
    parser.add_argument('--output-dir', type=str, default=os.path.join(OUTPUT_BASE_PATH, "arc_results"),
                        help='Directory to store any output files')
    return parser.parse_args()


def log(msg, verbose=DEFAULT_VERBOSE):
    """Log a message if verbose mode is enabled."""
    if verbose:
        print(msg)


def list_task_files(directory):
    """Lists all JSON files in the given directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please check your ARC data directory.")
        sys.exit(1)
    files = sorted([f for f in os.listdir(directory) if f.endswith('.json')], key=lambda x: x.lower())
    if not files:
        print(f"No JSON files found in directory '{directory}'.")
        sys.exit(1)
    log(f"Found {len(files)} JSON files in '{directory}'")
    return files


def select_task_file(files, directory, task_index, verbose=DEFAULT_VERBOSE):
    """Selects a task file either by index or by prompting the user."""
    if task_index < 1 or task_index > len(files):
        print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
        sys.exit(1)
    chosen_file = os.path.join(directory, files[task_index - 1])
    log(f"Selected file by index: {chosen_file}", verbose)
    return chosen_file


def load_arc_task(file_path, verbose=DEFAULT_VERBOSE):
    """Loads an ARC task file (contains 'train' and 'test' keys)."""
    log(f"Loading ARC task from '{file_path}'...", verbose)
    try:
        with open(file_path, "r") as f:
            task = json.load(f)
        if "train" not in task or "test" not in task:
            raise ValueError("The JSON file must contain 'train' and 'test' keys.")
        log(f"Successfully loaded ARC task with {len(task['train'])} training and {len(task['test'])} test examples.",
            verbose)
        return task
    except Exception as e:
        print(f"Error loading task file {file_path}: {e}")
        sys.exit(1)


def clean_generated_code(text):
    """Extracts Python code from the model's output, handling various formats."""
    # First, try to extract a Python code block
    code_block_pattern = r"```python(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, flags=re.DOTALL)
    if code_blocks:
        # Return the longest code block
        code = max(code_blocks, key=len).strip()
        # Ensure the code has a solve function
        if not re.search(r'def\s+solve\s*\(', code):
            # Add a wrapper solve function if needed
            code = _add_solve_function_wrapper(code)
        return code

    # If no code block is found, just try to clean up the text
    cleaned = text.strip()

    # Remove markdown-style code block markers without language
    cleaned = re.sub(r"```\s*\n", "", cleaned)
    cleaned = re.sub(r"\n\s*```", "", cleaned)

    # Ensure the code has a solve function
    if not re.search(r'def\s+solve\s*\(', cleaned):
        cleaned = _add_solve_function_wrapper(cleaned)

    # Remove any JSON imports that might cause confusion
    if 'import json' in cleaned and 'from' not in cleaned:
        cleaned = re.sub(r'import\s+json\s*', '', cleaned)

    # Fix common issues with JSON handling
    if 'json.loads(sys.stdin)' in cleaned:
        cleaned = cleaned.replace('json.loads(sys.stdin)', 'json.loads(sys.stdin.read())')

    # Ensure the code has necessary imports
    if 'sys' in cleaned and 'import sys' not in cleaned:
        cleaned = f"import sys\n{cleaned}"

    return cleaned


def _add_solve_function_wrapper(code):
    """Add a solve function wrapper to code that doesn't have one."""
    # Check if there's already a transform_grid or similar function
    transform_match = re.search(r'def\s+(transform_grid|transform|convert|process)[\s\(]', code)

    if transform_match:
        function_name = transform_match.group(1)
        # Add a solve function that calls the existing transformation function
        wrapper = f"""
# Adding solve wrapper function
def solve(grid):
    return {function_name}(grid)

"""
        return wrapper + code
    else:
        # If we can't find any transformation function, wrap all the code
        # in a solve function as a best effort
        wrapper = """
def solve(grid):
    # Wrapper function added automatically
    input_grid = grid
    output_grid = []

    # Original code follows:
"""
        indented_code = "\n".join(f"    {line}" for line in code.split("\n"))
        wrapper += indented_code

        # Add a return statement if one isn't present
        if "return" not in code:
            wrapper += "\n    return output_grid"

        return wrapper


def analyze_task(task):
    """
    Analyzes the training examples to extract patterns and insights.
    Returns a string with observations about the task.
    """
    analysis = []

    analysis.append(f"\n### TRAINING ###\n")

    # Count number of training examples
    num_training_examples = len(task['train'])
    analysis.append(f"Task has {num_training_examples} training examples.")

    # Analysis
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        output_grid = example['output']
        input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
        output_dims = f"{len(output_grid)}×{len(output_grid[0])}"
        analysis.append(f"Example {i + 1}: Input grid {input_dims}, Output grid {output_dims}")

        input_colors = set(item for row in input_grid for item in row)
        output_colors = set(item for row in output_grid for item in row)
        analysis.append(f"  - Input colors: {sorted(input_colors)}")
        analysis.append(f"  - Output colors: {sorted(output_colors)}")

        analysis.append(f"Example {i + 1}:\nInput:\n{input_grid}\nOutput:\n{output_grid}\n\n")

    analysis.append(f"\n### TEST ###\n")

    num_test_examples = len(task['test'])
    analysis.append(f"Task has {num_test_examples} test examples.")

    for i, example in enumerate(task['test']):
        input_grid = example['input']
        # output hidden / not given - LLM shouldn't see test outputs
        input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
        analysis.append(f"Test Example {i + 1}: Input grid {input_dims}")

        input_colors = set(item for row in input_grid for item in row)
        analysis.append(f"  - Input colors: {sorted(input_colors)}")

        analysis.append(f"Example {i + 1}:\nInput:\n{input_grid}\n")

    return "\n".join(analysis)


def generate_solution_code(task, args, iteration=0, past=None):
    """Sends the ARC task to the LLM and gets the generated Python code."""
    # Use hint from command line arguments
    hint = args.hint

    # Generate task analysis to help the model
    task_with_analysis = analyze_task(task)

    # Base prompt
    prompt = (
        "# ARC Challenge Task\n\n"
        "You are given examples of input and output grids from the Abstraction and Reasoning Corpus (ARC). "
        "Your task is to figure out the transformation rule and implement it in Python.\n\n"
        f"## Task with Analysis\n{task_with_analysis}\n\n"
    )

    # Past tries
    if past:
        prompt += "## Past Tries\n"
        for i, output in enumerate(past, start=1):
            prompt += f"### Try {i}\n{output}\n\n"

    # Add hint if provided
    if hint:
        prompt += f"## Hint\n{hint}\n\n"

    # Final instructions
    prompt += (
        "## Instructions\n"
        "1. Write a Python function that implements the transformation rule\n"
        "2. Your solution should implement a function 'def solve(grid:list[list[int]]) -> list[list[int]]'\n"
        "3. Focus on identifying patterns like: rotations, reflections, translations, color changes, etc.\n"
        "4. Make your code robust to handle different grid sizes if appropriate\n"
        "5. Provide ONLY executable Python code with no explanations (I will run your code directly)\n\n"
        "## Solution (Python)"
    )

    log(f"Sending prompt to LLM: {args.model}", args.verbose)
    log(prompt, args.verbose)

    try:
        llm = LLM(
            model=args.model,
            download_dir=os.path.join(MODEL_BASE_PATH, "policy"),
            tensor_parallel_size=args.gpus
        )

        outputs = llm.generate(prompts=[prompt])
        full_response = outputs[0].outputs[0].text

        if not full_response:
            raise ValueError("No code output received from LLM.")

        log(f"\nFull output received", args.verbose)

        # Clean the generated code
        code = clean_generated_code(full_response)
        log(f"Cleaned code:", args.verbose)
        log(code, args.verbose)

        return code

    except Exception as e:
        print(f"Error during code generation: {e}")
        sys.exit(1)


def run_code_in_process(code_string, input_data=None, verbose=DEFAULT_VERBOSE):
    """
    Runs the generated code in another process without saving to a file.
    Returns a tuple (stdout, stderr).
    """
    # Create a wrapper that imports the code as a string and runs it
    wrapper_code = f"""
import sys
import json
import importlib.util
from io import StringIO
from types import ModuleType

# Create a module from the code string
code_string = '''
{code_string}
'''

# Create a module from the string
module = ModuleType('solution')
exec(code_string, module.__dict__)

# Read input grid
try:
    input_str = sys.stdin.read()
    grid = None
    if input_str:
        try:
            grid = eval(input_str)
        except Exception as e:
            print(f"Error parsing input: {{e}}", file=sys.stderr)
            grid = []
    else:
        grid = []

    # Call the solve function with the input grid
    if hasattr(module, 'solve'):
        result = module.solve(grid)

        # Ensure the result is properly formatted for comparison
        if result is not None:
            print(repr(result))
        else:
            print("[]")
            print("Error: Solution returned None", file=sys.stderr)
    else:
        print("[]")
        print("Error: No solve function found in the solution", file=sys.stderr)
except Exception as e:
    import traceback
    print("[]")
    print(f"Error: {{e}}\\n{{traceback.format_exc()}}", file=sys.stderr)
"""

    log(f"Running code with input: {input_data}", verbose)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper_code],
            input=input_data if input_data else "",
            capture_output=True,
            text=True,
            timeout=5  # adjust timeout as needed
        )
        log(f"Stdout: {proc.stdout}", verbose)
        log(f"Stderr: {proc.stderr}", verbose)
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        log("Execution timed out.", verbose)
        return "", "Execution timed out."
    except Exception as e:
        log(f"Exception: {str(e)}", verbose)
        return "", str(e)


def test_on_training_examples(task, code_string, verbose=DEFAULT_VERBOSE):
    """
    Tests the generated code against the training examples only.
    Returns a list of dictionaries with test results.
    """
    test_results = []
    training_examples = task.get("train", [])
    if not training_examples:
        print("Warning: No training examples found in the ARC task.")
        return test_results

    log(f"Running {len(training_examples)} training examples...", verbose)
    for idx, example in enumerate(training_examples, start=1):
        # Prepare the input data as Python list literal
        input_grid = example.get("input")
        input_str = str(input_grid)
        expected_output = example.get("output")

        stdout, stderr = run_code_in_process(code_string, input_data=input_str, verbose=verbose)

        # Try to parse the actual output
        actual_output = None
        if stdout and stdout.strip():
            try:
                # Try to evaluate as a Python literal
                actual_output = eval(stdout.strip())
            except Exception as e:
                log(f"Warning: Could not parse output: {stdout.strip()} - {str(e)}", verbose)

        result = {
            "example_number": idx,
            "input": input_grid,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "raw_output": stdout.strip(),
            "error": stderr.strip(),
            "passed": False
        }

        # Check if the test passed
        if actual_output is not None:
            try:
                # Check for equality by converting both to numpy arrays if needed
                if isinstance(actual_output, list) and isinstance(expected_output, list):
                    result["passed"] = np.array_equal(np.array(expected_output), np.array(actual_output))
                else:
                    result["passed"] = False
                    log(f"Warning: Output format mismatch. Expected a list, got {type(actual_output)}", verbose)
            except Exception as e:
                result["passed"] = False
                log(f"Error comparing outputs: {e}", verbose)

        log(f"Training Example {idx}: {'PASSED' if result['passed'] else 'FAILED'}", verbose)
        test_results.append(result)
    return test_results


def run_on_test_examples(task, code_string, verbose=DEFAULT_VERBOSE):
    """
    Runs the code on test examples only, without comparing to expected outputs.
    Returns a list of dictionaries with the generated outputs.
    """
    test_results = []
    test_examples = task.get("test", [])
    if not test_examples:
        print("Warning: No test examples found in the ARC task.")
        return test_results

    log(f"Running {len(test_examples)} test examples...", verbose)
    for idx, example in enumerate(test_examples, start=1):
        # Prepare the input data as Python list literal
        input_grid = example.get("input")
        input_str = str(input_grid)
        expected_output = example.get("output")  # For final validation only

        stdout, stderr = run_code_in_process(code_string, input_data=input_str, verbose=verbose)

        # Try to parse the actual output
        actual_output = None
        if stdout and stdout.strip():
            try:
                # Try to evaluate as a Python literal
                actual_output = eval(stdout.strip())
            except Exception as e:
                log(f"Warning: Could not parse output: {stdout.strip()} - {str(e)}", verbose)

        result = {
            "example_number": idx,
            "input": input_grid,
            "expected_output": expected_output,  # For internal comparison only
            "actual_output": actual_output,
            "raw_output": stdout.strip(),
            "error": stderr.strip(),
            "passed": False  # Will be evaluated later
        }

        # Check if valid output was produced (no errors)
        if actual_output is not None and not stderr.strip():
            # Now check if it matches the expected output (for our internal tracking)
            try:
                if isinstance(actual_output, list) and isinstance(expected_output, list):
                    result["passed"] = np.array_equal(np.array(expected_output), np.array(actual_output))
                else:
                    result["passed"] = False
            except Exception as e:
                result["passed"] = False
                log(f"Error comparing test outputs: {e}", verbose)

        log(f"Test Example {idx}: Generated output", verbose)
        test_results.append(result)
    return test_results


def print_test_results(test_results, test_type="Training"):
    """
    Prints the results of the tests in text format.
    """
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)

    print(f"\n--- {test_type} Results: {passed_count}/{total_count} passed ---")

    for result in test_results:
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"\n{test_type} Example {result['example_number']}: {status}")

        # Print input grid
        print("Input:")
        for row in result["input"]:
            print(" ".join(str(cell) for cell in row))

        # Print expected output grid
        print("\nExpected Output:")
        for row in result["expected_output"]:
            print(" ".join(str(cell) for cell in row))

        # Print actual output grid if available
        if result["actual_output"]:
            print("\nActual Output:")
            try:
                if isinstance(result["actual_output"], list) and len(result["actual_output"]) > 0:
                    for row in result["actual_output"]:
                        print(" ".join(str(cell) for cell in row))
                else:
                    print(result["actual_output"])
            except:
                print(f"Could not format actual output: {result['actual_output']}")
        else:
            print("\nActual Output: No valid output")

        # Print any errors
        if result["error"]:
            print("\nError:")
            print(result["error"])


def process_task(task_file, args):
    """Process a single ARC task file with the correct logical flow"""
    task = load_arc_task(task_file, args.verbose)

    # Extract the task name for reporting
    task_name = os.path.splitext(os.path.basename(task_file))[0]

    # Log task being processed
    print(f"\n==================================================")
    print(f"Processing task: {task_name}")
    print(f"==================================================")

    # Run iteration loop
    iteration = 0
    past_attempts = []

    # Generate initial solution
    current_code = generate_solution_code(task, args)

    while iteration < args.max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")

        # First, test on training examples
        training_results = test_on_training_examples(task, current_code, args.verbose)
        print_test_results(training_results, "Training")

        # Check if all training examples pass
        all_training_passed = all(tr["passed"] for tr in training_results)

        if all_training_passed:
            print("\n🎉 All training examples passed! Testing on test examples...")

            # Only run on test examples if all training examples pass
            test_results = run_on_test_examples(task, current_code, args.verbose)

            # Check for any execution errors in test examples
            has_test_errors = any(tr["error"] for tr in test_results)

            if has_test_errors:
                print("\n❌ Errors encountered in test examples:")
                for tr in test_results:
                    if tr["error"]:
                        print(f"- Test Example {tr['example_number']}: {tr['error']}")

                # Add to past attempts and generate new solution
                error_feedback = "\n".join([f"Test Example {tr['example_number']}: {tr['error']}"
                                            for tr in test_results if tr["error"]])
                past_attempts.append(current_code + f"\n\n# Errors on test examples:\n{error_feedback}")

                iteration += 1
                if iteration >= args.max_iterations:
                    break

                print(f"\nAttempting to fix test errors (iteration {iteration + 1})")
                current_code = generate_solution_code(task, args, iteration=iteration, past=past_attempts)
            else:
                # No errors in test examples, we're done!
                # Calculate final score (for our tracking, not shown to model)
                passed_count = sum(1 for tr in test_results if tr["passed"])
                total_count = len(test_results)

                # Don't display this to the model - just for our tracking
                print(f"\n--- Final Test Score: {passed_count}/{total_count} correct ---")

                # For each test example, show if it matched the expected output (internally)
                for result in test_results:
                    status = "CORRECT" if result["passed"] else "INCORRECT"
                    print(f"Test Example {result['example_number']}: {status}")

                return True  # Success - completed all iterations
        else:
            # Training examples failed, prepare feedback
            feedback = "Training results summary:\n"
            for tr in training_results:
                status = "PASSED" if tr["passed"] else "FAILED"
                feedback += f"- Example {tr['example_number']}: {status}\n"

                # Add specific error information
                if not tr["passed"]:
                    if tr["error"]:
                        feedback += f"  Error: {tr['error']}\n"
                    elif tr["actual_output"] is not None:
                        try:
                            # Show shapes
                            input_shape = np.array(tr["input"]).shape
                            expected_shape = np.array(tr["expected_output"]).shape
                            actual_shape = None

                            if isinstance(tr["actual_output"], list):
                                try:
                                    actual_shape = np.array(tr["actual_output"]).shape
                                except:
                                    actual_shape = "non-array"
                            else:
                                actual_shape = "non-array"

                            feedback += f"  Input shape: {input_shape}\n"
                            feedback += f"  Expected output shape: {expected_shape}\n"
                            feedback += f"  Actual output shape: {actual_shape}\n"
                        except Exception as e:
                            feedback += f"  Error analyzing shapes: {str(e)}\n"
                    else:
                        feedback += f"  No valid output was produced\n"

            # Add to past attempts and generate new solution
            past_attempts.append(current_code + f"\n\n# Errors:\n{feedback}")

            iteration += 1
            if iteration >= args.max_iterations:
                break

            print(f"\nAttempting to fix training failures (iteration {iteration + 1})")
            current_code = generate_solution_code(task, args, iteration=iteration, past=past_attempts)

    # Reached maximum iterations without success
    print(f"Reached maximum iterations ({args.max_iterations}).")

    # Run one final evaluation on test examples to see how we did
    # (this is just for our tracking, not shown to the model)
    test_results = run_on_test_examples(task, current_code, args.verbose)
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)

    print(f"\nFinal score for {task_name}: {passed_count}/{total_count} tests correct")

    return False  # Not fully successful


def main():
    # Parse command line arguments
    args = parse_args()

    # Print system info
    print(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")

    # Determine data directory based on eval flag
    data_dir = os.path.join(DATA_SAMPLE_BASE_PATH, "evaluation" if args.eval else "training")
    args.eval_dir = data_dir

    # Get list of task files
    files = list_task_files(data_dir)

    # Determine which task(s) to process
    if args.task_file:
        # Process specific file
        task_file = os.path.join(data_dir, args.task_file)
        if not os.path.exists(task_file):
            print(f"Error: Task file '{task_file}' not found.")
            sys.exit(1)
        tasks_to_process = [task_file]
    elif args.all_tasks:
        # Process all tasks
        tasks_to_process = [os.path.join(data_dir, f) for f in files]
    else:
        # Process single task by index
        if args.task_index < 1 or args.task_index > len(files):
            print(f"Error: Task index {args.task_index} is out of range (1-{len(files)}).")
            sys.exit(1)
        tasks_to_process = [os.path.join(data_dir, files[args.task_index - 1])]

    # Process each task
    successful = 0
    total = len(tasks_to_process)

    print(f"Processing {total} task(s)...")

    # Process tasks one by one for simplicity in this version
    # In future versions, this could be parallelized
    for i, task_file in enumerate(tasks_to_process, 1):
        task_name = os.path.splitext(os.path.basename(task_file))[0]
        print(f"\nTask {i}/{total}: {task_name}")

        success = process_task(task_file, args)
        if success:
            successful += 1

    # Final summary
    print(f"\n==================================================")
    print(f"Processing complete: {successful}/{total} tasks solved successfully")
    print(f"==================================================")


if __name__ == "__main__":
    main()
