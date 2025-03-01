import os
import json
import sys
import subprocess
import time
import re
import argparse
import numpy as np
import ollama

# GLOBAL SETTINGS - Now configurable via command line arguments
DEFAULT_LLM = "qwen2.5-coder:7b"
DEFAULT_MAX_ITERATIONS = 5  # Increased from 3 to 5 for better results
DEFAULT_EVAL_DIR = "data/training"
DEFAULT_VERBOSE = True
DEFAULT_OUTPUT_DIR = "results"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple ARC Task Solver using Ollama')
    parser.add_argument('--llm', type=str, default=DEFAULT_LLM,
                        help=f'LLM model to use for code generation (default: {DEFAULT_LLM})')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f'Maximum number of fix iterations allowed (default: {DEFAULT_MAX_ITERATIONS})')
    parser.add_argument('--eval-dir', type=str, default=DEFAULT_EVAL_DIR,
                        help=f'Directory where evaluation ARC task files are stored (default: {DEFAULT_EVAL_DIR})')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                        help=f'Print detailed progress information (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--task-index', type=int, default=1,
                        help='Index of the task to test (1-based) (default: 1)')
    parser.add_argument('--task-file', type=str, default=None,
                        help='Specific task file to use (overrides task-index)')
    parser.add_argument('--hint', type=str, default='',
                        help='Hint to provide to the LLM')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434',
                        help='Ollama API host (default: http://localhost:11434)')
    parser.add_argument('--all-tasks', action='store_true', default=False,
                        help='Process all tasks in the eval directory')
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


def select_task_file(files, directory, task_index=None, verbose=DEFAULT_VERBOSE):
    """Selects a task file either by index or by prompting the user."""
    if task_index is not None:
        if task_index < 1 or task_index > len(files):
            print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
            sys.exit(1)
        chosen_file = os.path.join(directory, files[task_index - 1])
        log(f"Selected file by index: {chosen_file}", verbose)
        return chosen_file

    # For batch workloads, if no index provided, just select the first task
    chosen_file = os.path.join(directory, files[0])
    log(f"No task index specified, selecting first file: {chosen_file}", verbose)
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
    import re

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

    # Count number of training examples
    num_examples = len(task['train'])
    analysis.append(f"Task has {num_examples} training examples.")

    # Analyze grid dimensions
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        output_grid = example['output']
        input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
        output_dims = f"{len(output_grid)}×{len(output_grid[0])}"
        analysis.append(f"Example {i + 1}: Input grid {input_dims}, Output grid {output_dims}")

        # Count colors/symbols used in the grids
        input_colors = set(item for row in input_grid for item in row)
        output_colors = set(item for row in output_grid for item in row)
        analysis.append(f"  - Input colors: {sorted(input_colors)}")
        analysis.append(f"  - Output colors: {sorted(output_colors)}")

        # Special analysis: look for pattern type
        # Is output grid larger?
        if len(output_grid) > len(input_grid) or len(output_grid[0]) > len(input_grid[0]):
            analysis.append(f"  - Note: Output grid is larger than input grid")

            # Is it a repeating pattern?
            if len(output_grid) % len(input_grid) == 0 and len(output_grid[0]) % len(input_grid[0]) == 0:
                h_repeats = len(output_grid) // len(input_grid)
                w_repeats = len(output_grid[0]) // len(input_grid[0])
                if h_repeats > 1 or w_repeats > 1:
                    analysis.append(f"  - Potential repeating pattern: {h_repeats}×{w_repeats} repetitions")

        # Check for specific patterns
        # Row wise repetition
        repeated_rows = True
        for j in range(1, len(output_grid)):
            if j % len(input_grid) == 0 and not np.array_equal(output_grid[j], output_grid[0]):
                repeated_rows = False
                break
        if repeated_rows and len(output_grid) > len(input_grid):
            analysis.append(f"  - Row-wise repetition pattern detected")

        # Column wise repetition
        if len(output_grid[0]) > len(input_grid[0]):
            repeated_cols = True
            for row in output_grid:
                for j in range(1, len(output_grid[0])):
                    if j % len(input_grid[0]) == 0 and row[j] != row[0]:
                        repeated_cols = False
                        break
            if repeated_cols:
                analysis.append(f"  - Column-wise repetition pattern detected")

    return "\n".join(analysis)


def generate_solution_code(task, ollama_client, args, iteration=0, errors=None, past_outputs=None):
    """Sends the ARC task to the LLM via Ollama and gets the generated Python code."""
    import re

    # In batch mode, only use hints from command line arguments
    hint = args.hint

    # Generate task analysis to help the model
    task_analysis = analyze_task(task)

    # Create examples formatted more visually
    example_text = []
    for i, example in enumerate(task['train']):
        input_str = "\n".join([" ".join(map(str, row)) for row in example['input']])
        output_str = "\n".join([" ".join(map(str, row)) for row in example['output']])

        example_text.append(f"Example {i + 1}:\nInput:\n{input_str}\n\nOutput:\n{output_str}")

    examples_formatted = "\n\n".join(example_text)

    # Base prompt
    prompt = (
        "# ARC Challenge Task\n\n"
        "You are given examples of input and output grids from the Abstraction and Reasoning Corpus (ARC). "
        "Your task is to figure out the transformation rule and implement it in Python.\n\n"
        f"## Task Analysis\n{task_analysis}\n\n"
        f"## Training Examples (Formatted)\n{examples_formatted}\n\n"
        f"## Training Examples (JSON)\n{json.dumps(task['train'], indent=2)}\n\n"
    )

    # Add error information if available
    if errors and iteration > 0:
        prompt += f"## Previous Errors\n{errors}\n\n"

    # Add hint if provided
    if hint:
        prompt += f"## Hint\n{hint}\n\n"

    # Final instructions
    prompt += (
        "## Instructions\n"
        "1. Write a Python function that implements the transformation rule\n"
        "2. Your solution should read a JSON grid from stdin and output the transformed grid as JSON\n"
        "3. Focus on identifying patterns like: rotations, reflections, translations, color changes, etc.\n"
        "4. Make your code robust to handle different grid sizes if appropriate\n"
        "5. Provide ONLY executable Python code with no explanations (I will run your code directly)\n\n"
        "## Solution (Python)"
    )

    log(f"Sending prompt to LLM: {args.llm}", args.verbose)
    # Only log the full prompt on first iteration to reduce verbosity
    if args.verbose:
        if iteration == 0:
            log(prompt)
        else:
            # Only log what's changed in subsequent iterations
            log("Prompt changes for this iteration:")
            if errors:
                log(f"## Previous Errors\n{errors}")
            if hint and iteration == 0:
                log(f"## Hint\n{hint}")

    try:
        # Set the Ollama host
        ollama.host = args.ollama_host

        # Get the response from Ollama
        stream = ollama.generate(model=args.llm, prompt=prompt, stream=True)
        full_response = ""
        log(f"Streaming output from LLM:", args.verbose)

        # Iterate over streaming tokens
        for chunk in stream:
            token = chunk.get("response", "")
            if token:
                if args.verbose:
                    print(token, end="", flush=True)
                full_response += token

        full_response = full_response.strip()
        if not full_response:
            raise ValueError("No code output received from LLM.")

        log(f"\nFull streamed output received", args.verbose)

        # Clean the generated code
        code = clean_generated_code(full_response)
        log(f"Cleaned code:", args.verbose)
        if args.verbose:
            log(code)

        return code
    except Exception as e:
        print(f"Error during code generation: {e}")
        sys.exit(1)


def write_generated_code(code, filename="generated_solution.py", verbose=DEFAULT_VERBOSE):
    """Writes the generated code to a file."""
    with open(filename, "w") as f:
        f.write(code)
    log(f"Code written to {filename}", verbose)
    return filename


def run_generated_code(filename="generated_solution.py", input_data=None, verbose=DEFAULT_VERBOSE):
    """
    Creates a wrapper script that imports the generated solution and calls the solve function directly.
    Returns a tuple (stdout, stderr).
    """
    # Create a temporary wrapper script
    wrapper_filename = filename.replace('.py', '_wrapper.py')

    with open(wrapper_filename, 'w') as f:
        f.write("""
import sys
import json
from importlib.util import spec_from_file_location, module_from_spec

# Import the generated solution module
spec = spec_from_file_location("solution", "{}")
solution = module_from_spec(spec)
spec.loader.exec_module(solution)

# Read input grid directly from stdin as Python literal
try:
    # Get the input
    input_str = sys.stdin.read()

    # Convert input to grid using eval (safe in this controlled environment)
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
    if hasattr(solution, 'solve'):
        result = solution.solve(grid)

        # Ensure the result is properly formatted for comparison
        if result is not None:
            print(repr(result))  # Print as Python literal for more reliable parsing
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
""".format(filename))

    log(f"Created wrapper script: {wrapper_filename}", verbose)
    log(f"Running code with input: {input_data}", verbose)

    try:
        proc = subprocess.run(
            [sys.executable, wrapper_filename],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30  # adjust timeout as needed
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


def test_solution(task, filename="generated_solution.py", verbose=DEFAULT_VERBOSE):
    """
    Runs the generated solution code against the test examples provided in the ARC task.
    Returns a list of dictionaries with test results.
    """
    test_results = []
    test_examples = task.get("test", [])
    if not test_examples:
        print("Warning: No test examples found in the ARC task.")
        return test_results

    log(f"Running {len(test_examples)} test examples...", verbose)
    for idx, example in enumerate(test_examples, start=1):
        # Prepare the input data as Python list literal for more reliable parsing
        input_grid = example.get("input")
        input_str = str(input_grid)
        expected_output = example.get("output")

        stdout, stderr = run_generated_code(filename=filename, input_data=input_str, verbose=verbose)

        # Try to parse the actual output
        actual_output = None
        if stdout and stdout.strip():
            try:
                # First try to parse as JSON
                actual_output = json.loads(stdout.strip())
            except json.JSONDecodeError:
                try:
                    # Then try to evaluate as a Python literal
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

        log(f"Test Example {idx}: {'PASSED' if result['passed'] else 'FAILED'}", verbose)
        test_results.append(result)
    return test_results


def print_test_results(test_results):
    """
    Prints the results of the tests in text format.
    """
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)

    print(f"\n--- Test Results: {passed_count}/{total_count} passed ---")

    for result in test_results:
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"\nTest Example {result['example_number']}: {status}")

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
            print("\nActual Output: No valid JSON output")

        # Print any errors
        if result["error"]:
            print("\nError:")
            print(result["error"])


def save_results(task_file, test_results, code, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Saves the test results and generated code to files in the output directory.
    Returns the base filename used (without extension).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract task name from the file path
    task_name = os.path.splitext(os.path.basename(task_file))[0]

    # Save the test results to a JSON file
    results_file = os.path.join(output_dir, f"{task_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to {results_file}")

    # Save the generated code to a Python file
    code_file = os.path.join(output_dir, f"{task_name}_solution.py")
    with open(code_file, 'w') as f:
        f.write(code)
    print(f"Generated code saved to {code_file}")

    return task_name


def process_task(task_file, args):
    """Process a single ARC task file"""
    task = load_arc_task(task_file, args.verbose)

    # Extract the task name for saving results
    task_name = os.path.splitext(os.path.basename(task_file))[0]

    # Create task specific output directory
    task_output_dir = os.path.join(args.output_dir, f"task{args.task_index}")
    os.makedirs(task_output_dir, exist_ok=True)

    # Log task being processed
    print(f"\n==================================================")
    print(f"Processing task: {task_name}")
    print(f"==================================================")

    # Run iteration loop
    iteration = 0
    code_file = os.path.join(task_output_dir, f"{task_name}_solution_temp.py")
    errors = None

    # Generate initial solution
    current_code = generate_solution_code(task, ollama, args)

    while iteration < args.max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")
        write_generated_code(current_code, code_file, args.verbose)

        # Run the code with a simple test case first
        test_input = "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"
        stdout, stderr = run_generated_code(code_file, input_data=test_input, verbose=args.verbose)

        if stderr and ("Error" in stderr or "error" in stderr or "Traceback" in stderr):
            print("Error encountered during execution:")
            print(stderr)
            errors = stderr
            iteration += 1
            if iteration >= args.max_iterations:
                break

            print(f"\nAttempting to fix errors (iteration {iteration + 1})")
            current_code = generate_solution_code(task, ollama, args, iteration=iteration, errors=errors)
        else:
            # Code ran without immediate errors; now test against the test examples
            test_results = test_solution(task, code_file, args.verbose)
            if test_results:
                all_passed = all(tr["passed"] for tr in test_results)
                print_test_results(test_results)

                if all_passed:
                    print("\n🎉 All tests passed!")
                    # Save the results
                    save_results(task_file, test_results, current_code, task_output_dir)
                    return True  # Success
                else:
                    print("\n❌ Some tests failed.")

                    # Prepare detailed error feedback for the model
                    feedback = "Test results summary:\n"
                    for tr in test_results:
                        status = "PASSED" if tr["passed"] else "FAILED"
                        feedback += f"- Example {tr['example_number']}: {status}\n"

                        # Add specific error information if available
                        if not tr["passed"]:
                            if tr["error"]:
                                feedback += f"  Error: {tr['error']}\n"
                            elif tr["actual_output"] is not None:
                                try:
                                    # Show input and expected output dimensions
                                    input_shape = np.array(tr["input"]).shape
                                    expected_shape = np.array(tr["expected_output"]).shape

                                    # Try to get actual output shape, but handle non-array outputs
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

                                    # Add a hint about the relationship between input and output shapes
                                    if input_shape and expected_shape:
                                        input_rows, input_cols = input_shape
                                        output_rows, output_cols = expected_shape

                                        # Add specific transformation hints based on the pattern
                                        if output_rows == input_rows * 3 and output_cols == input_cols * 3:
                                            feedback += "  Hint: The output grid is 3x3 times larger than the input grid\n"

                                        # Compare actual vs expected shapes
                                        if actual_shape != "non-array" and len(actual_shape) == 2:
                                            actual_rows, actual_cols = actual_shape
                                            if output_rows != actual_rows or output_cols != actual_cols:
                                                feedback += f"  Hint: Your solution produced a {actual_rows}x{actual_cols} grid, but the expected output is {output_rows}x{output_cols}\n"
                                except Exception as e:
                                    feedback += f"  Error analyzing shapes: {str(e)}\n"
                            else:
                                feedback += f"  No valid output was produced\n"

                    errors = feedback
                    iteration += 1
                    if iteration >= args.max_iterations:
                        break

                    print(f"\nAttempting to fix failed tests (iteration {iteration + 1})")
                    current_code = generate_solution_code(task, ollama, args, iteration=iteration, errors=errors)
            else:
                print("No test examples available. Here is the output from the code:")
                print(stdout)
                return True  # Consider as success since we can't test

    # Save final results after reaching max iterations
    print(f"Reached maximum iterations ({args.max_iterations}). Saving the last result.")
    test_results = test_solution(task, code_file, args.verbose)
    save_results(task_file, test_results, current_code, task_output_dir)
    print_test_results(test_results)

    # Final notification for this task
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)
    print(f"\nFinal score for {task_name}: {passed_count}/{total_count} tests passed")

    return False  # Not fully successful


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the Ollama client
    ollama.host = args.ollama_host

    # Get list of task files
    files = list_task_files(args.eval_dir)

    # Create a summary file for results
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"ARC Task Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.llm}\n")
        f.write(f"Max iterations: {args.max_iterations}\n\n")

    # Determine which task(s) to process
    if args.task_file:
        # Process specific file
        task_file = os.path.join(args.eval_dir, args.task_file)
        if not os.path.exists(task_file):
            print(f"Error: Task file '{task_file}' not found.")
            sys.exit(1)
        tasks_to_process = [task_file]
    elif args.all_tasks:
        # Process all tasks
        tasks_to_process = [os.path.join(args.eval_dir, f) for f in files]
    else:
        # Process single task by index
        if args.task_index < 1 or args.task_index > len(files):
            print(f"Error: Task index {args.task_index} is out of range (1-{len(files)}).")
            sys.exit(1)
        tasks_to_process = [os.path.join(args.eval_dir, files[args.task_index - 1])]

    # Process each task
    successful = 0
    total = len(tasks_to_process)

    print(f"Processing {total} task(s)...")

    for i, task_file in enumerate(tasks_to_process, 1):
        task_name = os.path.splitext(os.path.basename(task_file))[0]
        print(f"\nTask {i}/{total}: {task_name}")

        success = process_task(task_file, args)
        if success:
            successful += 1

        # Update summary file
        with open(summary_file, "a") as f:
            f.write(f"{task_name}: {'SUCCESS' if success else 'FAILED'}\n")

    # Final summary
    print(f"\n==================================================")
    print(f"Processing complete: {successful}/{total} tasks solved successfully")
    print(f"Results saved to {args.output_dir}/")
    print(f"Summary saved to {summary_file}")
    print(f"==================================================")

    # Write final summary to summary file
    with open(summary_file, "a") as f:
        f.write(f"\nFinal Score: {successful}/{total} tasks solved successfully\n")


if __name__ == "__main__":
    main()