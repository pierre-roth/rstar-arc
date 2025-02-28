import os
import json
import subprocess
import sys
import time
import re
import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for cluster environment
import matplotlib.pyplot as plt

import ollama  # Ensure you have installed the Python Ollama library

# GLOBAL SETTINGS - Now configurable via command line arguments
DEFAULT_LLM = "qwen2.5-coder:7b"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_EVAL_DIR = "data/training"
DEFAULT_VERBOSE = True
DEFAULT_OUTPUT_DIR = "results"  # Directory to save results and images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ARC Task Solver using Ollama')
    parser.add_argument('--llm', type=str, default=DEFAULT_LLM,
                        help=f'LLM model to use for code generation (default: {DEFAULT_LLM})')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f'Maximum number of fix iterations allowed (default: {DEFAULT_MAX_ITERATIONS})')
    parser.add_argument('--eval-dir', type=str, default=DEFAULT_EVAL_DIR,
                        help=f'Directory where evaluation ARC task files are stored (default: {DEFAULT_EVAL_DIR})')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                        help=f'Print detailed progress information (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save results and images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--task-index', type=int, default=None,
                        help='Index of the task to test (1-based). If not provided, will prompt user for selection')
    parser.add_argument('--hint', type=str, default='',
                        help='Hint to provide to the LLM')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434',
                        help='Ollama API host (default: http://localhost:11434)')
    return parser.parse_args()


def log(msg, verbose=DEFAULT_VERBOSE):
    """Log a message if verbose mode is enabled."""
    if verbose:
        print(msg)


def list_task_files(directory):
    """
    Lists all JSON files in the given directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please check your ARC data directory.")
        sys.exit(1)
    files = sorted([f for f in os.listdir(directory) if f.endswith('.json')], key=lambda x: x.lower())
    if not files:
        print(f"No JSON files found in directory '{directory}'.")
        sys.exit(1)
    log(f"[list_task_files] Found {len(files)} JSON files in '{directory}'")
    return files


def select_task_file(files, directory, task_index=None, verbose=DEFAULT_VERBOSE):
    """
    Selects a task file either by index or by prompting the user.
    Returns the full path to the selected file.
    """
    if task_index is not None:
        if task_index < 1 or task_index > len(files):
            print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
            sys.exit(1)
        chosen_file = os.path.join(directory, files[task_index - 1])
        log(f"[select_task_file] Selected file by index: {chosen_file}", verbose)
        return chosen_file

    # If no index provided, prompt the user
    print(f"Select an ARC task (1-{len(files)}):")
    for idx, file in enumerate(files, start=1):
        print(f"  {idx}: {file}")
    selection = input("Enter the number of the ARC task file you want to test: ").strip()
    try:
        index = int(selection) - 1
        if index < 0 or index >= len(files):
            raise ValueError("Selection out of range.")
    except Exception as e:
        print("Invalid input. Please enter a valid task number.")
        sys.exit(1)
    chosen_file = os.path.join(directory, files[index])
    log(f"[select_task_file] Selected file: {chosen_file}", verbose)
    return chosen_file


def load_arc_task(file_path, verbose=DEFAULT_VERBOSE):
    """
    Loads an ARC task file which should contain two keys: 'train' and 'test'.
    """
    log(f"[load_arc_task] Loading ARC task from '{file_path}'...", verbose)
    try:
        with open(file_path, "r") as f:
            task = json.load(f)
        if "train" not in task or "test" not in task:
            raise ValueError("The JSON file must contain 'train' and 'test' keys.")
        log(f"[load_arc_task] Successfully loaded ARC task with {len(task['train'])} training and {len(task['test'])} test examples.",
            verbose)
        return task
    except Exception as e:
        print(f"Error loading task file {file_path}: {e}")
        sys.exit(1)


def remove_thinking_steps(text):
    """
    Removes hidden thinking steps from the model output.
    Checks for several common formats.
    """
    # Check for <think> tags
    pattern1 = r"<think>(.*?)</think>"
    cleaned = re.sub(pattern1, "", text, flags=re.DOTALL)

    # Check for [THINKING] tags
    pattern2 = r"\[THINKING\](.*?)\[/THINKING\]"
    cleaned = re.sub(pattern2, "", cleaned, flags=re.DOTALL)

    # Try to extract just the code block if it exists
    code_block_pattern = r"```python(.*?)```"
    code_blocks = re.findall(code_block_pattern, cleaned, flags=re.DOTALL)
    if code_blocks:
        # Return the longest code block
        return max(code_blocks, key=len).strip()

    return cleaned.strip()


def generate_solution_code(task, ollama_client, args, prompt_prefix=""):
    """
    Sends the ARC task (training examples) to the LLM via Ollama and gets the generated Python code.
    This version uses streaming output: tokens are printed as they are received and accumulated.
    """
    hint = args.hint
    if not hint and not prompt_prefix:
        # Only ask for a hint if this is the initial generation, not a fix
        hint_input = input("Do you want to provide a hint to the model? (y/n): ")
        if hint_input.lower() == "y":
            hint = input("Enter a hint for the model: ")

    prompt = (
        f"{prompt_prefix}"
        f"ARC Task Training Data (in JSON): {json.dumps(task['train'])}\n\n"
        "Your task is to generate a complete Python solution that implements the transformation shown in the training examples. "
        "The solution should be self-contained, and when given an input grid (a JSON list of lists) via standard input, it should output the correct output grid (as a JSON list of lists). "
        "Do all reasoning in hidden <think>...</think> tags and output only the final Python code (with no commentary or extra text)."
    )

    if hint:
        prompt += f"\nHint: {hint}"

    log(f"[generate_solution_code] Sending prompt to LLM: {args.llm}", args.verbose)
    if args.verbose:
        log(prompt)

    try:
        # Set the Ollama host
        ollama.host = args.ollama_host

        # Enable streaming by setting stream=True
        stream = ollama.generate(model=args.llm, prompt=prompt, stream=True)
        full_code = ""
        log(f"[generate_solution_code] Streaming output from LLM:", args.verbose)
        # Iterate over streaming tokens
        for chunk in stream:
            token = chunk.get("response", "")
            if token:
                if args.verbose:
                    print(token, end="", flush=True)
                full_code += token
        full_code = full_code.strip()
        if not full_code:
            raise ValueError("No code output received from LLM.")
        log(f"\n[generate_solution_code] Full streamed output received:", args.verbose)
        if args.verbose:
            log(full_code)
        # Remove hidden thinking steps
        full_code = remove_thinking_steps(full_code)
        log(f"[generate_solution_code] Code after removing thinking steps:", args.verbose)
        if args.verbose:
            log(full_code)
        return full_code
    except Exception as e:
        print(f"Error during code generation: {e}")
        sys.exit(1)


def write_generated_code(code, filename="generated_solution.py", verbose=DEFAULT_VERBOSE):
    """
    Writes the generated code to a file.
    """
    with open(filename, "w") as f:
        f.write(code)
    log(f"[write_generated_code] Code written to {filename}", verbose)
    return filename


def run_generated_code(filename="generated_solution.py", input_data=None, verbose=DEFAULT_VERBOSE):
    """
    Runs the generated solution code as a separate process.
    Optionally sends input_data (string) to the process's stdin.
    Returns a tuple (stdout, stderr).
    """
    log(f"[run_generated_code] Running code from {filename} with input: {input_data}", verbose)
    try:
        proc = subprocess.run(
            [sys.executable, filename],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30  # adjust timeout as needed
        )
        log(f"[run_generated_code] Stdout: {proc.stdout}", verbose)
        log(f"[run_generated_code] Stderr: {proc.stderr}", verbose)
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        log("[run_generated_code] Execution timed out.", verbose)
        return "", "Execution timed out."
    except Exception as e:
        log(f"[run_generated_code] Exception: {str(e)}", verbose)
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

    log(f"[test_solution] Running {len(test_examples)} test examples...", verbose)
    for idx, example in enumerate(test_examples, start=1):
        input_str = json.dumps(example.get("input"))
        expected_output = json.dumps(example.get("output"))
        stdout, stderr = run_generated_code(filename=filename, input_data=input_str, verbose=verbose)
        result = {
            "example_number": idx,
            "input": example.get("input"),
            "expected_output": example.get("output"),
            "actual_output": stdout.strip(),
            "error": stderr.strip()
        }
        log(f"[test_solution] Test Example {idx}:", verbose)
        log(f"Input: {result['input']}", verbose)
        log(f"Expected Output: {result['expected_output']}", verbose)
        log(f"Actual Output: {result['actual_output']}", verbose)
        if result["error"]:
            log(f"Error: {result['error']}", verbose)
        test_results.append(result)
    return test_results


def print_test_results(test_results):
    """
    Nicely prints the results of the tests.
    """
    for result in test_results:
        print(f"\n--- Test Example {result['example_number']} ---")
        print("Input:")
        print(result["input"])
        print("Expected Output:")
        print(result["expected_output"])
        print("Actual Output:")
        print(result["actual_output"])
        if result["error"]:
            print("Error:")
            print(result["error"])


def visualize_arc_grid(grid, title=""):
    """
    Visualizes a grid (list of lists) using matplotlib.
    Uses a discrete colormap similar to the one ARC uses.
    """
    arr = np.array(grid)
    plt.imshow(arr, cmap="tab20", interpolation="none")
    plt.title(title)
    plt.axis("off")


def visualize_test_results(test_results, output_dir=DEFAULT_OUTPUT_DIR, task_name="unknown"):
    """
    For each test example, creates a figure showing the input grid, the expected output, and the actual output.
    Saves figures to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for result in test_results:
        try:
            # The ARC grids are lists of lists of integers.
            input_grid = result["input"]
            expected_grid = result["expected_output"]
            # Try to parse the actual output (which should be JSON)
            actual_grid = json.loads(result["actual_output"])

            # Check if the solution is correct
            result_status = "correct" if np.array_equal(np.array(expected_grid), np.array(actual_grid)) else "incorrect"

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.array(input_grid), cmap="tab20", interpolation="none")
            axes[0].set_title("Input")
            axes[0].axis("off")
            axes[1].imshow(np.array(expected_grid), cmap="tab20", interpolation="none")
            axes[1].set_title("Expected")
            axes[1].axis("off")
            axes[2].imshow(np.array(actual_grid), cmap="tab20", interpolation="none")
            axes[2].set_title("Actual")
            axes[2].axis("off")
            plt.suptitle(f"Test Example {result['example_number']} - {result_status}")

            # Save the figure
            filename = os.path.join(output_dir, f"{task_name}_example_{result['example_number']}_{result_status}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Visualization saved to {filename}")

        except Exception as e:
            print(f"Could not visualize results for test example {result['example_number']}: {e}")
            continue


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


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # List and select an ARC task file from the evaluation directory
    files = list_task_files(args.eval_dir)
    task_file = select_task_file(files, args.eval_dir, args.task_index, args.verbose)
    task = load_arc_task(task_file, args.verbose)

    # Extract the task name for saving results
    task_name = os.path.splitext(os.path.basename(task_file))[0]

    # Initialize the Ollama client
    ollama.host = args.ollama_host

    iteration = 0
    current_code = generate_solution_code(task, ollama, args)
    code_file = os.path.join(args.output_dir, f"{task_name}_solution_temp.py")
    final_code = current_code

    while iteration < args.max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")
        write_generated_code(current_code, code_file, args.verbose)

        # Run the code without input to catch any immediate errors.
        stdout, stderr = run_generated_code(code_file, verbose=args.verbose)
        if stderr:
            print("Error encountered during execution:")
            print(stderr)
            fix_prompt = (
                f"ARC Task Training Data (in JSON): {json.dumps(task['train'])}\n\n"
                f"The previously generated Python code produced the following error when executed:\n{stderr}\n\n"
                "Please fix the code so that it runs without error and implements the transformation correctly. "
                "Output only the updated Python code, with no extra commentary."
            )
            log("[main] Requesting fix from LLM due to execution error.", args.verbose)
            current_code = generate_solution_code(task, ollama, args, prompt_prefix=fix_prompt)
        else:
            # Code ran without immediate errors; now test against the test examples.
            test_results = test_solution(task, code_file, args.verbose)
            if test_results:
                all_passed = True
                for tr in test_results:
                    expected = json.dumps(tr["expected_output"], sort_keys=True)
                    actual = tr["actual_output"]
                    try:
                        # Try to parse the actual output to JSON for a fair comparison
                        actual_json = json.dumps(json.loads(actual), sort_keys=True)
                        if expected != actual_json:
                            all_passed = False
                            break
                    except:
                        # If parsing fails, compare the raw strings
                        if expected != actual:
                            all_passed = False
                            break

                if all_passed:
                    print("All tests passed!")
                    print_test_results(test_results)
                    final_code = current_code
                    # Save the results and visualize
                    save_results(task_file, test_results, final_code, args.output_dir)
                    visualize_test_results(test_results, args.output_dir, task_name)
                    break
                else:
                    print("Some tests failed.")
                    print_test_results(test_results)
                    # Still save and visualize the current results
                    visualize_test_results(test_results, args.output_dir, task_name)

                    fix_prompt = (
                        f"ARC Task Training Data (in JSON): {json.dumps(task['train'], indent=2)}\n\n"
                        f"The following test results were obtained from the generated Python code:\n{json.dumps(test_results, indent=2)}\n\n"
                        "Please update the code so that it produces the correct output for all test examples. "
                        "Output only the updated Python code, with no extra commentary."
                    )
                    log("[main] Requesting fix from LLM due to failed test examples.", args.verbose)
                    current_code = generate_solution_code(task, ollama, args, prompt_prefix=fix_prompt)
            else:
                print("No test examples available. Here is the output from the code:")
                print(stdout)
                final_code = current_code
                break

        iteration += 1
        if iteration >= args.max_iterations:
            print("Reached maximum iterations. Saving the best result obtained.")
            # Save the last version
            final_code = current_code
            test_results = test_solution(task, code_file, args.verbose)
            save_results(task_file, test_results, final_code, args.output_dir)
            visualize_test_results(test_results, args.output_dir, task_name)
            break

    # Final notification
    print(f"\nExecution complete. All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()