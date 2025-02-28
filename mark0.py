import os
import json
import re
import subprocess
import sys
import argparse
import numpy as np
import ollama  # Ensure you have installed the Python Ollama library

# GLOBAL SETTINGS - Now configurable via command line arguments
DEFAULT_LLM = "qwen2.5-coder:7b"
DEFAULT_MAX_ITERATIONS = 3
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
    log(f"Selected file: {chosen_file}", verbose)
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
        return max(code_blocks, key=len).strip()

    # If no code block is found, just try to clean up the text
    cleaned = text.strip()

    # Remove markdown-style code block markers without language
    cleaned = re.sub(r"```\s*\n", "", cleaned)
    cleaned = re.sub(r"\n\s*```", "", cleaned)

    return cleaned


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

    return "\n".join(analysis)


def generate_solution_code(task, ollama_client, args, iteration=0, errors=None, past_outputs=None):
    """Sends the ARC task to the LLM via Ollama and gets the generated Python code."""

    hint = args.hint
    if not hint and iteration == 0:
        # Only ask for a hint if this is the initial generation
        hint_input = input("Do you want to provide a hint to the model? (y/n): ")
        if hint_input.lower() == "y":
            hint = input("Enter a hint for the model: ")

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
    if args.verbose:
        log(prompt)

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
    Runs the generated solution code as a separate process.
    Optionally sends input_data (string) to the process's stdin.
    Returns a tuple (stdout, stderr).
    """
    log(f"Running code with input: {input_data}", verbose)
    try:
        proc = subprocess.run(
            [sys.executable, filename],
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
        input_str = json.dumps(example.get("input"))
        expected_output = example.get("output")
        stdout, stderr = run_generated_code(filename=filename, input_data=input_str, verbose=verbose)

        # Try to parse the actual output
        actual_output = None
        if stdout:
            try:
                actual_output = json.loads(stdout.strip())
            except json.JSONDecodeError:
                pass

        result = {
            "example_number": idx,
            "input": example.get("input"),
            "expected_output": expected_output,
            "actual_output": actual_output,
            "raw_output": stdout.strip(),
            "error": stderr.strip(),
            "passed": False
        }

        # Check if the test passed
        if actual_output:
            result["passed"] = np.array_equal(np.array(expected_output), np.array(actual_output))

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
            for row in result["actual_output"]:
                print(" ".join(str(cell) for cell in row))
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

    # Run iteration loop
    iteration = 0
    current_code = generate_solution_code(task, ollama, args)
    code_file = os.path.join(args.output_dir, f"{task_name}_solution_temp.py")
    final_code = current_code

    errors = None

    while iteration < args.max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")
        write_generated_code(current_code, code_file, args.verbose)

        # Run the code without input to catch any immediate errors
        stdout, stderr = run_generated_code(code_file, verbose=args.verbose)
        if stderr:
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
                    final_code = current_code
                    # Save the results
                    save_results(task_file, test_results, final_code, args.output_dir)
                    break
                else:
                    print("\n❌ Some tests failed.")

                    # Prepare error feedback for the model
                    feedback = "Test results summary:\n"
                    for tr in test_results:
                        status = "PASSED" if tr["passed"] else "FAILED"
                        feedback += f"- Example {tr['example_number']}: {status}\n"

                    errors = feedback
                    iteration += 1
                    if iteration >= args.max_iterations:
                        break

                    print(f"\nAttempting to fix failed tests (iteration {iteration + 1})")
                    current_code = generate_solution_code(task, ollama, args, iteration=iteration, errors=errors)
            else:
                print("No test examples available. Here is the output from the code:")
                print(stdout)
                final_code = current_code
                break

    # Save final results
    if iteration >= args.max_iterations:
        print(f"Reached maximum iterations ({args.max_iterations}). Saving the last result.")

    test_results = test_solution(task, code_file, args.verbose)
    save_results(task_file, test_results, current_code, args.output_dir)
    print_test_results(test_results)

    # Final notification
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)
    print(f"\nFinal score: {passed_count}/{total_count} tests passed")
    print(f"All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()