import sys
import os
import re
import json
import subprocess
import argparse
import numpy as np
from vllm import LLM

# Simple configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_BASE_PATH = "/itet-stor/piroth/net_scratch/outputs"
MODEL_BASE_PATH = "/itet-stor/piroth/net_scratch/models"
DATA_BASE_PATH = "/itet-stor/piroth/net_scratch/data"
DATA_SAMPLE_BASE_PATH = "/itet-stor/piroth/net_scratch/rstar-arc/data_sample"

DEFAULT_VERBOSE = True
DEFAULT_MAX_ITERATIONS = 3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ARC Task Solver Baseline')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='LLM model to use')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS, help='Maximum fix iterations')
    parser.add_argument('--eval', action='store_true', default=False, help='Use evaluation tasks instead of training')
    parser.add_argument('--task-index', type=int, default=1, help='Index of task to test (1-based)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for LLM')
    parser.add_argument('--hint', type=str, default='', help='Hint for the LLM')
    parser.add_argument('--output-dir', type=str, default=os.path.join(OUTPUT_BASE_PATH, "arc_results"), help='Directory to store any output files')
    parser.add_argument('--dtype', type=str, default='float16', help='Data type for model')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE, help=f'Print detailed progress information (default: {DEFAULT_VERBOSE})')
    return parser.parse_args()


def load_arc_task(file_path):
    """Loads an ARC task file."""
    print(f"Loading ARC task from '{file_path}'...")
    with open(file_path, "r") as f:
        task = json.load(f)
    print(f"Loaded ARC task with {len(task['train'])} training and {len(task['test'])} test examples.")
    return task


def analyze_task(task):
    """Analyzes the training examples."""
    analysis = ["\n### TRAINING ###\n"]
    analysis.append(f"Task has {len(task['train'])} training examples.")

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
    analysis.append(f"Task has {len(task['test'])} test examples.")

    for i, example in enumerate(task['test']):
        input_grid = example['input']
        input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
        analysis.append(f"Test Example {i + 1}: Input grid {input_dims}")
        input_colors = set(item for row in input_grid for item in row)
        analysis.append(f"  - Input colors: {sorted(input_colors)}")
        analysis.append(f"Example {i + 1}:\nInput:\n{input_grid}\n")

    return "\n".join(analysis)


def extract_code(text):
    """Extract Python code from between triple backticks."""
    # Look for Python code block
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # If no code block found, just return the text
    return text.strip()


def run_code(code_string, input_grid):
    """Run the generated code with the input grid."""
    wrapper_code = f"""
import sys
import traceback

code_string = '''
{code_string}
'''

# Execute the code
try:
    # Create a namespace
    namespace = {{'grid': {input_grid}}}

    # Execute the code
    exec(code_string, namespace)

    # Call the solve function
    if 'solve' in namespace:
        result = namespace['solve'](namespace['grid'])
        print(repr(result))
    else:
        print("Error: No solve function found")
except Exception as e:
    print(f"Error: {{e}}")
    traceback.print_exc()
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper_code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", "Execution timed out."
    except Exception as e:
        return "", str(e)


def test_on_examples(task, code, examples_key="train"):
    """Test the code on examples."""
    results = []
    examples = task.get(examples_key, [])

    print(f"Testing on {len(examples)} {examples_key} examples...")
    for idx, example in enumerate(examples):
        input_grid = example.get("input")
        expected_output = example.get("output") if examples_key == "train" else None

        stdout, stderr = run_code(code, input_grid)

        # Try to parse output
        actual_output = None
        if stdout:
            try:
                actual_output = eval(stdout)
            except:
                pass

        result = {
            "example_number": idx + 1,
            "input": input_grid,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "error": stderr,
            "passed": False
        }

        # Check if test passed (only for training examples)
        if examples_key == "train" and actual_output and expected_output:
            try:
                result["passed"] = np.array_equal(np.array(actual_output), np.array(expected_output))
            except:
                result["passed"] = False

        results.append(result)

    return results


def create_prompt(task, past_results=None, hint=""):
    """Create a prompt for the LLM."""
    task_analysis = analyze_task(task)

    prompt = (
        "# ARC Challenge Task\n\n"
        "You are given examples of input and output grids from the Abstraction and Reasoning Corpus (ARC). "
        "Figure out the transformation rule and implement it in Python.\n\n"
        f"## Task\n{task_analysis}\n\n"
    )

    # Add past results if available
    if past_results:
        prompt += "## Past Attempts\n"
        for i, (code, results) in enumerate(past_results, start=1):
            prompt += f"### Attempt {i}\n```python\n{code}\n```\n"
            # Add basic error info
            errors = [r for r in results if not r["passed"]]
            if errors:
                prompt += "\nErrors:\n"
                for error in errors:
                    if error["error"]:
                        prompt += f"- Example {error['example_number']}: {error['error']}\n"
            prompt += "\n"

    # Add hint if provided
    if hint:
        prompt += f"## Hint\n{hint}\n\n"

    # Final instructions
    prompt += (
        "## Instructions\n"
        "1. Write a Python function `solve(grid)` that takes a 2D grid of integers and returns the transformed grid\n"
        "2. The function should implement the transformation demonstrated in the examples\n"
        "3. Don't forget to close the python code block\n"
        "4. Return just the Python code - no explanations except as a python comment\n\n"
        
        "```python\n"
    )

    return prompt


def main():
    args = parse_args()

    # Determine data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data_sample", "evaluation" if args.eval else "training")

    # List task files
    try:
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')], key=lambda x: x.lower())
        if not files:
            print(f"No JSON files found in directory '{data_dir}'.")
            sys.exit(1)
    except:
        print(f"Error: Directory '{data_dir}' not found.")
        sys.exit(1)

    # Select task file
    if args.task_index < 1 or args.task_index > len(files):
        print(f"Error: Task index {args.task_index} is out of range (1-{len(files)})")
        sys.exit(1)
    task_file = os.path.join(data_dir, files[args.task_index - 1])
    task_name = os.path.splitext(os.path.basename(task_file))[0]

    # Load task
    task = load_arc_task(task_file)

    # Initialize LLM (just once)
    print(f"Loading LLM: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.gpus,
        dtype=args.dtype
    )

    # Main iteration loop
    iteration = 0
    past_results = []

    print(f"\nProcessing task: {task_name}")

    while iteration < args.max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")

        # Generate solution
        prompt = create_prompt(task, past_results, args.hint)
        outputs = llm.generate(prompts=[prompt])
        full_response = outputs[0].outputs[0].text

        # Extract code
        code = extract_code(full_response)
        print("\nGenerated code:")
        print(code)

        # Test on training examples
        train_results = test_on_examples(task, code, "train")

        # Print results
        passed_count = sum(1 for r in train_results if r["passed"])
        total_count = len(train_results)
        print(f"\nTraining Results: {passed_count}/{total_count} passed")

        # Check if all training examples pass
        if passed_count == total_count:
            print("\n🎉 All training examples passed! Testing on test examples...")

            # Test on test examples
            test_results = test_on_examples(task, code, "test")

            # For reporting only
            test_passed = 0
            for idx, test in enumerate(test_results):
                expected = task["test"][idx].get("output")
                if test["actual_output"] and np.array_equal(np.array(test["actual_output"]), np.array(expected)):
                    test_passed += 1

            print(f"\nTest Results: {test_passed}/{len(test_results)} correct")
            return

        # Save results for next iteration
        past_results.append((code, train_results))
        iteration += 1

    print(f"\nReached maximum iterations ({args.max_iterations}) without success.")


if __name__ == "__main__":
    main()

