#!/usr/bin/env python3
import sys
import os
import re
import json
import subprocess
import argparse
import platform
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from vllm import LLM, SamplingParams

# Suppress PyTorch/TF warnings that aren't helpful
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Default paths and configurations
OUTPUT_BASE_PATH = "/itet-stor/piroth/net_scratch/outputs"
MODEL_BASE_PATH = "/itet-stor/piroth/net_scratch/models"
DATA_BASE_PATH = "/itet-stor/piroth/net_scratch/data"
DATA_SAMPLE_BASE_PATH = "/itet-stor/piroth/net_scratch/rstar-arc/data_sample"

DEFAULT_POLICY_LLM = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_PP_LLM = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_VERBOSE = True
DEFAULT_MAX_ITERATIONS = 3


@dataclass
class TaskExample:
    """Represents a single example in an ARC task."""
    input_grid: List[List[int]]
    output_grid: Optional[List[List[int]]] = None


class ARCTask:
    """Represents an ARC task with methods to load and analyze the task data."""

    def __init__(self, file_path: str, verbose: bool = DEFAULT_VERBOSE):
        """Initialize an ARC task from a file path."""
        self.file_path: str = file_path
        self.verbose: bool = verbose
        self.name: str = os.path.splitext(os.path.basename(file_path))[0]
        self.train_examples: List[TaskExample] = []
        self.test_examples: List[TaskExample] = []
        self.load()

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def load(self) -> None:
        """Load task data from file."""
        self.log(f"Loading ARC task from '{self.file_path}'...")
        try:
            with open(self.file_path, "r") as f:
                task_data = json.load(f)

            if "train" not in task_data or "test" not in task_data:
                raise ValueError("The JSON file must contain 'train' and 'test' keys.")

            # Load training examples
            for example in task_data["train"]:
                self.train_examples.append(TaskExample(
                    input_grid=example["input"],
                    output_grid=example["output"]
                ))

            # Load test examples
            for example in task_data["test"]:
                self.test_examples.append(TaskExample(
                    input_grid=example["input"],
                    output_grid=example.get("output")  # Output may not exist for evaluation tasks
                ))

            self.log(f"Successfully loaded ARC task with {len(self.train_examples)} training and "
                     f"{len(self.test_examples)} test examples.")

        except Exception as e:
            print(f"Error loading task file {self.file_path}: {e}")
            sys.exit(1)

    def analyze(self) -> str:
        """Analyze the task and return insights as a string."""
        analysis = []

        # Training examples analysis
        analysis.append(f"\n### TRAINING ###\n")
        analysis.append(f"Task has {len(self.train_examples)} training examples.")

        for i, example in enumerate(self.train_examples):
            input_grid = example.input_grid
            output_grid = example.output_grid
            input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
            output_dims = f"{len(output_grid)}×{len(output_grid[0])}"
            analysis.append(f"Example {i + 1}: Input grid {input_dims}, Output grid {output_dims}")

            input_colors = set(item for row in input_grid for item in row)
            output_colors = set(item for row in output_grid for item in row)
            analysis.append(f"  - Input colors: {sorted(input_colors)}")
            analysis.append(f"  - Output colors: {sorted(output_colors)}")

            analysis.append(f"Example {i + 1}:\nInput:\n{input_grid}\nOutput:\n{output_grid}\n\n")

        # Test examples analysis
        analysis.append(f"\n### TEST ###\n")
        analysis.append(f"Task has {len(self.test_examples)} test examples.")

        for i, example in enumerate(self.test_examples):
            input_grid = example.input_grid
            input_dims = f"{len(input_grid)}×{len(input_grid[0])}"
            analysis.append(f"Test Example {i + 1}: Input grid {input_dims}")

            input_colors = set(item for row in input_grid for item in row)
            analysis.append(f"  - Input colors: {sorted(input_colors)}")

            analysis.append(f"Example {i + 1}:\nInput:\n{input_grid}\n")

        return "\n".join(analysis)


@dataclass
class TestResult:
    """Represents the results of running an example."""
    example_number: int
    input_grid: List[List[int]]
    expected_output: Optional[List[List[int]]]
    actual_output: Optional[List[List[int]]] = None
    raw_output: str = ""
    error: str = ""
    passed: bool = False


class CodeGenerator:
    """Handles LLM-based code generation and cleaning."""

    def __init__(self, model_name: str, gpus: int = 1, dtype: str = "float16",
                 verbose: bool = DEFAULT_VERBOSE):
        """Initialize the code generator."""
        self.model_name: str = model_name
        self.gpus: int = gpus
        self.dtype: str = dtype
        self.verbose: bool = verbose
        self.llm: Optional[LLM] = None

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def initialize_llm(self) -> None:
        """Initialize the language model."""
        if self.llm is None:
            self.log(f"Initializing LLM: {self.model_name}")
            try:
                self.llm = LLM(
                    model=self.model_name,
                    download_dir=os.path.join(MODEL_BASE_PATH, "policy"),
                    tensor_parallel_size=self.gpus,
                    dtype=self.dtype
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                sys.exit(1)

    def generate_solution(self, task: ARCTask, hint: str = "",
                          iteration: int = 0, past_attempts: List[str] = None) -> str:
        """Generate solution code for the given task."""
        # Initialize LLM if needed
        self.initialize_llm()

        # Generate task analysis
        task_analysis = task.analyze()

        # Build the prompt
        prompt = self._build_prompt(task_analysis, hint, past_attempts)

        self.log(f"Sending prompt to LLM: {self.model_name}")
        if self.verbose:
            self.log(prompt)

        try:
            # Generate code using the LLM
            # Find this line:

            params = SamplingParams(max_tokens=2048, temperature=0.15)
            outputs = self.llm.generate(prompt, params)
            full_response = outputs[0].outputs[0].text

            if not full_response:
                raise ValueError("No code output received from LLM.")

            self.log(f"\nFull output received")
            self.log(full_response)
            self.log("\n\n")

            # Clean the generated code
            code = self._clean_generated_code(full_response)
            self.log("Cleaned code:")
            self.log(code)

            return code

        except Exception as e:
            print(f"Error during code generation: {e}")
            sys.exit(1)

    @staticmethod
    def _build_prompt(task_analysis: str, hint: str = "",
                      past_attempts: List[str] = None) -> str:
        """Build the prompt for code generation."""
        # Base prompt
        prompt = (
            "# ARC Challenge Task\n\n"
            "You are given examples of input and output grids from the Abstraction and Reasoning Corpus (ARC). "
            "Your task is to figure out the transformation rule and implement it in Python.\n\n"
            f"## Task with Analysis\n{task_analysis}\n\n"
        )

        # Add past attempts if provided
        if past_attempts:
            prompt += "## Past Tries\n"
            for i, attempt in enumerate(past_attempts, start=1):
                prompt += f"### Try {i}\n{attempt}\n\n"

        # Add hint if provided
        if hint:
            prompt += f"## Hint\n{hint}\n\n"

        # Final instructions with example
        prompt += (
            "## Instructions\n"
            "1. Write a Python function called 'solve' that takes a grid (2D list of integers) and returns the transformed grid\n"
            "2. Your solution must implement the function: def solve(grid: list[list[int]]) -> list[list[int]]\n"
            "3. Focus on identifying patterns like: rotations, reflections, translations, color changes, etc.\n"
            "4. Make your code robust to handle different grid sizes if appropriate\n"
            "5. DO NOT use emojis, special characters, or any non-ASCII characters in your code\n"
            "6. DO NOT try to handle JSON directly - the input/output is already handled by the framework\n"
            "7. CRITICAL: Your code MUST be enclosed between <code> and </code> tags exactly as shown in the example\n"
            "8. ONLY include valid Python code between these tags - no explanations or other text\n"
            "9. If you don't follow the format exactly, your solution will be rejected\n\n"

            "## Example of Correct Solution Format:\n\n"
            "<code>\n"
            "# Can add a comment here to explain the thought process\n"
            "def solve(grid):\n"
            "    # Create a new grid that is 3x the size in both dimensions\n"
            "    height = len(grid)\n"
            "    width = len(grid[0])\n"
            "    \n"
            "    # Initialize the new grid with zeros\n"
            "    new_height = height * 3\n"
            "    new_width = width * 3\n"
            "    new_grid = [[0 for _ in range(new_width)] for _ in range(new_height)]\n"
            "    \n"
            "    # Copy the pattern to each of the 9 sub-grids\n"
            "    for i in range(height):\n"
            "        for j in range(width):\n"
            "            for di in range(3):\n"
            "                for dj in range(3):\n"
            "                    new_grid[i + height*di][j + width*dj] = grid[i][j]\n"
            "    \n"
            "    return new_grid\n"
            "</code>\n\n"

            "## Your Solution (must be enclosed in <code></code> tags):\n\n"
        )

        return prompt

    @staticmethod
    def _clean_generated_code(text: str) -> str:
        """
        Extract Python code from the model's output.
        Only accepts code properly enclosed in <code></code> tags.
        """
        # Check if text is empty
        if not text:
            return ""

        # Remove invalid characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Extract code block using HTML-style tags - strictly enforce <code></code> format
        code_pattern = r"<code>(.*?)</code>"
        match = re.search(code_pattern, text, flags=re.DOTALL)

        if not match:
            print("Warning: No properly formatted code found in model output. Looking for <code>...</code> format.")
            return ""

        # Get the matched content
        code = match.group(1).strip()

        # Verify the code contains a solve function
        if not re.search(r'def\s+solve\s*\(', code):
            print("Warning: Code does not contain a 'solve' function. Returning empty code.")
            return ""

        return code


class Solver:
    """Main class that handles solving ARC tasks."""

    def __init__(self, code_generator: CodeGenerator, max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 verbose: bool = DEFAULT_VERBOSE):
        """Initialize the solver."""
        self.code_generator: CodeGenerator = code_generator
        self.max_iterations: int = max_iterations
        self.verbose: bool = verbose

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def solve_task(self, task: ARCTask, hint: str = "") -> bool:
        """
        Solve the given ARC task and return True if successful.

        Args:
            task: The ARC task to solve
            hint: Optional hint to provide to the code generator

        Returns:
            True if the task was solved successfully, False otherwise
        """
        # Log task being processed
        print(f"\n==================================================")
        print(f"Processing task: {task.name}")
        print(f"==================================================")

        # Initialize variables for iteration
        iteration = 0
        past_attempts = []

        # Generate initial solution
        current_code = self.code_generator.generate_solution(task, hint)

        # Main iteration loop
        while iteration < self.max_iterations:
            print(f"\n=== Iteration {iteration + 1} ===")

            # Test on training examples
            training_results = self.test_on_training(task, current_code)
            self.print_results(training_results, "Training")

            # Check if all training examples pass
            all_training_passed = all(result.passed for result in training_results)

            if all_training_passed:
                print("\n🎉 All training examples passed! Testing on test examples...")

                # Run on test examples
                test_results = self.run_on_test(task, current_code)

                # Check for execution errors in test examples
                has_test_errors = any(result.error for result in test_results)

                if has_test_errors:
                    print("\n❌ Errors encountered in test examples:")
                    for result in test_results:
                        if result.error:
                            print(f"- Test Example {result.example_number}: {result.error}")

                    # Add to past attempts and generate new solution
                    error_feedback = "\n".join([
                        f"Test Example {result.example_number}: {result.error}"
                        for result in test_results if result.error
                    ])
                    past_attempts.append(current_code + f"\n\n# Errors on test examples:\n{error_feedback}")

                    iteration += 1
                    if iteration >= self.max_iterations:
                        break

                    print(f"\nAttempting to fix test errors (iteration {iteration + 1})")
                    current_code = self.code_generator.generate_solution(
                        task, hint, iteration=iteration, past_attempts=past_attempts
                    )
                else:
                    # No errors in test examples, calculate final score
                    passed_count = sum(1 for result in test_results if result.passed)
                    total_count = len(test_results)

                    print(f"\n--- Final Test Score: {passed_count}/{total_count} correct ---")

                    # Show results for each test example
                    for result in test_results:
                        status = "CORRECT" if result.passed else "INCORRECT"
                        print(f"Test Example {result.example_number}: {status}")

                    return True  # Success
            else:
                # Training examples failed, prepare feedback
                feedback = "Training results summary:\n"
                for result in training_results:
                    status = "PASSED" if result.passed else "FAILED"
                    feedback += f"- Example {result.example_number}: {status}\n"

                    # Add specific error information for failed examples
                    if not result.passed:
                        if result.error:
                            feedback += f"  Error: {result.error}\n"
                        elif result.actual_output is not None:
                            try:
                                # Show shapes for comparison
                                input_shape = np.array(result.input_grid).shape
                                expected_shape = np.array(result.expected_output).shape
                                actual_shape = None

                                if isinstance(result.actual_output, list):
                                    try:
                                        actual_shape = np.array(result.actual_output).shape
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
                if iteration >= self.max_iterations:
                    break

                print(f"\nAttempting to fix training failures (iteration {iteration + 1})")
                current_code = self.code_generator.generate_solution(
                    task, hint, iteration=iteration, past_attempts=past_attempts
                )

        # Reached maximum iterations without success
        print(f"Reached maximum iterations ({self.max_iterations}).")

        # Run final evaluation on test examples for tracking
        test_results = self.run_on_test(task, current_code)
        passed_count = sum(1 for result in test_results if result.passed)
        total_count = len(test_results)

        print(f"\nFinal score for {task.name}: {passed_count}/{total_count} tests correct")

        return False  # Not fully successful

    def test_on_training(self, task: ARCTask, code: str) -> List[TestResult]:
        """
        Test the code on training examples.

        Args:
            task: The ARC task containing training examples
            code: The Python code to test

        Returns:
            List of TestResult objects with the test results
        """
        test_results = []
        training_examples = task.train_examples

        if not training_examples:
            print("Warning: No training examples found in the ARC task.")
            return test_results

        self.log(f"Running {len(training_examples)} training examples...")

        for idx, example in enumerate(training_examples, start=1):
            # Prepare input data
            input_grid = example.input_grid
            input_str = str(input_grid)
            expected_output = example.output_grid

            # Run the code
            stdout, stderr = self._run_code(code, input_str)

            # Parse the output
            actual_output = None
            if stdout and stdout.strip():
                try:
                    actual_output = eval(stdout.strip())
                except Exception as e:
                    self.log(f"Warning: Could not parse output: {stdout.strip()} - {str(e)}")

            # Create result object
            result = TestResult(
                example_number=idx,
                input_grid=input_grid,
                expected_output=expected_output,
                actual_output=actual_output,
                raw_output=stdout.strip(),
                error=stderr.strip(),
                passed=False
            )

            # Check if the test passed
            if actual_output is not None:
                try:
                    if isinstance(actual_output, list) and isinstance(expected_output, list):
                        result.passed = np.array_equal(np.array(expected_output), np.array(actual_output))
                    else:
                        result.passed = False
                        self.log(f"Warning: Output format mismatch. Expected a list, got {type(actual_output)}")
                except Exception as e:
                    result.passed = False
                    self.log(f"Error comparing outputs: {e}")

            self.log(f"Training Example {idx}: {'PASSED' if result.passed else 'FAILED'}")
            test_results.append(result)

        return test_results

    def run_on_test(self, task: ARCTask, code: str) -> List[TestResult]:
        """
        Run the code on test examples.

        Args:
            task: The ARC task containing test examples
            code: The Python code to run

        Returns:
            List of TestResult objects with the test results
        """
        test_results = []
        test_examples = task.test_examples

        if not test_examples:
            print("Warning: No test examples found in the ARC task.")
            return test_results

        self.log(f"Running {len(test_examples)} test examples...")

        for idx, example in enumerate(test_examples, start=1):
            # Prepare input data
            input_grid = example.input_grid
            input_str = str(input_grid)
            expected_output = example.output_grid  # May be None

            # Run the code
            stdout, stderr = self._run_code(code, input_str)

            # Parse the output
            actual_output = None
            if stdout and stdout.strip():
                try:
                    actual_output = eval(stdout.strip())
                except Exception as e:
                    self.log(f"Warning: Could not parse output: {stdout.strip()} - {str(e)}")

            # Create result object
            result = TestResult(
                example_number=idx,
                input_grid=input_grid,
                expected_output=expected_output,
                actual_output=actual_output,
                raw_output=stdout.strip(),
                error=stderr.strip(),
                passed=False
            )

            # Check if valid output was produced and matches expected output (if available)
            if actual_output is not None and not stderr.strip() and expected_output is not None:
                try:
                    if isinstance(actual_output, list) and isinstance(expected_output, list):
                        result.passed = np.array_equal(np.array(expected_output), np.array(actual_output))
                    else:
                        result.passed = False
                except Exception as e:
                    result.passed = False
                    self.log(f"Error comparing test outputs: {e}")

            self.log(f"Test Example {idx}: Generated output")
            test_results.append(result)

        return test_results

    def _run_code(self, code_string: str, input_data: str = None) -> Tuple[str, str]:
        """
        Run the generated code in another process.

        Args:
            code_string: The Python code to run
            input_data: Optional input data as a string

        Returns:
            Tuple of (stdout, stderr)
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

        self.log(f"Running code with input: {input_data}")

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper_code],
                input=input_data if input_data else "",
                capture_output=True,
                text=True,
                timeout=5  # adjust timeout as needed
            )
            self.log(f"Stdout: {proc.stdout}")
            self.log(f"Stderr: {proc.stderr}")
            return proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            self.log("Execution timed out.")
            return "", "Execution timed out."
        except Exception as e:
            self.log(f"Exception: {str(e)}")
            return "", str(e)

    def print_results(self, results: List[TestResult], test_type: str = "Training") -> None:
        """
        Print the results of the tests in text format.

        Args:
            results: List of TestResult objects
            test_type: The type of test ("Training" or "Test")
        """
        passed_count = sum(1 for result in results if result.passed)
        total_count = len(results)

        print(f"\n--- {test_type} Results: {passed_count}/{total_count} passed ---")

        for result in results:
            status = "PASSED" if result.passed else "FAILED"
            print(f"\n{test_type} Example {result.example_number}: {status}")

            # Print input grid
            print("Input:")
            for row in result.input_grid:
                print(" ".join(str(cell) for cell in row))

            # Print expected output grid if available
            if result.expected_output:
                print("\nExpected Output:")
                for row in result.expected_output:
                    print(" ".join(str(cell) for cell in row))

            # Print actual output grid if available
            if result.actual_output:
                print("\nActual Output:")
                try:
                    if isinstance(result.actual_output, list) and len(result.actual_output) > 0:
                        for row in result.actual_output:
                            print(" ".join(str(cell) for cell in row))
                    else:
                        print(result.actual_output)
                except:
                    print(f"Could not format actual output: {result.actual_output}")
            else:
                print("\nActual Output: No valid output")

            # Print any errors
            if result.error:
                print("\nError:")
                print(result.error)


class CLI:
    """Handles command line interface and orchestration."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='ARC Task Solver')
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
        parser.add_argument('--dtype', type=str, default='float16',
                            help='Data type for model (float16, bfloat16) - use float16 for older GPUs')
        return parser.parse_args()

    @staticmethod
    def list_task_files(directory: str) -> List[str]:
        """List all JSON files in the given directory."""
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found. Please check your ARC data directory.")
            sys.exit(1)

        files = sorted([f for f in os.listdir(directory) if f.endswith('.json')],
                       key=lambda x: x.lower())

        if not files:
            print(f"No JSON files found in directory '{directory}'.")
            sys.exit(1)

        print(f"Found {len(files)} JSON files in '{directory}'")
        return files

    @staticmethod
    def select_task_file(files: List[str], directory: str, task_index: int,
                         verbose: bool = DEFAULT_VERBOSE) -> str:
        """Select a task file by index."""
        if task_index < 1 or task_index > len(files):
            print(f"Error: Task index {task_index} is out of range (1-{len(files)})")
            sys.exit(1)

        chosen_file = os.path.join(directory, files[task_index - 1])
        if verbose:
            print(f"Selected file by index: {chosen_file}")

        return chosen_file

    @staticmethod
    def run() -> None:
        """Run the ARC solver CLI."""
        # Parse command line arguments
        args = CLI.parse_args()

        # Print system info
        print(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")

        # Determine data directory
        data_dir = os.path.join(DATA_SAMPLE_BASE_PATH,
                                "evaluation" if args.eval else "training")

        # Get list of task files
        files = CLI.list_task_files(data_dir)

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
            tasks_to_process = [CLI.select_task_file(files, data_dir, args.task_index, args.verbose)]

        # Initialize code generator and solver
        code_generator = CodeGenerator(
            model_name=args.model,
            gpus=args.gpus,
            dtype=args.dtype,
            verbose=args.verbose
        )

        solver = Solver(
            code_generator=code_generator,
            max_iterations=args.max_iterations,
            verbose=args.verbose
        )

        # Process each task
        successful = 0
        total = len(tasks_to_process)

        print(f"Processing {total} task(s)...")

        for i, task_file in enumerate(tasks_to_process, 1):
            task_name = os.path.splitext(os.path.basename(task_file))[0]
            print(f"\nTask {i}/{total}: {task_name}")

            # Create and solve task
            task = ARCTask(task_file, args.verbose)
            success = solver.solve_task(task, args.hint)

            if success:
                successful += 1

        # Final summary
        print(f"\n==================================================")
        print(f"Processing complete: {successful}/{total} tasks solved successfully")
        print(f"==================================================")


if __name__ == "__main__":
    CLI.run()

