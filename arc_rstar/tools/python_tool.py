from __future__ import annotations

import json
import logging
import signal
import subprocess
import textwrap

from config import TIMEOUT_SECONDS, CODE, CODE_END, STEP_END, MEMORY_LIMIT_BYTES

logger = logging.getLogger(__name__)


def extract_python_code(text):
    """Extract Python code from text after the last CODE marker"""

    logger.debug(f"Extracting code from text (which has {len(text)} characters)")

    if not text.strip().endswith(STEP_END) and not text.strip().endswith(CODE_END) and not text.strip().endswith(
            "def solve(I):"):
        logger.warning(f"Text does not end with a valid marker (STEP_END or CODE_END)")

    # Find the last CODE marker and get all content after it
    last_code_start = text.rindex(CODE) + len(CODE)
    code = text[last_code_start:].strip()

    logger.debug(f"Extracted code block:\n{code}")

    return code


def prepare_code_for_execution(code):
    """Prepare code by removing STEP_END and CODE_END markers and ensuring it returns a value."""
    # Remove both types of markers
    clean_code = code.replace(f"{STEP_END}", "").replace(f"{CODE_END}", "")

    # If the code doesn't contain a return statement
    if 'return ' not in "\n".join(filter(lambda x: '#' not in x, clean_code.split())):
        clean_code += "\n    return []"

    return clean_code


def execute_code_in_subprocess(code_str, input_grids, expected_outputs):
    """
    Spawns a new Python interpreter to execute code against multiple grids.

    Parameters:
      - code_str: A string containing the code to test (defining a 'solve' function)
      - input_grids: List of input grids to test
      - expected_outputs: Optional list of expected output grids for validation

    Returns a tuple (error, passed, outputs):
      - error: True if execution failed
      - passed: True if all outputs match expected_outputs (when provided)
      - outputs: List of outputs from solve(grid) for each input grid
    """
    # Minimal wrapper code for the subprocess.
    # It sets memory and CPU limits, imports numpy, reads the user code,
    # loads the input data from a JSON argument, and executes solve() on each grid.
    wrapper_code = textwrap.dedent(f"""
        import resource
        import sys
        import json
        import numpy as np

        # Set memory limit
        try:
            resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_LIMIT_BYTES}, {MEMORY_LIMIT_BYTES}))
        except Exception as e:
            print(f"Warning: Could not set memory limit: {{e}}", file=sys.stderr)

        # Set CPU time limit
        try:
            resource.setrlimit(resource.RLIMIT_CPU, ({TIMEOUT_SECONDS}, {TIMEOUT_SECONDS}))
        except Exception as e:
            print(f"Warning: Could not set CPU time limit: {{e}}", file=sys.stderr)

        # Read user code from stdin
        user_code = sys.stdin.read()

        try:
            # Execute the user code to define solve()
            exec(user_code, globals())

            # Parse input data from command line arg
            data = json.loads(sys.argv[1])
            input_grids = data["input_grids"]
            expected_outputs = data.get("expected_outputs", [])

            # Test if solve is defined
            if 'solve' not in globals():
                print(json.dumps({{"error": True, "message": "Function 'solve' not defined"}}))
                sys.exit(1)

            # Run solve on each input grid
            results = []
            passed = True

            for i, grid in enumerate(input_grids):
                try:
                    result = solve(grid)

                    # Convert numpy arrays to lists
                    if hasattr(result, 'tolist'):
                        result = result.tolist()

                    results.append(result)

                    # Check against expected output if provided
                    if expected_outputs[i] is not None and result != expected_outputs[i]:
                        passed = False
                except Exception as e:
                    print(f"Error processing grid {{i}}: {{str(e)}}", file=sys.stderr)
                    results.append(None)
                    passed = False

            # Return the results as JSON
            print(json.dumps({{"error": False, "passed": passed, "results": results}}))

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({{"error": True, "message": str(e)}}))
            sys.exit(1)
    """)

    # Prepare input data
    input_data = {
        "input_grids": input_grids,
        "expected_outputs": expected_outputs
    }

    try:
        # Run the subprocess without a wall clock timeout (we rely solely on CPU time limit)
        proc = subprocess.run(
            ["python3", "-c", wrapper_code, json.dumps(input_data)],
            input=code_str.encode("utf-8"),
            capture_output=True
        )

        # Check if the subprocess was terminated by a signal
        if proc.returncode < 0:
            sig = -proc.returncode
            if sig == signal.SIGXCPU:
                logger.error("Process terminated due to CPU time limit exceeded.")
                return True, False, []
            else:
                logger.error(f"Process terminated by signal: {sig}")
                return True, False, []

        # Handle subprocess output
        stdout = proc.stdout.decode("utf-8").strip()
        stderr = proc.stderr.decode("utf-8").strip()

        if stderr:
            logger.debug(f"Subprocess stderr: {stderr}")

        if proc.returncode != 0:
            logger.error(f"Code execution failed with exit code {proc.returncode}: {stderr}")
            return True, False, []

        try:
            result_data = json.loads(stdout)
            if result_data.get("error", False):
                logger.error(f"Error in executed code: {result_data.get('message', 'Unknown error')}")
                return True, False, []

            return False, result_data.get("passed", False), result_data.get("results", [])

        except json.JSONDecodeError:
            logger.error(f"Failed to parse subprocess output: {stdout}")
            return True, False, []

    except subprocess.TimeoutExpired:
        # This should not happen now that we removed wall-clock timeout
        logger.error("Subprocess timed out unexpectedly. (This should be impossible)")
        return True, False, []
    except Exception as e:
        logger.error(f"Exception during code execution: {str(e)}")
        return True, False, []


def execute_code_with_task(code: str, input_grids: list[list[list[int]]],
                           expected_outputs: list) -> (bool, bool, list[list[list[int]]]):
    """
    Execute code against multiple input grids using a subprocess.

    Args:
        code: Python code to execute (must define a 'solve' function)
        input_grids: list of input grids to test
        expected_outputs: Optional list of expected output grids for validation

    Returns:
        Tuple of (error, passed, outputs):
            - error: True if execution failed
            - passed: True if all outputs match expected_outputs (when provided)
            - outputs: list of output grids produced by the code
    """
    if not code.strip():
        logger.warning("Cannot execute empty code")
        return True, False, []

    # Prepare the code if any pre-processing is needed
    code = prepare_code_for_execution(code)

    # Execute in subprocess
    return execute_code_in_subprocess(code, input_grids, expected_outputs)


def run_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all examples in a single process."""
    input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
    # expected_outputs = [example.output_grid.grid if example.output_grid is not None else None for example in task.training_examples + task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples] + [None] * len(
        task.test_examples)

    return execute_code_with_task(code, input_grids, expected_outputs)


def training_correct(node: "Node") -> (bool, bool, list[list[list[int]]]):
    if not node.valid:
        return True, False, []
    return False, node.passes_training, node.execution_outputs


def test_correct(node: "Node") -> (bool, bool, list[list[list[int]]]):
    if not node.valid:
        return True, False, []

    test_outputs = [example.output_grid.grid for example in node.task.test_examples]
    for generated_output, expected_output in zip(node.execution_outputs[len(node.task.training_examples):],
                                                 test_outputs):
        if generated_output != expected_output:
            return False, False, node.execution_outputs[len(node.task.training_examples):]

    return False, node.passes_training, node.execution_outputs


def run_training_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all training examples in a single process."""
    input_grids = [example.input_grid.grid for example in task.training_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples]

    return execute_code_with_task(code, input_grids, expected_outputs)


def run_test_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all test examples in a single process."""
    input_grids = [example.input_grid.grid for example in task.test_examples]
    expected_outputs = [example.output_grid.grid if example.output_grid is not None else None for example in
                        task.test_examples]

    return execute_code_with_task(code, input_grids, expected_outputs)
