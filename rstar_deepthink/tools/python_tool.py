import json
import logging
import os
import signal
import subprocess
import textwrap

from rstar_deepthink.config import TIMEOUT_SECONDS, CODE, CODE_END, STEP_END, MEMORY_LIMIT_BYTES

logger = logging.getLogger(__name__)

# --- Get the path for the subprocess python ---
# Read from environment variable set by the SLURM script
SUBPROCESS_PYTHON_PATH = os.environ.get("SUBPROCESS_PYTHON_EXEC")

if SUBPROCESS_PYTHON_PATH:
    logger.info(f"Using subprocess Python executable: {SUBPROCESS_PYTHON_PATH}")
else:
    # Fallback if the environment variable is not set
    SUBPROCESS_PYTHON_PATH = "python3"
    logger.warning(
        "SUBPROCESS_PYTHON_EXEC environment variable not set. "
        f"Falling back to '{SUBPROCESS_PYTHON_PATH}'. Subprocess performance might be suboptimal."
    )


# ---------------------------------------------


def remove_markers(code):
    """Remove STEP_END and CODE_END markers from the code."""
    # Remove both types of markers
    clean_code = code.replace(f"{CODE}\n", "").replace(f"{STEP_END}", "").replace(f"{CODE_END}", "")
    return clean_code


def comment_out_markers(code):
    """Comment out STEP_END and CODE_END markers in the code."""
    # Comment out both types of markers
    commented_code = code.replace(f"{CODE}\n", "").replace(f"{STEP_END}", f"# {STEP_END}").replace(f"{CODE_END}", "")
    return commented_code


def execute_code_in_subprocess(code_str, input_grids, expected_outputs):
    """
    Spawns a new Python interpreter (potentially minimal) to execute code.
    Uses the python executable specified by the SUBPROCESS_PYTHON_PATH variable.

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
    # Uses f-string interpolation for TIMEOUT_SECONDS and MEMORY_LIMIT_BYTES
    wrapper_code = textwrap.dedent(f"""
        import resource
        import sys
        import json
        import numpy as np

        # Set memory limit
        try:
            # Using f-string interpolation requires escaping curly braces used by f-string itself {{}}
            # But resource.setrlimit needs a tuple, so we construct it directly
            mem_limit = {MEMORY_LIMIT_BYTES}
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        except Exception as e:
            # Print errors to stderr so they are captured separately
            print(f"Warning: Could not set memory limit: {{e}}", file=sys.stderr)

        # Set CPU time limit
        try:
            cpu_limit = {TIMEOUT_SECONDS}
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except Exception as e:
            print(f"Warning: Could not set CPU time limit: {{e}}", file=sys.stderr)

        # Read user code from stdin
        user_code = sys.stdin.read()

        results = []
        passed = True
        error_occurred = False # Use a more specific variable name
        error_message = "Unknown error"

        try:
            # Execute the user code to define solve()
            # Using globals() allows solve to be defined in the global scope of the wrapper
            exec_globals = {{"np": np}}
            exec(user_code, exec_globals)

            # Parse input data from command line arg
            data = json.loads(sys.argv[1])
            input_grids = data["input_grids"]
            # Handle cases where expected_outputs might be missing or None
            expected_outputs = data.get("expected_outputs") or [None] * len(input_grids)

            # Test if solve is defined
            if 'solve' not in exec_globals:
                 # Critical error: solve not defined. Exit with error JSON.
                error_occurred = True
                error_message = "Function 'solve' not defined in submitted code"
                print(json.dumps({{"error": True, "passed": False, "results": [], "message": error_message}}))
                sys.exit(1) # Exit early

            solve_func = exec_globals['solve']

            # Run solve on each input grid
            for i, grid in enumerate(input_grids):
                grid_result = None
                try:
                    grid_result = solve_func(grid) # Call the extracted function

                    # Convert numpy arrays to lists for JSON serialization
                    if hasattr(grid_result, 'tolist'):
                        grid_result = grid_result.tolist()

                    results.append(grid_result)

                    # Check against expected output if provided (handle index errors if lengths mismatch)
                    if i < len(expected_outputs) and expected_outputs[i] is not None:
                         if grid_result != expected_outputs[i]:
                            passed = False # Failed this specific test case, overall 'passed' is False
                except Exception as e:
                    # Error occurred processing this specific grid
                    print(f"Error processing grid {{i}}: {{str(e)}}", file=sys.stderr) # Log error for debugging
                    results.append(None) # Append None for this grid's result
                    passed = False # If any grid fails or errors, overall 'passed' is False
                    error_occurred = True # Mark that an error happened during execution

            # If loop completes, print final results
            # The 'error' flag indicates if *any* runtime error happened during grid processing
            # 'passed' indicates if all results matched expectations (where provided)
            print(json.dumps({{"error": error_occurred, "passed": passed, "results": results}}))

        except Exception as e:
            # Catch errors during exec, JSON parsing, or other setup before the loop
            import traceback
            traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
            error_occurred = True
            error_message = str(e)
             # Return error JSON; results list might be empty or partially filled
            print(json.dumps({{"error": True, "passed": False, "results": results, "message": error_message}}))
            sys.exit(1) # Exit indicating failure
    """)  # End of dedent

    # Prepare input data dictionary
    input_data = {
        "input_grids": input_grids,
        "expected_outputs": expected_outputs
    }

    try:
        # --- Use the SUBPROCESS_PYTHON_PATH determined earlier ---
        proc = subprocess.run(
            [SUBPROCESS_PYTHON_PATH, "-c", wrapper_code, json.dumps(input_data)],
            input=code_str.encode("utf-8"),
            capture_output=True,
            # We removed the wall-clock timeout, relying on RLIMIT_CPU set inside the wrapper.
            # Consider adding a safety timeout here slightly longer than TIMEOUT_SECONDS,
            # e.g., timeout=TIMEOUT_SECONDS + 5, to catch hangs not stopped by RLIMIT_CPU.
            # timeout=TIMEOUT_SECONDS + 5
        )

        # Check if the subprocess was terminated by a signal
        if proc.returncode < 0:
            sig = -proc.returncode
            # Log signal terminations at DEBUG level as requested
            if sig == signal.SIGXCPU:
                logger.debug(
                    f"Subprocess terminated due to CPU time limit ({TIMEOUT_SECONDS}s) exceeded (Signal {sig}).")
            elif sig == signal.SIGSEGV:
                logger.debug(
                    f"Subprocess terminated by segmentation fault (Signal {sig}). Check for memory issues or invalid operations in user code.")
            # 9 is SIGKILL, often from OOM killer or manual kill
            elif sig == signal.SIGKILL:
                logger.debug(
                    f"Subprocess terminated by KILL signal (Signal {sig}). Possibly OOM killed (Memory Limit: {MEMORY_LIMIT_BYTES} bytes).")
            else:
                logger.debug(f"Subprocess terminated unexpectedly by signal: {sig}")
            return True, False, []  # Error, Not Passed, Empty Results

        # Handle subprocess output and stderr
        stdout = proc.stdout.decode("utf-8").strip()
        stderr = proc.stderr.decode("utf-8").strip()

        # Log stderr for debugging purposes if it's not empty
        if stderr:
            logger.debug(f"Subprocess stderr:\n{stderr}")

        # Check for non-zero exit code AFTER signal check (signals have negative return codes)
        # A non-zero code here usually means the wrapper script exited via sys.exit(1)
        if proc.returncode != 0:
            logger.debug(
                f"Subprocess execution failed with exit code {proc.returncode}. See stderr log above. Attempting to parse stdout for JSON error.")
            # Try to parse stdout anyway, as the wrapper might print JSON even on exit(1)
            try:
                result_data = json.loads(stdout)
                logger.debug(f"Parsed error JSON from stdout: {result_data.get('message', 'No message')}")
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse stdout as JSON. Raw stdout: {stdout}")
            return True, False, []  # Error, Not Passed, Empty Results

        # If return code is 0, attempt to parse the JSON output
        try:
            result_data = json.loads(stdout)
            # Check the 'error' flag within the JSON, as non-fatal errors during grid processing set this
            internal_error = result_data.get("error", False)
            if internal_error:
                logger.debug(
                    f"Subprocess reported an internal error during execution (error flag set in JSON). Message: {result_data.get('message', 'None')}")

            # Return: internal_error flag, passed flag, results list
            return internal_error, result_data.get("passed", False), result_data.get("results", [])

        except json.JSONDecodeError:
            # This should ideally not happen if return code is 0
            logger.error(
                f"CRITICAL: Subprocess had return code 0 but failed to produce valid JSON output. Stdout: {stdout}")
            return True, False, []  # Treat as error

    # Exception handling for the subprocess.run call itself (e.g., executable not found)
    except FileNotFoundError:
        logger.error(
            f"CRITICAL: Subprocess Python executable not found at '{SUBPROCESS_PYTHON_PATH}'. Check installation and environment variable.")
        return True, False, []
    except subprocess.TimeoutExpired:
        # This happens only if you add a wall-clock 'timeout' to subprocess.run
        logger.error(f"Subprocess exceeded wall-clock timeout. Potential hang not caught by RLIMIT_CPU.")
        return True, False, []
    except Exception as e:
        # Catch any other unexpected errors during subprocess management
        logger.error(f"CRITICAL: Exception during subprocess invocation/management: {str(e)}", exc_info=True)
        return True, False, []


def execute_code_with_task(code: str, input_grids: list[list[list[int]]],
                           expected_outputs: list) -> (bool, bool, list[list[list[int]]]):
    """
    Execute code against multiple input grids using the configured subprocess environment.

    Args:
        code: Python code to execute (must define a 'solve' function)
        input_grids: list of input grids to test
        expected_outputs: Optional list of expected output grids for validation

    Returns:
        Tuple of (error, passed, outputs):
            - error: True if execution failed or reported errors internally
            - passed: True if all outputs match expected_outputs (when provided)
            - outputs: list of output grids produced by the code
    """
    if not code.strip():
        logger.debug("Cannot execute empty code!")
        return True, False, []  # Error: True, Passed: False, Empty Results

    # Prepare the code (if needed - e.g., uncomment if you bring back prepare_code_for_execution)
    # code = prepare_code_for_execution(code)

    # Execute in subprocess - uses the globally determined SUBPROCESS_PYTHON_PATH
    return execute_code_in_subprocess(code, input_grids, expected_outputs)


# --- The rest of your functions (run_examples, test_correct) remain unchanged ---
# Assuming they correctly call execute_code_with_task

def run_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all examples in a single process."""

    input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples] + [None] * len(
        task.test_examples)

    return execute_code_with_task(code, input_grids, expected_outputs)


def test_correct(node) -> (bool, bool, list[list[list[int]]]):
    """Test correctness against test examples based on prior execution results."""
    # Ensure node object has expected attributes
    try:
        if not node.valid:
            logger.debug(f"Node {getattr(node, 'id', 'N/A')} is not valid, skipping correctness test.")
            return True, False, []  # Treat as error / not passed

        num_training = len(node.task.training_examples)
        # Check if execution_outputs exist and have enough elements
        if not hasattr(node, 'execution_outputs') or node.execution_outputs is None or len(
                node.execution_outputs) < num_training + len(node.task.test_examples):
            logger.warning(
                f"Node {getattr(node, 'id', 'N/A')} execution outputs missing or incomplete. Cannot test correctness.")
            return True, False, []  # Treat as error / not passed

        test_outputs_expected = [example.output_grid.grid for example in node.task.test_examples]
        # Extract the portion of execution outputs corresponding to test examples
        test_outputs_generated = node.execution_outputs[num_training:]

        passed_test = True
        for generated_output, expected_output in zip(test_outputs_generated, test_outputs_expected):
            if generated_output != expected_output:
                passed_test = False
                break  # Exit loop early if one mismatch is found

        # The function should return (error_flag, passes_test_examples, test_outputs)
        # Assuming 'node.passes_training' indicates if training examples passed during execution
        # We return passes_test flag based on our check here.
        # We assume no new error occurred during this check itself, so error_flag is False.
        return False, passed_test, test_outputs_generated

    except AttributeError as e:
        logger.error(f"Error accessing node/task attributes during correctness test: {e}")
        return True, False, []  # Treat as error / not passed
    except Exception as e:
        logger.error(f"Unexpected error during correctness test: {e}", exc_info=True)
        return True, False, []  # Treat as error / not passed
