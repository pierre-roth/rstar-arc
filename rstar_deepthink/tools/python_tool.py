import json
import logging
import os
import signal
import subprocess
import sys
import textwrap

import numpy as np  # Added for direct execution context

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import CPU_TIMEOUT_SECONDS, WALL_TIMEOUT_SECONDS, CODE, CODE_END, STEP_END, MEMORY_LIMIT_BYTES

logger = logging.getLogger(__name__)

use_subprocess = True
logger.info(f"Code execution method: {'Subprocess' if use_subprocess else 'Direct'}")

GLOBAL_IMPORTS = "from train.common import *\nimport numpy as np\nfrom typing import *\n"

# --- Get the path for the subprocess python ---
SUBPROCESS_PYTHON_PATH = os.environ.get("SUBPROCESS_PYTHON_EXEC")

if SUBPROCESS_PYTHON_PATH:
    logger.info(f"Using subprocess Python executable: {SUBPROCESS_PYTHON_PATH}")
else:
    SUBPROCESS_PYTHON_PATH = "python3"
    logger.info(
        "SUBPROCESS_PYTHON_EXEC environment variable not set. "
        f"Falling back to '{SUBPROCESS_PYTHON_PATH}'. Subprocess performance might be suboptimal."
    )


def remove_markers(code):
    """Remove STEP_END and CODE_END markers from the code."""
    clean_code = code.replace(f"{CODE}\n", "").replace(f"{STEP_END}", "").replace(f"{CODE_END}", "")
    return clean_code


def comment_out_markers(code):
    """Comment out STEP_END and CODE_END markers in the code."""
    commented_code = code.replace(f"{CODE}\n", "").replace(f"{STEP_END}", f"# {STEP_END}").replace(f"{CODE_END}", "")
    return commented_code


def execute_code_in_subprocess(code_str, input_grids, expected_outputs):
    """
    Spawns a new Python interpreter (potentially minimal) to execute code.
    (Original subprocess logic - kept for when use_subprocess is True)
    """
    wrapper_code = textwrap.dedent(f"""
        import resource
        import sys
        import json
        # Ensure numpy is available in the subprocess environment if user code needs it
        try:
            import numpy as np
        except ImportError:
            np = None # Allow code to potentially run without numpy if not used

        # Set memory limit
        try:
            mem_limit = {MEMORY_LIMIT_BYTES}
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        except Exception as e:
            print(f"Warning: Could not set memory limit: {{e}}", file=sys.stderr)

        # Set CPU time limit
        try:
            cpu_limit = {CPU_TIMEOUT_SECONDS}
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except Exception as e:
            print(f"Warning: Could not set CPU time limit: {{e}}", file=sys.stderr)

        # Read user code from stdin
        user_code = sys.stdin.read()

        results = []
        passed = True
        error_occurred = False
        error_message = "Unknown error"

        try:
            # Execute the user code to define solve()
            exec_globals = {{"np": np}} # Pass numpy alias if available
            exec(user_code, exec_globals)

            # Parse input data from command line arg
            data = json.loads(sys.argv[1])
            input_grids = data["input_grids"]
            expected_outputs = data.get("expected_outputs") or [None] * len(input_grids)

            if 'solve' not in exec_globals:
                error_occurred = True
                error_message = "Function 'solve' not defined in submitted code"
                print(json.dumps({{"error": True, "passed": False, "results": [], "message": error_message}}))
                sys.exit(1)

            solve_func = exec_globals['solve']

            # Run solve on each input grid
            for i, grid in enumerate(input_grids):
                grid_result = None
                try:
                    grid_result = solve_func(grid)

                    # Convert numpy arrays to lists for JSON serialization
                    if hasattr(grid_result, 'tolist'):
                        grid_result = grid_result.tolist()
                    # Basic grid validation could be added here if needed

                    results.append(grid_result)

                    if i < len(expected_outputs) and expected_outputs[i] is not None:
                         if grid_result != expected_outputs[i]:
                            passed = False
                except Exception as e:
                    print(f"Error processing grid {{i}}: {{str(e)}}", file=sys.stderr)
                    results.append(None)
                    passed = False
                    error_occurred = True

            print(json.dumps({{"error": error_occurred, "passed": passed, "results": results}}))

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            error_occurred = True
            error_message = str(e)
            print(json.dumps({{"error": True, "passed": False, "results": results, "message": error_message}}))
            sys.exit(1)
    """)  # End of dedent

    input_data = {
        "input_grids": input_grids,
        "expected_outputs": expected_outputs
    }

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [SUBPROCESS_PYTHON_PATH, "-c", wrapper_code, json.dumps(input_data)],
            input=code_str.encode("utf-8"),
            capture_output=True,
            timeout=WALL_TIMEOUT_SECONDS,  # Use the wall-clock timeout for the subprocess run
            env=env,
            cwd=str(project_root),
        )

        if proc.returncode < 0:
            sig = -proc.returncode
            if sig == signal.SIGXCPU:
                logger.debug(
                    f"Subprocess terminated due to CPU time limit ({CPU_TIMEOUT_SECONDS}s) exceeded (Signal {sig}).")
            elif sig == signal.SIGSEGV:
                logger.debug(f"Subprocess terminated by segmentation fault (Signal {sig}).")
            elif sig == signal.SIGKILL:
                logger.debug(
                    f"Subprocess terminated by KILL signal (Signal {sig}). Possibly OOM killed (Memory Limit: {MEMORY_LIMIT_BYTES} bytes).")
            else:
                logger.debug(f"Subprocess terminated unexpectedly by signal: {sig}")
            return True, False, []

        stdout = proc.stdout.decode("utf-8").strip()
        stderr = proc.stderr.decode("utf-8").strip()

        if stderr:
            logger.debug(f"Subprocess stderr:\n{stderr}")

        if proc.returncode != 0:
            logger.debug(f"Subprocess failed with exit code {proc.returncode}.")
            # Attempt to parse potential error JSON from stdout even on non-zero exit
            try:
                result_data = json.loads(stdout.splitlines()[-1])
                logger.debug(f"Parsed error JSON from stdout: {result_data.get('message', 'No message')}")
            except (json.JSONDecodeError, IndexError):
                logger.debug(f"Failed to parse stdout as JSON on non-zero exit. Raw stdout: {stdout}")
            return True, False, []

        try:
            last_line = stdout.splitlines()[-1]
            result_data = json.loads(last_line)
            internal_error = result_data.get("error", False)
            if internal_error:
                logger.debug(f"Subprocess reported internal error. Message: {result_data.get('message', 'None')}")
            return internal_error, result_data.get("passed", False), result_data.get("results", [])
        except (json.JSONDecodeError, IndexError):
            logger.error(f"CRITICAL: Subprocess OK but failed to parse JSON output. Stdout: {stdout}")
            return True, False, []

    except FileNotFoundError:
        logger.error(f"CRITICAL: Subprocess Python executable not found: '{SUBPROCESS_PYTHON_PATH}'.")
        return True, False, []
    except subprocess.TimeoutExpired:
        logger.error(f"Subprocess exceeded wall-clock timeout ({WALL_TIMEOUT_SECONDS}s). Potential hang.")
        return True, False, []
    except Exception as e:
        logger.error(f"CRITICAL: Exception during subprocess management: {str(e)}", exc_info=True)
        return True, False, []


def execute_code_directly(code_str, input_grids, expected_outputs):
    """
    Executes the code directly in the current process.
    WARNING: Lacks isolation, resource limits, and subprocess timeouts.
    """
    results = []
    passed = True
    error_occurred = False
    exec_globals = {"np": np}  # Provide numpy alias

    try:
        # Execute the user code string in the controlled context
        exec(code_str, exec_globals)

        if 'solve' not in exec_globals:
            logger.error("Direct execution failed: Function 'solve' not defined.")
            return True, False, []  # Error, Not Passed, Empty Results

        solve_func = exec_globals['solve']

        # Ensure expected_outputs has the same length as input_grids
        if expected_outputs is None:
            expected_outputs = [None] * len(input_grids)

        # Run solve on each input grid
        for i, grid in enumerate(input_grids):
            grid_result = None
            try:
                # Potentially time the individual call if needed for debugging
                # start_time = time.time()
                grid_result = solve_func(grid)
                # elapsed = time.time() - start_time
                # logger.debug(f"Direct execution grid {i} took {elapsed:.4f}s")

                # Convert numpy arrays to lists if necessary (less likely needed here, but for consistency)
                if hasattr(grid_result, 'tolist'):
                    grid_result = grid_result.tolist()

                results.append(grid_result)

                # Check against expected output if provided
                if i < len(expected_outputs) and expected_outputs[i] is not None:
                    if grid_result != expected_outputs[i]:
                        passed = False  # Failed this specific test case
            except Exception as e:
                # Error occurred processing this specific grid
                logger.debug(f"Error during direct execution of solve() on grid {i}: {str(e)}")
                # traceback.print_exc() # Uncomment for full traceback during debugging
                results.append(None)  # Append None for this grid's result
                passed = False  # If any grid fails or errors, overall 'passed' is False
                error_occurred = True  # Mark that an error happened

    except Exception as e:
        # Catch errors during the initial exec call
        logger.error(f"Error executing provided code string directly: {str(e)}")
        # traceback.print_exc() # Uncomment for full traceback during debugging
        error_occurred = True
        passed = False  # Mark as not passed if exec fails

    return error_occurred, passed, results


def execute_code_with_task(code: str, input_grids: list[list[list[int]]],
                           expected_outputs: list) -> (bool, bool, list[list[list[int]]]):
    """
    Execute code against multiple input grids using either a subprocess (default)
    or direct execution based on the global 'use_subprocess' flag.
    """
    if not code.strip():
        logger.debug("Cannot execute empty code!")
        return True, False, []

    code = GLOBAL_IMPORTS + code

    if use_subprocess:
        logger.debug("Executing code via subprocess.")
        return execute_code_in_subprocess(code, input_grids, expected_outputs)
    else:
        logger.debug("Executing code directly.")
        return execute_code_directly(code, input_grids, expected_outputs)


def run_examples(
        task,
        code: str,
        test_test: bool = False,
) -> (bool, bool, list[list[list[int]]]):
    """
    Run code against all examples in a single process.

    By default, only training examples are validated. If test_test is True,
    test examples are also checked against their expected outputs.
    """
    input_grids = [example.input_grid.grid for example in task.training_examples + task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples]
    if test_test:
        expected_outputs += [example.output_grid.grid for example in task.test_examples]
    else:
        expected_outputs += [None] * len(task.test_examples)
    return execute_code_with_task(code, input_grids, expected_outputs)


def test_correct(node) -> (bool, bool, list[list[list[int]]]):
    """Test correctness against test examples based on prior execution results."""
    try:
        if not node.valid:
            logger.debug(f"Node {getattr(node, 'id', 'N/A')} is not valid, skipping correctness test.")
            return True, False, []

        num_training = len(node.task.training_examples)
        if not hasattr(node, 'execution_outputs') or node.execution_outputs is None or len(
                node.execution_outputs) < num_training + len(node.task.test_examples):
            logger.warning(
                f"Node {getattr(node, 'id', 'N/A')} execution outputs missing or incomplete. Cannot test correctness.")
            return True, False, []

        test_outputs_expected = [example.output_grid.grid for example in node.task.test_examples]
        test_outputs_generated = node.execution_outputs[num_training:]

        passed_test = True
        # Ensure lengths match before zipping to avoid index errors if outputs are truncated
        if len(test_outputs_generated) != len(test_outputs_expected):
            logger.warning(
                f"Node {getattr(node, 'id', 'N/A')}: Mismatch between generated ({len(test_outputs_generated)}) and expected ({len(test_outputs_expected)}) test outputs.")
            passed_test = False
        else:
            for generated_output, expected_output in zip(test_outputs_generated, test_outputs_expected):
                if generated_output != expected_output:
                    passed_test = False
                    break

        return False, passed_test, test_outputs_generated

    except AttributeError as e:
        logger.error(f"Error accessing node/task attributes during correctness test: {e}")
        return True, False, []
    except Exception as e:
        logger.error(f"Unexpected error during correctness test: {e}", exc_info=True)
        return True, False, []


def verify_prefixes_and_code(code: str, input_grids: list[list[list[int]]],
                             expected_outputs: list | None) -> tuple[bool, list[bool], bool, bool, list]:
    """Execute each code prefix and the full code, returning aggregated results."""

    import re

    steps = re.split(f"{STEP_END}", code)
    prefix_errors: list[bool] = []

    for k in range(1, len(steps) + 1):
        prefix_code = remove_markers("".join(steps[:k]))
        err, _, _ = execute_code_with_task(prefix_code, input_grids, [None] * len(input_grids))
        prefix_errors.append(err)

    err_full, passed_full, results_full = execute_code_with_task(remove_markers(code), input_grids, expected_outputs)
    success = not any(prefix_errors) and not err_full and passed_full
    return success, prefix_errors, err_full, passed_full, results_full
