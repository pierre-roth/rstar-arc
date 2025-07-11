import json
import logging
import os
import signal
import subprocess
import sys
import textwrap

# import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import CPU_TIMEOUT_SECONDS, WALL_TIMEOUT_SECONDS, CODE, CODE_END, STEP_END, MEMORY_LIMIT_BYTES

logger = logging.getLogger(__name__)

GLOBAL_IMPORTS = (
    "from train.common import *\n"
    "import numpy as np\n"
    "from typing import *\n"
)


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
    """
    wrapper_code = textwrap.dedent(f"""
        import resource
        import sys
        import json

        # Set memory limit
        try:
            mem_limit = {MEMORY_LIMIT_BYTES}
            # resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            resource.setrlimit(resource.RLIMIT_RSS, (mem_limit, mem_limit))
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
            exec_globals = {{}}
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
                    
                    if grid_result is not None:
                        if not isinstance(grid_result, list) or not all(isinstance(row, list) for row in grid_result):
                            raise ValueError("The result must be a 2D list (grid).")
                        elif not len(grid_result) <= 30 or not all(len(row) <= 30 for row in grid_result):
                            raise ValueError("The result grid must not exceed 30x30 in size.")
                    
                    results.append(grid_result)

                    if i < len(expected_outputs) and expected_outputs[i] is not None:
                         if grid_result != expected_outputs[i]:
                            passed = False
                            break
                except Exception as e:
                    print(f"Error processing grid {{i}}: {{str(e)}}", file=sys.stderr)
                    results.append(None)
                    passed = False
                    error_occurred = True
                    break
            
            while len(results) < len(input_grids):
                results.append(None)
            
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
            logger.debug(
                f"Subprocess OK but failed to parse JSON output. Stdout: {stdout}. Most likely there is a print statement making the output invalid JSON.")
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
    Executes user code in the current Python process.
    Mirrors the behaviour of `execute_code_in_subprocess`, but without
    spawning an external interpreter, so it still lacks full isolation.
    Returns (error_occurred, passed, results).
    """
    results: list = []
    passed = True
    error_occurred = False
    exec_globals: dict = {}

    try:
        # Compile and execute the submission – defines solve()
        exec(code_str, exec_globals)

        if "solve" not in exec_globals:
            logger.debug("Direct execution failed: Function 'solve' not defined.")
            return True, False, []

        solve_func = exec_globals["solve"]

        # Normalise expected_outputs length
        if expected_outputs is None:
            expected_outputs = [None] * len(input_grids)

        for i, grid in enumerate(input_grids):
            try:
                grid_result = solve_func(grid)

                # JSON-safe conversion for numpy arrays (no module import needed)
                if hasattr(grid_result, "tolist"):
                    grid_result = grid_result.tolist()

                if grid_result is not None:
                    if not isinstance(grid_result, list) or not all(isinstance(row, list) for row in grid_result):
                        raise ValueError("The result must be a 2D list (grid).")
                    elif not len(grid_result) <= 30 or not all(len(row) <= 30 for row in grid_result):
                        raise ValueError("The result grid must not exceed 30x30 in size.")

                results.append(grid_result)

                # Early-stop on first wrong answer
                if (expected_outputs[i] is not None and
                        grid_result != expected_outputs[i]):
                    passed = False
                    break
            except Exception as e:
                logger.debug(f"Error on grid {i}: {e}")
                results.append(None)
                passed = False
                error_occurred = True
                break

    except Exception as e:
        logger.debug(f"Error executing submission: {e}")
        error_occurred = True
        passed = False

    # Pad results so caller always receives len(input_grids) items
    if len(results) < len(input_grids):
        results.extend([None] * (len(input_grids) - len(results)))

    return error_occurred, passed, results


def execute_code_with_task(
        code: str,
        input_grids: list[list[list[int]]],
        expected_outputs: list,
        config=None
) -> (bool, bool, list[list[list[int]]]):
    """
    Execute code against multiple input grids using either a subprocess or direct execution.
    """
    if not code.strip():
        logger.debug("Cannot execute empty code!")
        return True, False, []

    code = GLOBAL_IMPORTS + code

    environment_var_sub = os.environ.get("RSTAR_SUBPROCESS_EXECUTION")
    use_subprocess_execution = environment_var_sub is not None and environment_var_sub == "1"

    if use_subprocess_execution or (config is not None and config.execute_in_subprocess):
        logger.debug("Executing code via subprocess.")
        return execute_code_in_subprocess(code, input_grids, expected_outputs)
    else:
        logger.debug("Executing code directly.")
        return execute_code_directly(code, input_grids, expected_outputs)


def run_examples(
        task,
        code: str,
        test_test: bool = False,
        config=None
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
    return execute_code_with_task(code, input_grids, expected_outputs, config=config)


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


"""def verify_prefixes_and_code(code: str, input_grids: list[list[list[int]]],
                             expected_outputs: list | None) -> tuple[bool, list[bool], bool, bool, list]:

    import re

    steps = re.split(f"{STEP_END}", code)

    if len(steps) < 3:
        logger.debug("Insufficient steps! The code must contain at least two steps!")
        return False, [], True, False, []

    if len(input_grids) > 50:
        input1 = input_grids[:50]
        output1 = expected_outputs[:50] if expected_outputs else None
        input2 = input_grids[50:]
        output2 = expected_outputs[50:] if expected_outputs else None

        success1, prefix_errors1, err_full1, passed_full1, results_full1 = verify_prefixes_and_code(code, input1,
                                                                                                    output1)
        success2, prefix_errors2, err_full2, passed_full2, results_full2 = verify_prefixes_and_code(code, input2,
                                                                                                    output2)

        return success1 and success2, prefix_errors1 + prefix_errors2, err_full1 or err_full2, passed_full1 and passed_full2, results_full1 + results_full2

    prefix_errors: list[bool] = []

    for k in range(1, len(steps) + 1):
        prefix_code = remove_markers("".join(steps[:k]))
        err, _, _ = execute_code_with_task(prefix_code, input_grids, [None] * len(input_grids))
        prefix_errors.append(err)

    err_full, passed_full, results_full = execute_code_with_task(remove_markers(code), input_grids, expected_outputs)
    success = not any(prefix_errors) and not err_full and passed_full
    return success, prefix_errors, err_full, passed_full, results_full"""


def verify_prefixes_and_code(
        code: str,
        input_grids: list[list[list[int]]],
        expected_outputs: list | None,
        config=None
) -> tuple[bool, list[bool], bool, bool, list]:
    """
    Execute each code prefix and the full code, returning aggregated results.
    Uses batching to avoid command-line argument length limits.
    """
    import re

    steps = re.split(f"({STEP_END})", code)
    # Re-join the delimiter to the preceding step to correctly form prefixes
    steps = ["".join(s) for s in zip(steps[0::2], steps[1::2] + [""])]

    if not steps or not steps[0].strip():
        return False, [], True, False, []

    # --- 1. Prefix Validation ---
    # Check prefixes on a single grid to be fast and avoid arg length errors.
    prefix_errors: list[bool] = []
    prefix_check_inputs = input_grids[:1]
    # We only check prefixes, not the full code here.
    for k in range(1, len(steps)):
        current_prefix_code = "".join(steps[:k])
        prefix_code_to_run = remove_markers(current_prefix_code)
        if not prefix_code_to_run.strip():
            prefix_errors.append(False)  # Empty prefix is not an error
            continue
        err, _, _ = execute_code_with_task(prefix_code_to_run, prefix_check_inputs, [None], config=config)
        prefix_errors.append(err)

    if any(prefix_errors):
        # Fail fast if any prefix is invalid. No need to run the full code.
        logger.debug("Prefix validation failed. Aborting full execution.")
        # Pad prefix_errors to match number of steps for consistency if needed by caller
        while len(prefix_errors) < len(steps) - 1:
            prefix_errors.append(True)
        return False, prefix_errors, True, False, []

    # --- 2. Full Code Execution in Batches ---
    BATCH_SIZE = 50
    full_code = remove_markers(code)
    all_results = []
    overall_passed = True
    any_error_occurred = False

    # Ensure expected_outputs is a list of the same length as input_grids for batching
    if expected_outputs is None:
        expected_outputs = [None] * len(input_grids)

    for i in range(0, len(input_grids), BATCH_SIZE):
        input_batch = input_grids[i:i + BATCH_SIZE]
        output_batch = expected_outputs[i:i + BATCH_SIZE]

        err_full, passed_full, results_full = execute_code_with_task(full_code, input_batch, output_batch,
                                                                     config=config)

        if err_full:
            any_error_occurred = True
        if not passed_full:
            overall_passed = False

        all_results.extend(results_full)

        # If a critical error occurs in a batch, we can stop early.
        if any_error_occurred:
            logger.debug(f"Error occurred in batch starting at index {i}. Stopping execution.")
            # Fill remaining results with None to maintain correct length
            remaining_count = len(input_grids) - len(all_results)
            all_results.extend([None] * remaining_count)
            break

        if not overall_passed:
            logger.debug(f"Batch starting at index {i} did not pass. Stopping execution.")
            # Fill remaining results with None to maintain correct length
            remaining_count = len(input_grids) - len(all_results)
            all_results.extend([None] * remaining_count)
            break

    success = not any(prefix_errors) and not any_error_occurred and overall_passed
    return success, prefix_errors, any_error_occurred, overall_passed, all_results
