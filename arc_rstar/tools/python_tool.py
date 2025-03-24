import logging
import multiprocessing as mp
import traceback

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


def _execute_code_worker(code: str, input_grids: list[list[list[int]]],
                         expected_outputs: list[list[list[int]]], result_queue: mp.Queue):
    """Worker function to run in a separate process."""
    import numpy as np
    import sys
    from io import StringIO

    # Set memory limits
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES * 2))
    except (ImportError, AttributeError):
        logger.warning("Could not set memory limits! ")

    # Capture stdout/stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        # Execute the code to define the solve function
        namespace = {'__builtins__': __builtins__, 'np': np, 'numpy': np}
        exec(code, namespace)

        if 'solve' not in namespace:
            raise NameError("Function 'solve' not defined in the code")

        # Process each grid
        results = []
        passed = True

        for i, grid in enumerate(input_grids):
            # Call the solve function
            result = namespace['solve'](grid)

            # Convert numpy arrays to lists if needed
            if isinstance(result, np.ndarray):
                result = result.tolist()

            results.append(result)

            # Check if results match expected outputs
            if expected_outputs and result != expected_outputs[i]:
                passed = False

        # Send success result
        result_queue.put({
            'success': True,
            'results': results,
            'passed': passed,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
        })

    except Exception as e:
        # Send error information
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
        })
    finally:
        # Reset stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def execute_code_with_task(code: str, input_grids: list[list[list[int]]],
                           expected_outputs: list[list[list[int]]]) -> (bool, bool, list[list[list[int]]]):
    """
    Execute code against multiple input grids in a single in-memory subprocess.

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

    # Prepare the code
    code = prepare_code_for_execution(code)

    # Use spawn context for better isolation
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_code_worker,
        args=(code, input_grids, expected_outputs, result_queue)
    )

    try:
        # Start and wait with timeout
        process.start()
        process.join(TIMEOUT_SECONDS)

        # Handle timeout
        if process.is_alive():
            logger.warning(f"Code execution timed out after {TIMEOUT_SECONDS} seconds")
            process.terminate()
            process.join(0.5)
            if process.is_alive():
                process.kill()
            return True, False, []

        # Process finished - get result
        if result_queue.empty():
            logger.error("Process ended but no result was returned")
            return True, False, []

        result = result_queue.get()

        # Process the result
        if result['success']:
            if result['stdout']:
                logger.debug(f"Code stdout: {result['stdout']}")
            if result['stderr']:
                logger.debug(f"Code stderr: {result['stderr']}")
            return False, result['passed'], result['results']
        else:
            logger.error(f"Error in executed code: {result['error']}")
            logger.debug(f"Traceback: {result['traceback']}")
            return True, False, []

    except Exception as e:
        logger.error(f"Exception during code execution: {str(e)}")
        return True, False, []
    finally:
        # Ensure process is terminated
        if process.is_alive():
            process.terminate()
            try:
                process.join(0.5)
                if process.is_alive():
                    process.kill()
            except Exception as e:
                logger.error(f"Exception during code execution: {str(e)}")


def run_training_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all training examples in a single process."""
    input_grids = [example.input_grid.grid for example in task.training_examples]
    expected_outputs = [example.output_grid.grid for example in task.training_examples]

    return execute_code_with_task(code, input_grids, expected_outputs)


def run_test_examples(task, code: str) -> (bool, bool, list[list[list[int]]]):
    """Run code against all test examples in a single process."""
    input_grids = [example.input_grid.grid for example in task.test_examples]
    expected_outputs = [example.output_grid.grid for example in task.test_examples
                        if example.output_grid is not None]

    return execute_code_with_task(code, input_grids, expected_outputs)
