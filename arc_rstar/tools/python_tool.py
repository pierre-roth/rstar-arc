import sys
import signal
from io import StringIO
import contextlib
import numpy as np
import re
from config import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, CODE, CODE_END, STEP_END


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException(TIMEOUT_MESSAGE)


def extract_python_code(text):
    """Extract Python code from text between CODE and CODE_END markers.
    Specifically extracts the last code block."""
    pattern = re.compile(f"{CODE}(.*?){CODE_END}", re.DOTALL)
    matches = list(pattern.finditer(text))

    # Check if there's at least one code block
    if not matches:
        raise ValueError(f"No code blocks found in text")

    # Return the last code block
    return matches[-1].group(1).strip()


def prepare_code(code):
    """Prepare code by removing STEP_END markers and ensuring it returns a value."""
    # Simple string replacement for STEP_END markers
    clean_code = code.replace(f"{STEP_END}", "")

    # If the code doesn't contain a complete solve function (no 'return' statement)
    if 'def solve(I):' in clean_code and 'return' not in clean_code:
        clean_code += "\n    return []"

    return clean_code


@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr."""
    old_out, old_err = sys.stdout, sys.stderr
    new_out, new_err = StringIO(), StringIO()
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield new_out, new_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def execute_code_with_grid(code, input_grid):
    """Execute Python code with the provided input grid."""
    if not code.strip():
        return None

    # Prepare code
    code = prepare_code(code)

    # Set up execution environment
    execution_globals = {
        'np': np,
        'input_grid': input_grid,
        'result': None
    }

    # Add wrapper to call solve function
    wrapper_code = f"{code}\n\nresult = solve(input_grid)"

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        # Execute code
        with capture_output():
            exec(wrapper_code, execution_globals)

        # Get result
        result = execution_globals.get('result')

        # Convert numpy arrays to lists
        if isinstance(result, np.ndarray):
            result = result.tolist()

        # Validate result is a 2D grid
        if not (isinstance(result, list) and
                (not result or all(isinstance(row, list) for row in result))):
            return None

        return result

    except TimeoutException:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)  # Cancel alarm in all cases
