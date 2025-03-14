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


def extract_python_code(text, verbose=False):
    """Extract Python code from text after the last CODE marker, removing any CODE_END or STEP_END markers."""
    import sys
    
    if verbose:
        print(f"Extracting code from text (which has {len(text)} characters)")
        
    # Check if text contains the CODE marker
    if CODE not in text:
        if verbose:
            print(f"CODE marker not found in text")
        raise ValueError(f"No CODE marker found in text")
    
    # Find the last CODE marker and get all content after it
    last_code_start = text.rindex(CODE) + len(CODE)
    code = text[last_code_start:].strip()
    
    if not code:
        raise ValueError(f"No code was extracted after the last CODE marker")
    
    if verbose:
        print(f"Extracted code block (with {len(code)} characters, {len(code.splitlines())} lines)")
        print(f"Code block:\n{code}")
        
    return code


def prepare_code(code):
    """Prepare code by removing STEP_END and CODE_END markers and ensuring it returns a value."""
    # Remove both types of markers
    clean_code = code.replace(f"{STEP_END}", "").replace(f"{CODE_END}", "")

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


def execute_code_with_grid(code, input_grid, verbose=False):
    """Execute Python code with the provided input grid."""
    import sys
    
    if not code.strip():
        if verbose:
            print("Cannot execute empty code")
        return None

    # Prepare code
    code = prepare_code(code)
    
    if verbose:
        print(f"Executing code with grid of shape {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")
        
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
        with capture_output() as (stdout, stderr):
            exec(wrapper_code, execution_globals)
            
            if verbose and stdout.getvalue():
                print(f"Code stdout output:\n{stdout.getvalue()}")
            if verbose and stderr.getvalue():
                print(f"Code stderr output:\n{stderr.getvalue()}")

        # Get result
        result = execution_globals.get('result')

        # Convert numpy arrays to lists
        if isinstance(result, np.ndarray):
            if verbose:
                print(f"Converting numpy array of shape {result.shape} to list")
            result = result.tolist()

        # Validate result is a 2D grid
        if not (isinstance(result, list) and
                (not result or all(isinstance(row, list) for row in result))):
            if verbose:
                print(f"Invalid result type: {type(result)}")
                if isinstance(result, list):
                    print(f"Result is list but contains non-list elements or is empty")
            return None
            
        if verbose:
            print(f"Code execution successful, result grid shape: {len(result)}x{len(result[0]) if result and result[0] else 0}")

        return result

    except TimeoutException:
        if verbose:
            print(f"Code execution timed out after {TIMEOUT_SECONDS} seconds")
        return None
    except Exception as e:
        if verbose:
            print(f"Exception during code execution: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return None
    finally:
        signal.alarm(0)  # Cancel alarm in all cases
