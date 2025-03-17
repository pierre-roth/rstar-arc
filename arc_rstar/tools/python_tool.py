import sys
import signal
import os
import tempfile
import json
import subprocess
from io import StringIO
import contextlib
import numpy as np
import re
from config import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, CODE, CODE_END, STEP_END, MEMORY_LIMIT_BYTES


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException(TIMEOUT_MESSAGE)


def remove_thinking_blocks(text, verbose=False):
    """
    Remove all <think>...</think> blocks from text.

    Args:
        text (str): Text that may contain thinking blocks
        verbose (bool): Whether to print debug information

    Returns:
        str: Text with thinking blocks removed
    """
    import re

    original_length = 0
    if verbose:
        original_length = len(text)
        num_blocks = len(re.findall(r'<think>', text))
        print(f"Found {num_blocks} thinking blocks in text of length {original_length}")

    # Pattern to match <think>...</think> blocks, including nested content
    pattern = r'<think>.*?</think>'

    # Replace all thinking blocks with empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    if verbose:
        new_length = len(cleaned_text)
        print(f"Removed {original_length - new_length} characters worth of thinking blocks")

    return cleaned_text


def extract_python_code(text, verbose=False):
    """Extract Python code from text after the last CODE marker, removing any CODE_END or STEP_END markers."""

    # try removing thinking tokens before running the code
    text = remove_thinking_blocks(text, verbose)

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


def create_subprocess_script(code, temp_dir=None):
    """Create a temporary script file with the user's code and I/O handling.

    Args:
        code (str): The Python code to execute
        temp_dir (str, optional): Directory where the temporary script will be created.
                                 If None, system default temp directory is used.

    Returns:
        str: Path to the created script file
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.py', prefix='execute_', dir=temp_dir)

    # Prepare the wrapper script that:
    # 1. Sets up memory limits
    # 2. Loads the input grid
    # 3. Executes the user's code
    # 4. Captures stdout/stderr
    # 5. Returns the result as JSON
    wrapper_script = f"""
import sys
import json
import traceback
import numpy as np

# Set memory limit
try:
    import resource
    # Set the memory limit to {MEMORY_LIMIT_BYTES} bytes (soft limit)
    resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_LIMIT_BYTES}, {MEMORY_LIMIT_BYTES * 2}))
except ImportError:
    # resource module not available (e.g., Windows)
    pass

# Load input data
try:
    input_data = json.loads(sys.stdin.read())
    input_grid = input_data.get('grid')
except Exception as e:
    print(f"Error loading input data: {{e}}", file=sys.stderr)
    sys.exit(1)

# Store original stdout and stderr
import io
original_stdout = sys.stdout
original_stderr = sys.stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

# Execute user code
result = None
error = None
try:
{indent_code(code)}

    # Call the solve function with the input grid
    result = solve(input_grid)

    # Convert numpy arrays to lists
    if isinstance(result, np.ndarray):
        result = result.tolist()

except Exception as e:
    error = {{"type": str(type(e).__name__), "message": str(e), "traceback": traceback.format_exc()}}

# Restore stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

# Prepare output
output = {{
    "result": result,
    "stdout": stdout_capture.getvalue(),
    "stderr": stderr_capture.getvalue(),
    "error": error
}}

# Print output as JSON
print(json.dumps(output, default=lambda x: None))
"""

    try:
        with os.fdopen(fd, 'w') as f:
            f.write(wrapper_script)
        return path
    except Exception as e:
        if os.path.exists(path):
            os.unlink(path)
        raise e


def indent_code(code, spaces=4):
    """Properly indent code for inclusion in the wrapper script.

    Args:
        code (str): The Python code to indent
        spaces (int): Number of spaces for indentation

    Returns:
        str: Properly indented code
    """
    # Split code into lines
    lines = code.rstrip().split('\n')
    # Indent each line
    indented_lines = [' ' * spaces + line for line in lines]
    # Join back into a string
    return '\n'.join(indented_lines)


def execute_code_with_grid(code, input_grid, verbose=False, temp_dir=None):
    """Execute Python code with the provided input grid using a subprocess for isolation.

    Args:
        code (str): The Python code to execute
        input_grid (list): The input grid to pass to the solve function
        verbose (bool, optional): Whether to print debug information
        temp_dir (str, optional): Directory where temporary files will be created.
                                 If None, system default temp directory is used.

    Returns:
        list or None: The result grid if execution was successful, None otherwise
    """
    if not code.strip():
        if verbose:
            print("Cannot execute empty code")
        return None

    # Prepare code
    code = prepare_code(code)

    if verbose:
        print(f"Executing code with grid of shape {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")

    script_path = None
    try:
        # Create a temporary directory for our script if not provided
        created_temp_dir = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            created_temp_dir = True
        elif not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        # Create the script file
        script_path = create_subprocess_script(code, temp_dir)

        if verbose:
            print(f"Created temporary script at {script_path}")

        # Prepare input data as JSON
        input_data = {"grid": input_grid}
        input_json = json.dumps(input_data)

        # Run the subprocess with timeout
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=-1
        )

        # Send input data
        stdout, stderr = process.communicate(input=input_json, timeout=TIMEOUT_SECONDS)

        if verbose and stderr:
            print(f"Subprocess stderr: {stderr}")

        # Process returned successfully
        if process.returncode == 0:
            try:
                # Parse the JSON output
                output = json.loads(stdout)

                if verbose:
                    if output.get("stdout"):
                        print(f"Code stdout output:\n{output['stdout']}")
                    if output.get("stderr"):
                        print(f"Code stderr output:\n{output['stderr']}")

                # Check if there was an error
                if output.get("error"):
                    if verbose:
                        print(f"Error in executed code: {output['error']['message']}")
                        print(f"Traceback: {output['error']['traceback']}")
                    return None

                # Get the result
                result = output.get("result")

                # Validate result is a 2D grid
                if not (isinstance(result, list) and
                        (not result or all(isinstance(row, list) for row in result))):
                    if verbose:
                        print(f"Invalid result type: {type(result)}")
                        if isinstance(result, list):
                            print(f"Result is list but contains non-list elements or is empty")
                    return None

                if verbose:
                    print(
                        f"Code execution successful, result grid shape: {len(result)}x{len(result[0]) if result and result[0] else 0}")

                return result
            except json.JSONDecodeError:
                if verbose:
                    print(f"Failed to decode subprocess output as JSON: {stdout}")
                return None
        else:
            if verbose:
                print(f"Subprocess exited with code {process.returncode}")
            return None

    except subprocess.TimeoutExpired:
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
        # Clean up temporary files
        if script_path and os.path.exists(script_path):
            try:
                os.unlink(script_path)
            except:
                pass
