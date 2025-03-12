import sys
import traceback
from io import StringIO
import contextlib
import numpy as np
import re
import signal
from constants import TIMEOUT_SECONDS, TIMEOUT_MESSAGE


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException(TIMEOUT_MESSAGE)


@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr"""
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def safe_exec(code_string, globals_dict=None, locals_dict=None, timeout=TIMEOUT_SECONDS):
    """
    Safely execute Python code with timeout protection.
    
    Args:
        code_string: Python code to execute
        globals_dict: Global variables dictionary
        locals_dict: Local variables dictionary
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary with execution results, output, and error information
    """
    if globals_dict is None:
        globals_dict = {}
    
    # Set up the execution environment
    exec_globals = {
        'np': np,
        '__builtins__': __builtins__,
    }
    exec_globals.update(globals_dict)
    
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    result = {
        'success': False,
        'output': '',
        'error': '',
        'result': None,
        'locals': {}
    }
    
    try:
        with capture_output() as (out, err):
            exec(code_string, exec_globals, locals_dict)
            
        # Get the output
        result['output'] = out.getvalue()
        result['error'] = err.getvalue()
        result['success'] = True
        
        # Store the local variables for later use
        if locals_dict is not None:
            result['locals'] = locals_dict
        
    except TimeoutException as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = traceback.format_exc()
    finally:
        # Cancel the alarm
        signal.alarm(0)
    
    return result


def extract_python_code(text):
    """
    Extract Python code from text that may contain markdown or other content.
    
    Args:
        text: Text containing Python code
        
    Returns:
        Extracted Python code
    """
    # Look for code blocks in Markdown format (```python ... ```)
    code_block_pattern = r'```(?:python)?\s*([\s\S]*?)\s*```'
    code_blocks = re.findall(code_block_pattern, text)
    
    if code_blocks:
        # Join all code blocks
        return '\n'.join(code_blocks)
    
    # If no code blocks are found, look for <code> tags
    code_tag_pattern = r'<code>([\s\S]*?)</code>'
    code_tags = re.findall(code_tag_pattern, text)
    
    if code_tags:
        return '\n'.join(code_tags)
    
    # If neither is found, assume the entire text might be code
    # (this is a fallback, not ideal)
    return text


def execute_code_with_grid(code, input_grid):
    """
    Execute Python code with a grid as input and return the transformed grid.
    
    Args:
        code: Python code to execute
        input_grid: Input grid as a list of lists or numpy array
        
    Returns:
        Dictionary with execution results and transformed grid
    """
    # Create a dictionary for local variables
    locals_dict = {'input_grid': input_grid, 'output_grid': None}
    
    # Extract Python code if needed
    clean_code = extract_python_code(code)
    
    # Execute the code
    result = safe_exec(clean_code, locals_dict=locals_dict)
    
    # Try to get the output grid from locals
    if result['success']:
        if 'output_grid' in locals_dict and locals_dict['output_grid'] is not None:
            result['grid'] = locals_dict['output_grid']
        else:
            # Look for the last variable assignment that could be the output
            lines = clean_code.strip().split('\n')
            for line in reversed(lines):
                if '=' in line and not line.strip().startswith('#'):
                    var_name = line.split('=')[0].strip()
                    if var_name in locals_dict and isinstance(locals_dict[var_name], (list, np.ndarray)):
                        result['grid'] = locals_dict[var_name]
                        break
    
    return result