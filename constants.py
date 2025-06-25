ETH_USERNAME = "piroth"

HOME_PATH = f"/home/{ETH_USERNAME}"  # home directory
PROJECT_PATH = f"{HOME_PATH}/rstar-arc"  # project root directory

NET_SCRATCH_PATH = f"/itet-stor/{ETH_USERNAME}/net_scratch"  # net-scratch directory
LOCAL_SCRATCH_PATH = f"/scratch/{ETH_USERNAME}"  # local scratch directory
SECOND_LOCAL_SCRATCH_PATH = f"/scratch-second/{ETH_USERNAME}"  # second local scratch directory (doesn't exist on all nodes)

DATA_SAMPLE_DIR = f"{PROJECT_PATH}/data_sample"  # path for sample data in git repo
DEFAULT_DATA_FOLDER = f"{DATA_SAMPLE_DIR}/default"
EXAMPLE_DATA_FOLDER = f"{DATA_SAMPLE_DIR}/examples"  # path to prompt examples

NET_SCRATCH_MODEL_DIR = f"{NET_SCRATCH_PATH}/models"  # path for model in net-scratch
NET_SCRATCH_SFT_DATA_DIR = f"{NET_SCRATCH_PATH}/sft_data"  # path for sft data in net-scratch
NET_SCRATCH_TASK_DATA_DIR = f"{NET_SCRATCH_PATH}/task_data"  # path for task data in net-scratch
NET_SCRATCH_RE_ARC_DATA = f"{NET_SCRATCH_PATH}/re_arc"  # path to the re_arc directory

CPU_TIMEOUT_SECONDS = 10  # Maximum cpu time allowed for code execution
WALL_TIMEOUT_SECONDS = 60  # Maximum wall time allowed for code execution
MEMORY_LIMIT_MB = 512  # Maximum memory allowed for each code execution process
MEMORY_LIMIT_BYTES = MEMORY_LIMIT_MB * 1024 * 1024

# terminal reasons in order of priority
TERMINAL_INVALID = "Invalid code"  # Terminal marker for invalid code
TERMINAL_CODE_END = "Code end marker"  # Terminal marker for end of code tag
TERMINAL_MAX_DEPTH = "Maximum depth reached"  # Terminal marker for reaching max depth
TERMINAL_SUBTREE_TERMINAL = "Subtree terminal"  # Terminal marker for subtree terminal

# special markers for language model
STEP_END = "<end_of_step>"  # Marks the end of a reasoning step
CODE = "<beginning_of_code>"  # Begins a code section
CODE_END = "<end_of_code>"  # Ends a code section
SPECIAL_TOKENS = [CODE, CODE_END, STEP_END]

BOOTSTRAP_SYSTEM_PROMPT = f"""You are a powerful agent with broad problem solving skills, pattern matching abilities and great python programming expertise. You need to write Python code to solve an ARC (Abstraction and Reasoning Corpus) task, or more specifically implement the transformation function that can transform the input grids into their corresponding output grids.

ARC Task Description:
    - ARC tasks are composed of a set of training input-output examples and a set of test input grids.
    - Each grid is a 2D list of integers and is given to you as a list of lists. (parameter I of the function "solve")
    - Each integer represents a "color" and there is a total of 10 color values: the value 0 to 9.
    - Your task is to write Python code that can transform the input grids into their corresponding output grids.
    - You will get access to the training input-output examples to learn the transformation function.
    - The transformation function "solve" must be able to correctly transform the training input grids into their corresponding output grids.
    - The transformation function "solve" must also be able to correctly transform the test input grids into their corresponding hidden output grids.

Remember:
    - Write code that implements the transformation function step by step. The solution must include {CODE}, {CODE_END} and {STEP_END} markers appropriately.
    - The final code block must be valid Python code and implement the function `solve(I: list[list[int]]) -> list[list[int]]`. This function transforms input grids into their corresponding output grids.
    - You may use Python built-in functions and the standard libraries.
    - You are allowed to use all python standard library libraries such as collections, itertools, etc. after import them.
    - Additionally, you can use numpy after importing it.
    - Each step must be valid Python code (except {STEP_END}). Steps can be as simple as a single line of code or as complex as a multi-line function.
    - Every step must be valid Python code when combined with the previous steps! This means no "open" loops, if statements, or other incomplete code.
    - If you generate a {CODE_END} marker instead of a {STEP_END} marker, this signals the end of the code block, and thus the end of the transformation function.
    - It is important to accurately analyze the input-output examples to infer the transformation function "solve".
    - The transformation function "solve" might be completely different from the given example solution.
    - Look to the example for ideas and for guidance on how to correctly format your solution!

"""

BOOTSTRAP_TASK_PROMPT = f"""Now it's your turn! Carefully look at the input-output examples to infer the transformation function.
Then write correctly formatted step-by-set Python code with natural language chain-of-thought comments to implement the transformation function.\n"""

SFT_SYSTEM_PROMPT = f"""You are an expert Python agent tasked with solving an ARC puzzle. Each task provides training input-output grids and test inputs. A grid is a 2D list of integers from 0 to 9 representing colors. Deduce the transformation from the training pairs and implement it in a `solve(I)` function. Produce the solution step by step, ending each step with {STEP_END} and finishing with {CODE_END}. The steps must form valid Python when combined. Standard libraries and numpy may be used.\n\n"""
SFT_IN_BETWEEN_PROMPT = """\n\nStep-by-step Python code with brief reasoning comments:\n\n"""

CODE_PREFIX = f"{CODE}\ndef solve(I):\n    "
