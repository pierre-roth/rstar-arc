from vllm.model_executor.models.minicpmv import CPU_DEVICE

ETH_USERNAME = "piroth"

NET_SCRATCH_PATH = f"/itet-stor/{ETH_USERNAME}/net_scratch"  # net-scratch directory
LOCAL_SCRATCH_PATH = f"/scratch/{ETH_USERNAME}"  # local scratch directory
SECOND_LOCAL_SCRATCH_PATH = f"/scratch-second/{ETH_USERNAME}"  # second local scratch directory
HOME_PATH = f"/home/{ETH_USERNAME}"  # home directory

PROJECT_PATH = f"{HOME_PATH}/rstar-arc"  # project root directory
DEFAULT_DATA_FOLDER = f"{PROJECT_PATH}/data_sample"  # path for sample data in git repo
DEFAULT_DATA_PATH = f"{DEFAULT_DATA_FOLDER}/default"
DEFAULT_EXAMPLE_DATA_PATH = f"{DEFAULT_DATA_FOLDER}/examples"  # path to prompt examples

CPU_TIMEOUT_SECONDS = 10  # Maximum cpu time allowed for code execution
WALL_TIMEOUT_SECONDS = 60  # Maximum wall time allowed for code execution
MEMORY_LIMIT_MB = 128  # Maximum memory allowed for each code execution process
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
