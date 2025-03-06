TIMEOUT_SECONDS = 60
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

ERROR_COLOR = "red"

TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
TOO_MANY_STEPS = "Fail to sove the problem within limited steps."
NO_VALID_CHILD = "Fail to generate parsable text for next step."


OUTPUT = "<output>"
OUTPUT_END = "<end_of_output>"
STEP_END = "<end_of_step>"
CODE = "<code>"
CODE_TAG = "Now print the final answer"
CODE_END = "<end_of_code>"
ANSWER = "<answer>"
ANSWER_END = "<end_of_answer>"
REFINE = "<refine>"
REFINE_PASS = "I am sure that my answer is correct"
REFINE_END = "<end_of_refine>"


OUTPUT_BASE_PATH = "/itet-stor/piroth/net_scratch/outputs"
MODEL_BASE_PATH = "/itet-stor/piroth/net_scratch/models"
DATA_BASE_PATH = "/itet-stor/piroth/net_scratch/data"
DATA_SAMPLE_BASE_PATH = "/itet-stor/piroth/net_scratch/rstar-arc/data_sample"

# Default model configurations
DEFAULT_POLICY_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PP_LLM = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_MAX_TOKENS = 2048

# Default runtime configurations
DEFAULT_VERBOSE = True
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_DEPTH = 10
DEFAULT_SEARCH_MODE = "beam_search"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_DTYPE = "float16"

# Beam search specific configurations
DEFAULT_BEAM_WIDTH = 3


