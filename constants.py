# Constants used throughout the application

# Timeouts
TIMEOUT_SECONDS = 15
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

# Error messages
ERROR_COLOR = "red"
TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
TOO_MANY_STEPS = "Fail to solve the problem within limited steps."
NO_VALID_CHILD = "Fail to generate parsable text for next step."

# Output format tags
OUTPUT = "<o>"
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
