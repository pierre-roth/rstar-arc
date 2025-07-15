import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
from typing import Any, Dict, List, Set, Tuple

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import CODE, CODE_END, STEP_END
from utils import setup_logging
from data_utils import write_batch_data
from rstar_deepthink.tools import verify_prefixes_and_code

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Hardcode configuration variables for easy modification.
DATASET_NAME = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
# MODEL_NAME = "o4-mini"
MODEL_NAME = "gpt-4.1-mini"
# MODEL_NAME = "gpt-4.1"
# MODEL_NAME = "mistralai/devstral-small"
# MODEL_NAME = "deepseek/deepseek-chat-v3-0324"
# MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"
REASONING_EFFORT = "low"  # "low", "medium", or "high"
MAX_WORKERS = 16  # Number of parallel requests to the API
OUTPUT_FILE = "/Users/piroth/Downloads/output_dataset.jsonl"
PROCESSED_TASKS_FILE = "/Users/piroth/Downloads/processed_tasks.txt"
SKIP_PROBABILITY = 0.95
START_INDEX = 0

OPENAI_MODELS = ["o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]


# --- Pydantic Schema for OpenAI Structured Output ---
# This class defines the structure of the JSON object we expect from the LLM.
# It ensures that the API response is predictable and easy to parse.
class LLMSolution(BaseModel):
    """
    Pydantic model for the expected JSON output from the LLM.
    The LLM is instructed to return a JSON object with a single key, "solution_code".
    """
    solution_code: str = Field(
        description=f"The rewritten Python code, wrapped in {CODE} and {CODE_END} tags, with steps separated by {STEP_END}."
    )


# --- COLOR MAPPING ---
# This class definition is provided to the LLM in the prompt so it can map color names to integers.
COLOR_CLASS_DEFINITION = """
class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    
    ALL_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

# --- PROMPT TEMPLATE ---
# This is the detailed prompt that instructs the LLM on how to perform the code transformation.
SYSTEM_PROMPT = """
You are an expert Python programmer specializing in code refactoring and simplification for a specific data format.
Your task is to rewrite a given Python function to adhere to a new format.
You must follow all instructions precisely. Your output MUST be a JSON object conforming to the provided schema, containing only the refactored code.
"""

if MODEL_NAME.startswith("o"):
    USER_PROMPT_TEMPLATE = f"""
    **CONTEXT:**
    I have a Python function that solves a small visual reasoning puzzle. The original function uses numpy arrays and a custom `Color` enum. I need you to rewrite it based on the following strict rules.

    **INSTRUCTIONS:**

    1.  **Function Signature:** The rewritten function MUST be named `solve` and accept a single argument `I`, which is a Python `list[list[int]]`. It must return a `list[list[int]]`. The type hints in the final code are not required, but the logic must match this signature.

    2.  **Input Handling:** The first step inside the `solve` function MUST be to convert the input list `I` into a numpy array. For example: `import numpy as np; input_grid = np.array(I, dtype=int)`.

    3.  **Output Handling:** The final calculated grid, which you MUST name `O`, is a numpy array. It must be converted back to a Python list before being returned. For example: `return O.tolist()`.

    4.  **Code Structure (VERY IMPORTANT):**
        * The entire solution code MUST be wrapped in these tags: `{CODE}` and `{CODE_END}`. (not indented)
        * The logic inside the function should be broken down into logical steps.
        * Each step MUST be preceded by a detailed comment explaining the step's purpose and the intuition behind it.
        * Each step's code block MUST end with the tag: `{STEP_END}`. (indented like the code in the function)
        * The final return statement does not require a `{STEP_END}` tag.

    5.  **Prefix Verification:** Your code is checked after every `{STEP_END}` marker. Ensure each step is valid Python so all prefixes execute without errors.

    6.  **Integer-only Domain (VERY IMPORTANT):**
        * The original code uses a `Color` enum. You must "translate" all logic to use integers directly.
        * Do NOT include the `Color` class or any color names (e.g., "blue", "red") in your rewritten code or its comments.
        * Use the integer values directly. Here is the mapping for your reference:
            ```python
            {COLOR_CLASS_DEFINITION}
            ```
        * For example, if the original code says `if pixel == Color.BLUE:`, your code MUST say `if pixel == 1:`. If it says `colors = [Color.RED, Color.GREEN]`, your code MUST say `colors = [2, 3]`.

    7. **Non standard functions:**
        * There are some helper functions used. Assume the code works as is and you shouldn't need to change them.
        * Don't change the way the functions are used and called as you cannot know exactly what they do.

    **EXAMPLE OF DESIRED OUTPUT STRUCTURE:**

    {CODE}
    def solve(I):
        # Step 1: Convert the input list to a numpy array for easier manipulation.
        import numpy as np
        input_grid = np.array(I, dtype=int)
        {STEP_END}

        # Step 2: Create a copy of the input grid to modify.
        O = np.copy(input_grid)
        {STEP_END}

        # Step 3: Explain the core logic here.
        # ... more python code for the main logic ...
        {STEP_END}

        # potentially more steps here...

        # Step 4: Return the final grid as a list of lists.
        return O.tolist()
    {CODE_END}

    Obviously the output is expected in a single line JSON object. Only unfolded, it would look like above. 

    **YOUR TASK:**

    Rewrite the following Python code according to all the instructions above. Respond ONLY with a JSON object containing the 'solution_code' key.

    The original code might additionally contain imports at the top level and also comments explaining the logic in more detail. 
    Ignore the imports if they are part of these: 
    from common import *
    import numpy as np
    from typing import *

    Otherwise, move them inside the `solve` function, and ensure they are valid Python code.

    Use the additional comments to write better comments in the rewritten code, but do not include comments not linked to a step in the final output.
    
    However, if you feel the way it was rewritten is not optimal, you can change the way it is written, but it must still follow the rules above. You may write certain things more elegantly, but the logic must remain the same.

    **Original Code to Rewrite:**
    {{original_code}}
    """

else:
    USER_PROMPT_TEMPLATE = f"""
**CONTEXT:**
I have a Python function that solves a small visual reasoning puzzle. The original function uses numpy arrays and a custom `Color` enum. I need you to rewrite it based on the following strict rules.

**INSTRUCTIONS:**

1.  **Function Signature:** The rewritten function MUST be named `solve` and accept a single argument `I`, which is a Python `list[list[int]]`. It must return a `list[list[int]]`. The type hints in the final code are not required, but the logic must match this signature.

2.  **Input Handling:** The first step inside the `solve` function MUST be to convert the input list `I` into a numpy array. For example: `import numpy as np; input_grid = np.array(I, dtype=int)`.

3.  **Output Handling:** The final calculated grid, which you MUST name `O`, is a numpy array. It must be converted back to a Python list before being returned. For example: `return O.tolist()`.

4.  **Code Structure (VERY IMPORTANT):**
    * The entire solution code MUST be wrapped in these tags: `{CODE}` and `{CODE_END}`. (not indented)
    * The logic inside the function should be broken down into logical steps.
    * Each step MUST be preceded by a detailed comment explaining the step's purpose and the intuition behind it.
    * Each step's code block MUST end with the tag: `{STEP_END}`. (indented like the code in the function)
    * The final return statement does not require a `{STEP_END}` tag.

5.  **Prefix Verification:** Your code is checked after every `{STEP_END}` marker. Ensure each step is valid Python so all prefixes execute without errors.

6.  **Integer-only Domain (VERY IMPORTANT):**
    * The original code uses a `Color` enum. You must "translate" all logic to use integers directly.
    * Do NOT include the `Color` class or any color names (e.g., "blue", "red") in your rewritten code or its comments.
    * Use the integer values directly. Here is the mapping for your reference:
        ```python
        {COLOR_CLASS_DEFINITION}
        ```
    * For example, if the original code says `if pixel == Color.BLUE:`, your code MUST say `if pixel == 1:`. If it says `colors = [Color.RED, Color.GREEN]`, your code MUST say `colors = [2, 3]`.

7. **Non standard functions:**
    * There are some helper functions used. Assume the code works as is and you shouldn't need to change them.
    * Don't change the way the functions are used and called as you cannot know exactly what they do.

**EXAMPLE OF DESIRED OUTPUT STRUCTURE:**

{CODE}
def solve(I):
    # Step 1: Convert the input list to a numpy array for easier manipulation.
    import numpy as np
    input_grid = np.array(I, dtype=int)
    {STEP_END}

    # Step 2: Create a copy of the input grid to modify.
    O = np.copy(input_grid)
    {STEP_END}

    # Step 3: Explain the core logic here.
    # ... more python code for the main logic ...
    {STEP_END}
    
    # potentially more steps here...

    # Step 4: Return the final grid as a list of lists.
    return O.tolist()
{CODE_END}

Obviously the output is expected in a single line JSON object. Only unfolded, it would look like above. 

**YOUR TASK:**

Rewrite the following Python code according to all the instructions above. Respond ONLY with a JSON object containing the 'solution_code' key.

The original code might additionally contain imports at the top level and also comments explaining the logic in more detail. 
Ignore the imports if they are part of these: 
from common import *
import numpy as np
from typing import *

Otherwise, move them inside the `solve` function, and ensure they are valid Python code.

Use the additional comments to write better comments in the rewritten code, but do not include comments not linked to a step in the final output.


**Original Code to Rewrite:**
{{original_code}}
"""


# --- HELPER FUNCTIONS ---

def load_processed_tasks(filename: str) -> Set[str]:
    """Loads processed task names from a line-based text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def save_processed_task(task_name: str, filename: str):
    """Appends a processed task name to the tracking file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(task_name + "\n")


def split_source_code(source_code: str) -> Tuple[str, str]:
    """Splits the source code into the main logic and the generator function."""
    if "def generate_input()" in source_code:
        parts = source_code.split("def generate_input()", 1)
        main_logic = parts[0].strip()
        generator_code = "def generate_input()" + parts[1]
        return main_logic, generator_code
    return source_code, ""


def format_examples(examples: List[List[List[Any]]]) -> List[Dict[str, List[List[int]]]]:
    """Converts floats to ints and reformats the example structure."""
    formatted = []
    for pair in examples:
        input_grid = [[int(round(cell)) for cell in row] for row in pair[0]]
        output_grid = [[int(round(cell)) for cell in row] for row in pair[1]]
        formatted.append({"input": input_grid, "output": output_grid})
    return formatted


# --- WORKER FUNCTION ---

def process_item(args: Tuple[int, Dict], client: OpenAI, processed_tasks: Set[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Processes a single item from the dataset. This function is called by the thread pool.
    """
    index, item = args
    task_name = f"{index:08x}"

    if task_name in processed_tasks:
        return False, None  # Indicate that this was skipped

    try:
        # 1. Pre-process the data
        main_logic, generator_code = split_source_code(item['source'])
        formatted_examples = format_examples(item['examples'])

        if not formatted_examples:
            return False, {"task_name": task_name, "error": "No valid examples found"}

        # 2. Construct the LLM prompt
        prompt = USER_PROMPT_TEMPLATE.format(original_code=main_logic)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        if MODEL_NAME in OPENAI_MODELS:
            # 3. Call the OpenAI API using the structured output feature
            response = client.responses.parse(
                model=MODEL_NAME,
                input=messages,
                text_format=LLMSolution,
                reasoning={"effort": REASONING_EFFORT} if MODEL_NAME.startswith("o") else None,
            )

            # The .parse() helper automatically checks for refusals and other issues.
            # If we get here, response.output_parsed is a valid LLMSolution object.
            parsed_output = response.output_parsed
            solution_code = parsed_output.solution_code
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Extract the JSON string from the standard response format
            json_string = response.choices[0].message.content

            # Manually parse the JSON string using the Pydantic model
            parsed_output = LLMSolution.model_validate_json(json_string)
            solution_code = parsed_output.solution_code

        # 4. Verify the generated code with prefix checks
        input_grids = [ex["input"] for ex in formatted_examples]
        expected_outputs = [ex["output"] for ex in formatted_examples]
        success, _, err_full, passed_full, _ = verify_prefixes_and_code(
            solution_code, input_grids, expected_outputs
        )

        final_data = {
            "task_name": task_name,
            "solution_code": solution_code,
            "examples": formatted_examples,
            "generator": generator_code,
        }

        if not (success and not err_full and passed_full):
            return False, final_data

        return True, final_data

    except Exception as e:
        # Catch any other exception during processing
        # print(f"Error processing task {task_name}: {e}")
        # traceback.print_exc()
        return False, {"task_name": task_name, "error": str(e)}


# --- MAIN ORCHESTRATOR ---

def main():
    """
    Main function to orchestrate the dataset loading, processing, and saving.
    """
    setup_logging(logging.INFO)
    logger.info("--- Starting Dataset Reformatting Script ---")

    # Load API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if MODEL_NAME in OPENAI_MODELS:
        client = OpenAI(api_key=openai_api_key, timeout=120)
    else:
        client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1", timeout=120)

    # Load the dataset in streaming mode to avoid holding everything in memory
    logger.info(f"Loading dataset: {DATASET_NAME} ...")
    try:
        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    total_tasks = ds.info.splits["train"].num_examples
    logger.info(f"Dataset loaded successfully with {total_tasks} total tasks.")

    # Load the set of tasks that have already been processed
    processed_tasks = load_processed_tasks(PROCESSED_TASKS_FILE)
    logger.info(f"Found {len(processed_tasks)} previously processed tasks.")

    logger.info(f"Total tasks in dataset: {total_tasks}")
    processed_count = len(processed_tasks)

    attempted_count = 0
    success_count = 0

    # Process tasks sequentially while keeping a small pool of workers
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending_futures = set()
        with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
            for index, item in enumerate(ds):
                task_name = f"{index:08x}"

                if index < START_INDEX or task_name in processed_tasks or random.random() < SKIP_PROBABILITY:
                    pbar.update(1)
                    continue

                future = executor.submit(process_item, (index, item), client, processed_tasks)
                pending_futures.add(future)

                if len(pending_futures) >= MAX_WORKERS:
                    done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        attempted_count += 1
                        success, result_data = fut.result()
                        # print(success, result_data)
                        if success:
                            success_count += 1
                            write_batch_data(OUTPUT_FILE, [result_data])
                            processed_tasks.add(result_data['task_name'])
                            save_processed_task(result_data['task_name'], PROCESSED_TASKS_FILE)
                            processed_count += 1
                            logger.info(f"Successfully converted task: {result_data['task_name']}")
                        else:
                            logger.info(f"Failed to convert task: {result_data['task_name']}")

                        failed_count = attempted_count - success_count
                        success_rate = success_count / attempted_count if attempted_count else 0

                        logger.info(
                            f"Run stats - attempted: {attempted_count}, succeeded: {success_count}, "
                            f"failed: {failed_count}, success rate: {success_rate:.2%}, "
                            f"total success: {processed_count}"
                        )
                        pbar.update(1)

            # Handle any remaining futures
            if pending_futures:
                for fut in as_completed(pending_futures):
                    attempted_count += 1
                    success, result_data = fut.result()
                    if success:
                        success_count += 1
                        write_batch_data(OUTPUT_FILE, [result_data])
                        processed_tasks.add(result_data['task_name'])
                        save_processed_task(result_data['task_name'], PROCESSED_TASKS_FILE)
                        processed_count += 1
                        logger.info(f"Successfully converted task: {result_data['task_name']}")
                    else:
                        logger.info(f"Failed to convert task: {result_data['task_name']}")

                    failed_count = attempted_count - success_count
                    success_rate = success_count / attempted_count if attempted_count else 0

                    logger.info(
                        f"Run stats - attempted: {attempted_count}, succeeded: {success_count}, "
                        f"failed: {failed_count}, success rate: {success_rate:.2%}, "
                        f"total success: {processed_count}"
                    )
                    pbar.update(1)

    logger.info("--- Script finished successfully! ---")


if __name__ == "__main__":
    main()
