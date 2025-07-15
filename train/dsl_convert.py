import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, Set, Tuple

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
# DATASET_PATH = "/Users/piroth/Downloads/round_2_2/policy_dataset_validation.jsonl"
DATASET_PATH = "/Users/piroth/Downloads/dsl_convertable.jsonl"
MODEL_NAME = "o4-mini"
# MODEL_NAME = "gpt-4.1"
# MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"
REASONING_EFFORT = "high"  # "low", "medium", or "high"
MAX_WORKERS = 1  # Number of parallel requests to the API
OUTPUT_FILE = "/Users/piroth/Downloads/additional_dsl_dataset.jsonl"
PROCESSED_TASKS_FILE = "/Users/piroth/Downloads/additional_dsl_processed_tasks.txt"

OPENAI_MODELS = ["o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3"]


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


with open("common.py", "r", encoding="utf-8") as f:
    COMMON = f.read().replace("{", "{{").replace("}", "}}")


# --- PROMPT TEMPLATE ---
# This is the detailed prompt that instructs the LLM on how to perform the code transformation.
SYSTEM_PROMPT = """
You are an expert Python programmer specializing in code rewriting, refactoring and simplification for a specific data format.
Your task is to rewrite a given Python function to use a given set of DSL functions if possible, all while adhering to strict formatting and structural rules.
You must follow all instructions precisely. Your output MUST be a JSON object conforming to the provided schema, containing only the refactored code.
"""

USER_PROMPT_TEMPLATE = f"""
    **CONTEXT:**
    I have a step-by-step Python function that solves a small visual reasoning puzzle. The original function only pure python and numpy. I need you to rewrite it to, where possible, use the provided DSL functions to simplify the code.

    **INSTRUCTIONS:**

    1.  **Function Signature:** The rewritten function MUST be named `solve` and accept a single argument `I`, which is a Python `list[list[int]]`. It must return a `list[list[int]]`. The type hints in the final code are not required, but the logic must match this signature.

    2.  **Input Handling:** Most DSL functions are written to use numpy arrays, so you could make the first step inside the `solve` function to be a conversion of input I into a numpy array. For example: `import numpy as np; input_grid = np.array(I, dtype=int)`.

    3.  **Output Handling:** The final calculated grid, which you MUST name `O`, is potentially a numpy array. It must be converted back to a Python list before being returned. For example: `return O.tolist()`.

    4.  **Code Structure (VERY IMPORTANT):**
        * The entire solution code MUST be wrapped in these tags: `{CODE}` and `{CODE_END}`. (not indented)
        * The logic inside the function should be broken down into logical steps.
        * Each step MUST be preceded by a detailed comment explaining the step's purpose and the intuition behind it.
        * Each step's code block MUST end with the tag: `{STEP_END}`. (indented like the code in the function)
        * The final return statement does not require a `{STEP_END}` tag.

    5.  **Prefix Verification:** Your code is checked after every `{STEP_END}` marker. Ensure each step is valid Python so all prefixes execute without errors.

    6.  **Integer-only Domain (VERY IMPORTANT):**
        * The DSL uses a `Color` enum. The code you must rewrite doesn't. You must stick to only using integer values directly!
        * Do NOT include the `Color` class or any color names (e.g., "blue", "red") in your rewritten code or its comments.
        * Use the integer values directly!
        
    7. **DSL functions:**
        * Use the DSL functions provided in the `common.py` file where possible.

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
    Also, the output must only be converted with .tolist() if it is a numpy array, otherwise it should be returned as is.
    
    Below is the content of the `common.py` file that contains the DSL functions you can use:
{COMMON}
    
    **YOUR TASK:**

    Rewrite the following Python code according to all the instructions above. Respond ONLY with a JSON object containing the 'solution_code' key.

    You can assume that certain libraries are already imported as follows such that the function may use them directly:
    from common import *
    import numpy as np
    from typing import *
    
    You can import any other additional standard libraries you need, but you must do so inside the `solve` function.
    
    This is a complicated problem so take your time and make sure the new program is semantically equivalent to the original one, but uses the DSL functions where possible.
    It is up to your discretion to decide whether to use a DSL function or not, but you must follow the rules above. Your decision should be based on the appropriateness of the DSL function for the task at hand.
    There may be tasks, where not using the DSL at all is fine as there are no correctly fitting DSL functions available. Emphasize semantic equivalence over strict adherence to DSL usage. It is fine to sometimes not use the DSL functions if they do not fit the task.
    
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


# --- WORKER FUNCTION ---
def process_item(args: Tuple[int, Dict], client: OpenAI, processed_tasks: Set[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Processes a single item from the dataset. This function is called by the thread pool.
    """
    index, item = args
    task_name = item['task_name']
    # task_name = item['original_task_name']

    try:
        # 1. Pre-process the data
        # main_logic = item['solution']
        task_json = item['task_json']
        weight = item.get('weight', 1.0)
        main_logic = item['solution']
        # examples = item['examples']

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
        input_grids = [ex["input"] for ex in task_json['train']] + [ex["input"] for ex in task_json['test']]
        expected_outputs = [ex["output"] for ex in task_json['train']] + [ex["output"] for ex in task_json['test']]

        # input_grids = [ex["input"] for ex in examples]
        # expected_outputs = [ex["output"] for ex in examples]

        success, _, err_full, passed_full, _ = verify_prefixes_and_code(
            solution_code, input_grids, expected_outputs
        )

        final_data = {
            "task_name": task_name,
            "task_json": task_json,
            "solution": solution_code,
            "weight": weight,
            "index": str(index),
        }

        """final_data = {
            "task_name": f"{index:08x}",
            "original_task_name": task_name,
            "solution_code": solution_code,
            "examples": examples,
            "index": str(index),
        }"""

        if not (success and not err_full and passed_full):
            return False, final_data

        return True, final_data

    except Exception as e:
        # Catch any other exception during processing
        # print(f"Error processing task {task_name}: {e}")
        # traceback.print_exc()
        return False, {"index": str(index), "task_name": task_name, "error": str(e)}


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
        client = OpenAI(api_key=openai_api_key)
    else:
        client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")

    # Load the dataset in streaming mode to avoid holding everything in memory
    logger.info(f"Loading dataset: {DATASET_PATH} ...")
    try:
        ds = load_dataset("json", data_files=DATASET_PATH, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Dataset loaded successfully with {total_lines} total lines.")

    # Load the set of tasks that have already been processed
    processed_lines = load_processed_tasks(PROCESSED_TASKS_FILE)
    logger.info(f"Found {len(processed_lines)} previously processed lines.")

    logger.info(f"Total tasks in dataset: {total_lines}")
    processed_count = len(processed_lines)

    attempted_count = 0
    success_count = 0

    # Process tasks sequentially while keeping a small pool of workers
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending_futures = set()
        with tqdm(total=total_lines, desc="Processing tasks") as pbar:
            for index, item in enumerate(ds):

                if str(index) in processed_lines:
                    pbar.update(1)
                    continue

                future = executor.submit(process_item, (index, item), client, processed_lines)
                pending_futures.add(future)

                if len(pending_futures) >= MAX_WORKERS:
                    done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        attempted_count += 1
                        success, result_data = fut.result()
                        if success:
                            # print(success, result_data['original_task_name'], result_data['solution_code'])
                            success_count += 1
                            processed_lines.add(result_data['index'])
                            save_processed_task(result_data['index'], PROCESSED_TASKS_FILE)
                            processed_count += 1
                            logger.info(f"Successfully converted task: {result_data['index']}")
                            result_data.pop('index', None)  # Remove index from result data
                            write_batch_data(OUTPUT_FILE, [result_data])
                        else:
                            logger.info(f"Failed to convert task: {result_data['index']}")

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
                done, _ = wait(pending_futures)
                for fut in done:
                    attempted_count += 1
                    success, result_data = fut.result()
                    if success:
                        # print(success, result_data['original_task_name'], result_data['solution_code'])
                        success_count += 1
                        processed_lines.add(result_data['index'])
                        save_processed_task(result_data['index'], PROCESSED_TASKS_FILE)
                        processed_count += 1
                        logger.info(f"Successfully converted task: {result_data['index']}")
                        result_data.pop('index', None)  # Remove index from result data
                        write_batch_data(OUTPUT_FILE, [result_data])
                    else:
                        logger.info(f"Failed to convert task: {result_data['index']}")

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
