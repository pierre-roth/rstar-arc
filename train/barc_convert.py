import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Set

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from constants import CODE, CODE_END, STEP_END

# --- CONFIGURATION ---
# Hardcode configuration variables for easy modification.
DATASET_NAME = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
MODEL_NAME = "o4-mini"
REASONING_EFFORT = "low"  # "low", "medium", or "high"
MAX_WORKERS = 1  # Number of parallel requests to the API
OUTPUT_FILE = "/Users/piroth/Downloads/output_dataset.jsonl"
PROCESSED_TASKS_FILE = "/Users/piroth/Downloads/processed_tasks.json"


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
    * Each step MUST be preceded by a detailed comment explaining the step's purpose.
    * Each step's code block MUST end with the tag: `{STEP_END}`. (indented like the code in the function)
    * The final return statement does not require a `{STEP_END}` tag.

5.  **Integer-only Domain (VERY IMPORTANT):**
    * The original code uses a `Color` enum. You must "translate" all logic to use integers directly.
    * Do NOT include the `Color` class or any color names (e.g., "blue", "red") in your rewritten code or its comments.
    * Use the integer values directly. Here is the mapping for your reference:
        ```python
        {COLOR_CLASS_DEFINITION}
        ```
    * For example, if the original code says `if pixel == Color.BLUE:`, your code MUST say `if pixel == 1:`. If it says `colors = [Color.RED, Color.GREEN]`, your code MUST say `colors = [2, 3]`.

6. **Non standard functions:**
    * There are some helper functions used. Assume the code works as is and you shouldn't need to change them.

**EXAMPLE OF DESIRED OUTPUT STRUCTURE:**

```python
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
```

**YOUR TASK:**

Rewrite the following Python code according to all the instructions above. Respond ONLY with a JSON object containing the 'solution_code' key.

**Original Code to Rewrite:**
```python
{{original_code}}
```
"""


# --- HELPER FUNCTIONS ---

def load_processed_tasks(filename: str) -> Set[str]:
    """Loads the set of already processed task names from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def save_processed_task(task_name: str, processed_set: Set[str], filename: str):
    """Adds a task name to the processed set and rewrites the file."""
    processed_set.add(task_name)
    with open(filename, 'w') as f:
        json.dump(list(processed_set), f, indent=4)


def append_to_jsonl(data: Dict, filename: str):
    """Appends a JSON object to a JSONL file."""
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')


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


def verify_solution(solution_code: str, example: Dict) -> bool:
    """
    Executes the generated solution code against an example to verify its correctness.
    Returns True if the code runs and produces the correct output, False otherwise.
    """
    try:
        # Prepare a clean environment for execution
        exec_globals = {"np": np}
        # The tags are not python code, so we remove them before executing.
        code_to_exec = solution_code.strip().replace(CODE, "").replace(CODE_END, "")

        # Define the 'solve' function in our clean environment
        exec(code_to_exec, exec_globals)
        solve_func = exec_globals['solve']

        # Run the function with the example input
        output = solve_func(example['input'])

        # Check if the generated output matches the expected output
        return output == example['output']
    except Exception:
        # If any error occurs during execution or verification, log it and return False.
        # print(f"Verification failed for task. Error: {e}")
        # traceback.print_exc()
        return False


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

        # 3. Call the OpenAI API using the structured output feature
        response = client.responses.parse(
            model=MODEL_NAME,
            input=messages,
            text_format=LLMSolution,
            reasoning={"effort": REASONING_EFFORT}
        )

        # The .parse() helper automatically checks for refusals and other issues.
        # If we get here, response.output_parsed is a valid LLMSolution object.
        parsed_output = response.output_parsed
        solution_code = parsed_output.solution_code

        # 4. Verify the generated code
        is_verified = verify_solution(solution_code, formatted_examples[0])
        if not is_verified:
            return False, {"task_name": task_name, "error": "Verification failed"}

        # 5. Assemble the final JSON object
        final_data = {
            "task_name": task_name,
            "solution_code": solution_code,
            "examples": formatted_examples,
            "generator": generator_code,
        }
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
    print("--- Starting Dataset Reformatting Script ---")

    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    client = OpenAI(api_key=api_key)

    # Load the dataset from Hugging Face
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split='train', streaming=False)  # Use streaming=False for tqdm
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Dataset loaded successfully.")

    # Load the set of tasks that have already been processed
    processed_tasks = load_processed_tasks(PROCESSED_TASKS_FILE)
    print(f"Found {len(processed_tasks)} previously processed tasks.")

    # Use enumerate to get an index for each item for the task_name
    all_tasks = list(enumerate(ds))
    tasks_to_process = [task for task in all_tasks if f"{task[0]:08x}" not in processed_tasks]

    if not tasks_to_process:
        print("All tasks have already been processed. Exiting.")
        return

    print(f"Total tasks to process: {len(tasks_to_process)}")

    # Process tasks in parallel using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a progress bar with tqdm
        with tqdm(total=len(tasks_to_process), desc="Processing tasks") as pbar:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(process_item, task, client, processed_tasks): task
                for task in tasks_to_process
            }

            for future in as_completed(future_to_task):
                task_index = future_to_task[future][0]
                task_name = f"{task_index:08x}"
                try:
                    success, result_data = future.result()
                    if success:
                        # On successful processing, save the results
                        append_to_jsonl(result_data, OUTPUT_FILE)
                        save_processed_task(result_data['task_name'], processed_tasks, PROCESSED_TASKS_FILE)
                    # else:
                    # Optionally log failures or skips
                    # if result_data:
                    #     print(f"Failed or skipped task {result_data.get('task_name', 'N/A')}: {result_data.get('error', 'Skipped')}")

                except Exception as e:
                    # Log exceptions that might occur from the future itself
                    # print(f"An exception occurred for task {task_name}: {e}")
                    # traceback.print_exc()
                    pass

                pbar.update(1)

    print("--- Script finished successfully! ---")


if __name__ == "__main__":
    main()
