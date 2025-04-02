import json
import os

from constants import CODE, CODE_END, STEP_END, DEFAULT_EXAMPLE_DATA_PATH
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config


def task_to_prompt(task: ARCTask) -> str:
    """Generate the initial prompt for the task to feed into the LLM."""
    prompt = ["## Training Examples\n"]

    for i, example in enumerate(task.training_examples):
        prompt.append(f"### Training Example {i + 1}")
        prompt.append(f"Input shape: {example.input_grid.rows} x {example.input_grid.columns}")
        prompt.append("Input:\n")
        prompt.append(str(example.input_grid) + "\n")
        prompt.append(f"Output shape: {example.input_grid.rows} x {example.input_grid.columns}")
        prompt.append("Output:\n")
        prompt.append(str(example.output_grid) + "\n\n")

    prompt.append("## Test Examples\n")

    for i, example in enumerate(task.test_examples):
        prompt.append(f"### Test Example {i + 1}")
        prompt.append(f"Input shape: {example.input_grid.rows} x {example.input_grid.columns}")
        prompt.append("Input:\n")
        prompt.append(str(example.input_grid) + "\n")
        prompt.append(f"Output shape: determined by solve function")
        prompt.append("Output:\n")
        prompt.append("Result of applying solve function to the input grid" + "\n\n")

    return "\n".join(prompt)


# noinspection PyUnusedLocal
def get_base_prompt(config: Config, task: ARCTask) -> (str, str):
    ### PROMPT PREFIX ###
    prompt_prefix = f"""You are a powerful agent with broad problem solving skills, pattern matching abilities and great python programming expertise. You need to write Python code to solve an ARC (Abstraction and Reasoning Corpus) task, or more specifically implement the transformation function that can transform the input grids into their corresponding output grids.

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
    - Each step must be valid Python code. Steps can be as simple as a single line of code or as complex as a multi-line function.
    - If you generate a {CODE_END} marker instead of a {STEP_END} marker, this signals the end of the code block, and thus the end of the transformation function.
    - It is important to accurately analyze the input-output examples to infer the transformation function "solve".
    - The transformation function "solve" might be completely different from the given example solution.
    - Look to the example for ideas and for guidance on how to correctly format your solution!

"""

    ### PROMPT SUFFIX ###
    prompt_suffix = f"""Now it's your turn! Carefully analyze the input-output examples to infer the transformation function.
Then write Python code to implement the transformation function.

{task_to_prompt(task)}

"""

    return prompt_prefix, prompt_suffix


def get_example_prompt(config: Config, task_names: list[str]) -> str:
    """Generate the initial prompt for the task to feed into the LLM."""
    tasks = []
    for task_name in task_names:
        path = os.path.join(DEFAULT_EXAMPLE_DATA_PATH, f"{task_name}.json")
        tasks.append(ARCTask(config, path))

    # open "solution.jsonl" file and scan through it line by line until we find the task name
    solution_codes = {name: "" for name in task_names}
    with open(os.path.join(DEFAULT_EXAMPLE_DATA_PATH, "solutions.jsonl"), "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data["task_name"] in task_names:
                solution_codes[data["task_name"]] = data["solution_code"]

    if any(solution_code == "" for solution_code in solution_codes.values()):
        raise ValueError(f"Not all listed examples have solution!")

    if len(task_names) == 1:
        example_prompt = f"""Below is one example task with solution. This should give you an idea of what a solution looks like and what format you should adhere to.

{task_to_prompt(tasks[0])}

{solution_codes[task_names[0]]}

"""
    else:
        example_prompt = f"""Below are {len(tasks)} example tasks with solutions. They should give you an idea of what a solution looks like and what format you should adhere to.\n"""

        for i, task in enumerate(tasks):
            example_prompt += f"\n### Example Task {i + 1} ###\n"
            example_prompt += task_to_prompt(task)
            example_prompt += "\n"
            example_prompt += solution_codes[task_names[i]]
            example_prompt += "\n"

        example_prompt += "\n"

    return example_prompt
