import json
import os

from constants import CODE, CODE_END, STEP_END, EXAMPLE_DATA_FOLDER, BOOTSTRAP_SYSTEM_PROMPT
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


def get_example_prompt(config: Config, task_names: list[str]) -> str:
    """Generate the initial prompt for the task to feed into the LLM."""
    tasks = []
    for task_name in task_names:
        path = os.path.join(EXAMPLE_DATA_FOLDER, f"{task_name}.json")
        tasks.append(ARCTask(config, path))

    # open "solution.jsonl" file and scan through it line by line until we find the task name
    solution_codes = {name: "" for name in task_names}
    with open(os.path.join(EXAMPLE_DATA_FOLDER, "solutions.jsonl"), "r") as f:
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
