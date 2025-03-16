from arc_rstar.arc_task.task import ARCTask
from config import *


def get_prompt(config: Config, task: ARCTask) -> str:
    example_task = ARCTask(str(os.path.join(config.data_folder, "6d0aefbc.json")), config)

    prompt = f"""You are a powerful agent with broad problem solving/pattern matching knowledge and great python programming skills. 
You need to write Python code to solve ARC (Abstraction and Reasoning Corpus) tasks.

ARC Task Description:
    1. ARC tasks are rectangular grids of integers.
    2. Each integer represents a color and there are only 10 color values (0-9).
    3. Each task is made up of training and test input-output examples.
    4. Your task is to write Python code that can transform the input grids into their corresponding output grids.
    5. You will get access to the training input-output examples to learn the transformation function.
    6. You need to write code that can generate the correct output grids for the test input grids.
    
Remember:
    1. Write code that implements the transformation function step by step. The solution should include {CODE} {CODE_END} and intermediate steps.
    2. The final code block should be valid Python code and implement the function `solve(I: list[list[int]]) -> list[list[int]]`. This function transforms input grids into their corresponding output grids.
    3. You may use numpy functions (it is imported as "import numpy as np")
    5. Always generate the next step and the next step only, that it up to the {STEP_END} marker.
    6. Each step must be valid Python code. Steps can be as simple as a single line of code or as complex as a multi-line function.
    7. Each step,combined with the steps before it, however must be a valid Python code block i.e. no partial code blocks.
    8. If you generate a {CODE_END} marker instead of a {STEP_END} marker, this signals the end of the code block, and thus the end of the transformation function.
    9. Please use the following template:

Below follows an example task and solution. You need to write code to solve the task in the same format.

{example_task.to_prompt()}


{CODE}
def solve(I):
    
    # vertically mirror the grid
    x1 = [row[::-1] for row in I]
    {STEP_END}
    
    # horizontally concatenate the input grid with the vertically mirrored grid
    O = [I[i] + x1[i] for i in range(len(I))]
    {STEP_END}
    
    # return the output grid
    return 0
{CODE_END}





Now it's your turn! Write code to solve the task below. The code above is just an example and will not solve the task below.
You need to come up with a solution that solves the task below. Use the input output example to infer the transformation function!

{task.to_prompt()}


{CODE}
def solve(I):"""

    return prompt
