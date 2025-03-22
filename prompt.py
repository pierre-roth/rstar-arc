from arc_rstar.arc_task.task import ARCTask
from config import *


def get_prompt(config: Config, task: ARCTask) -> str:
    ### PROMPT PREFIX ###
    prompt_prefix = f"""You are a powerful agent with broad problem solving/pattern matching knowledge and great python programming skills. You need to write Python code to solve an ARC (Abstraction and Reasoning Corpus) task, or more specifcally implement the transformation function that can transform the input grids into their corresponding output grids.

ARC Task Description:
    - ARC tasks are composed of a set of training input-output examples and a set of test input grids.
    - Each grid is a 2D list of integers and is given to you as a list of lists. (argument of the function "solve")
    - Each integer represents a "color" and there are only 10 color values (0-9).
    - Your task is to write Python code that can transform the input grids into their corresponding output grids.
    - You will get access to the training input-output examples to learn the transformation function.
    - The transformation function "solve" must be able to correctly transform the training input grids into their corresponding output grids.
    - The transformation function "solve" must also be able to correctly transform the test input grids into their corresponding hidden output grids.
    
Remember:
    - Write code that implements the transformation function step by step. The solution must include {CODE}, {CODE_END} and {STEP_END} markers appropriately.
    - The final code block must be valid Python code and implement the function `solve(I: list[list[int]]) -> list[list[int]]`. This function transforms input grids into their corresponding output grids.
    - You may use Python built-in functions and libraries.
    - You may use numpy functions (it is imported as "import numpy as np")
    - Always generate the next step and the next step only, that it up and including the {STEP_END} marker.
    - Each step must be valid Python code. Steps can be as simple as a single line of code or as complex as a multi-line function.
    - Each step, combined with the steps before it, however must be a valid Python code block i.e. no partial code blocks.
    - If you generate a {CODE_END} marker instead of a {STEP_END} marker, this signals the end of the code block, and thus the end of the transformation function.
    - Please use the following template:

{CODE}
def solve(I):
    
    # comment explaining the step
    python code for the step
    {STEP_END}
    
    # comment explaining the step
    python code for the step
    {STEP_END}
    
    # comment explaining the step
    O = the correct output grid
    {STEP_END}
    
    # return the output grid
    return O
{CODE_END}"""

    ### PROMPT SUFFIX ###
    prompt_suffix = f"""Now it's your turn! Carefully analyze the input-output examples to infer the transformation function.
Then write Python code to implement the transformation function.

{task.to_prompt()}


{CODE}
def solve(I):"""

    ### EXAMPLE PROMPTS ###
    single_example_prompt = f"""Below is an example task with an example solution. They should give you an idea of what is expected."""
    multiple_example_prompt = f"""Below are {config.num_examples} example tasks with example solutions. They should give you an idea of what is expected."""

    ### EXAMPLES ###
    example_task_1 = ARCTask(config, str(os.path.join(config.data_folder, "6d0aefbc.json")))
    solution_code_1 = f"""{CODE}
def solve(I):
    
    # vertically mirror the grid
    x1 = [row[::-1] for row in I]
    {STEP_END}
    
    # horizontally concatenate the input grid with the vertically mirrored grid
    O = [I[i] + x1[i] for i in range(len(I))]
    {STEP_END}
    
    # return the output grid
    return O
{CODE_END}"""

    example_task_2 = ARCTask(config, str(os.path.join(config.data_folder, "1cf80156.json")))
    solution_code_2 = f"""{CODE}
def solve(I):
    # Convert input to numpy array for easier slicing
    I_np = np.array(I)
    {STEP_END}
    
    # Find rows containing non-zero values
    non_zero_rows = [i for i in range(len(I)) if any(val != 0 for val in I[i])]
    min_row, max_row = min(non_zero_rows), max(non_zero_rows)
    {STEP_END}
    
    # Find columns containing non-zero values
    non_zero_cols = [j for j in range(len(I[0])) if any(I[i][j] != 0 for i in range(len(I)))]
    min_col, max_col = min(non_zero_cols), max(non_zero_cols)
    {STEP_END}
    
    # Extract the subgrid using numpy slicing
    O = I_np[min_row:max_row+1, min_col:max_col+1].tolist()
    {STEP_END}
    
    # Return the output subgrid
    return O
{CODE_END}"""

    example_task_3 = ARCTask(config, str(os.path.join(config.data_folder, "00d62c1b.json")))
    solution_code_3 = f"""{CODE}
def solve(I):
    # Define helper function to find a connected component using BFS
    def find_component(grid, start_row, start_col, visited):
        height, width = len(grid), len(grid[0])
        value = grid[start_row][start_col]
        component = []
        queue = [(start_row, start_col)]
        
        while queue:
            r, c = queue.pop(0)
            if (r < 0 or r >= height or c < 0 or c >= width or 
                visited[r][c] or grid[r][c] != value):
                continue
            
            visited[r][c] = True
            component.append((r, c))
            
            # Add the four adjacent neighbors (no diagonals)
            queue.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                
        return value, component
    {STEP_END}
    
    # Define helper function to check if a component touches the border
    def is_border_touching(component, height, width):
        for r, c in component:
            if r == 0 or r == height-1 or c == 0 or c == width-1:
                return True
        return False
    {STEP_END}
    
    # Copy the input grid and get dimensions
    output = [row[:] for row in I]
    height, width = len(I), len(I[0])
    {STEP_END}
    
    # Initialize visited cells tracker
    visited = [[False for _ in range(width)] for _ in range(height)]
    {STEP_END}
    
    # Find all connected components in the grid
    components = []
    for i in range(height):
        for j in range(width):
            if not visited[i][j]:
                value, component = find_component(I, i, j, visited)
                if component:
                    components.append((value, component))
    {STEP_END}
    
    # Find all color 0 components that don't touch the border
    non_border_components = []
    for value, component in components:
        if value == 0 and not is_border_touching(component, height, width):
            non_border_components.append(component)
    {STEP_END} 
    
    # Fill all cells in non-border components with value 4
    for component in non_border_components:
        for r, c in component:
            output[r][c] = 4
    {STEP_END}
    
    # Return the transformed grid
    return output
{CODE_END}"""

    ### COMBINE PROMPT ###
    examples = [(example_task_1, solution_code_1), (example_task_2, solution_code_2), (example_task_3, solution_code_3)]

    assert config.num_examples <= len(
        examples), f"Number of examples requested ({config.num_examples}) exceeds the number of available examples ({len(examples)})!"

    if config.num_examples == 0:
        return prompt_prefix + "\n\n" + prompt_suffix
    else:
        prompt = prompt_prefix + "\n\n"

        if config.num_examples == 1:
            prompt += f"{single_example_prompt}\n\n"
        else:
            prompt += f"{multiple_example_prompt}\n\n"

        for i, (example_task, solution_code) in enumerate(examples[:config.num_examples]):
            prompt += f"Example {i + 1}:\n\n"
            prompt += example_task.to_prompt() + "\n\n"
            prompt += solution_code + "\n\n"

        prompt += prompt_suffix
        return prompt
