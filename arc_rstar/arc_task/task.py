import json
from typing import Dict, Any, List
from config import Config
import numpy as np


class Grid:
    """Class representing a 2D grid of integers"""

    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.rows = len(self.grid)
        self.columns = len(self.grid[0])

        # enforce that all rows have the same number of columns
        assert all(len(row) == self.columns for row in self.grid)

    def __eq__(self, other: "Grid"):
        return self.grid == other.grid

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)


class Example:
    """Class representing an input-output pair in an ARC task"""

    def __init__(self, input_grid: Grid, output_grid: Grid):
        self.input_grid = input_grid
        self.output_grid = output_grid

    def __eq__(self, other: "Example"):
        return self.input_grid == other.input_grid and self.output_grid == other.output_grid

    def __str__(self):
        result = "Input:\n\n"
        result += str(self.input_grid)
        result += "\n\nOutput: \n\n"
        result += str(self.output_grid)
        return result


class ArcTask:
    def __init__(self, path, config=None):
        self.path = path
        self.config = config
        self.name = path.split("/")[-1].split(".")[0]
        self.train_data = []
        self.test_data = []
        self._load_data()

    def _load_data(self):
        """Load task data from the provided path"""
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)

            # Process training data
            if 'train' in data:
                for item in data['train']:
                    self.train_data.append(Example(
                        input_grid=Grid(item['input']),
                        output_grid=Grid(item['output'])
                    ))

            # Process test data
            if 'test' in data:
                for item in data['test']:
                    self.test_data.append(Example(
                        input_grid=Grid(item['input']),
                        output_grid=Grid(item['output'])
                    ))
        except Exception as e:
            print(f"Error loading task data: {e}")

    def __eq__(self, other: "ARCTask"):
        """Check if two ARCTask objects are equal by comparing their train and test data"""

        # Compare training data
        if len(self.train_data) != len(other.train_data):
            return False

        for i in range(len(self.train_data)):
            if self.train_data[i] != other.train_data[i]:
                return False

        # Compare test data
        if len(self.test_data) != len(other.test_data):
            return False

        for i in range(len(self.test_data)):
            if self.test_data[i] != other.test_data[i]:
                return False

        return True

    def __str__(self):
        """String representation of the ARCTask"""
        result = [f"ARCTask {self.name}", f"Number of training examples: {len(self.train_data)}",
                  f"Number of test examples: {len(self.test_data)}", "\nTraining examples: "]

        # Add details for each training example
        for i, example in enumerate(self.train_data):
            result.append(f"\nTraining Example {i + 1}:")
            result.append(str(example))

        result.append("\nTest examples: ")

        # Add details for each test example
        for i, example in enumerate(self.test_data):
            result.append(f"\nTest Example {i + 1}:")
            result.append(str(example))

        return "\n".join(result)

    def get_initial_prompt(self) -> str:
        """Generate the initial prompt for the task to feed into the LLM."""
        prompt = [f"# ARC Task: {self.name}\n", "## Training Examples\n"]

        for i, example in enumerate(self.train_data):
            prompt.append(f"### Training Example {i + 1}")
            prompt.append("Input:")
            prompt.append("```")
            prompt.append(str(example.input_grid))
            prompt.append("```")
            prompt.append("Output:")
            prompt.append("```")
            prompt.append(str(example.output_grid))
            prompt.append("```\n")

        prompt.append("## Test Example\n")

        # For beam search, we'll use the first test example
        if self.test_data:
            prompt.append("Input:")
            prompt.append("```")
            prompt.append(str(self.test_data[0].input_grid))
            prompt.append("```")

        prompt.append("\nYour task is to solve this ARC problem by writing Python code step by step.")
        prompt.append("Analyze the pattern in the training examples, then write code that transforms the test input to "
                      "produce the correct output.")
        prompt.append("Use numpy for grid manipulations and clearly explain your reasoning at each step.")
        prompt.append("Define a function `solve(input_grid)` that returns the output grid.\n")
        prompt.append("The input_grid will be provided as a numpy array. Your solution should return a numpy array as "
                      "output_grid.")

        return "\n".join(prompt)

    def is_solved(self, state: Dict[str, Any]) -> bool:
        """
        Check if the task has been solved based on the state.
        This will execute the Python code to see if it produces the correct output grid.
        
        Args:
            state: The current state dictionary with text containing Python code
            
        Returns:
            True if the solution produces the correct output grid, False otherwise
        """
        from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid

        text = state.get("text", "")

        # Extract the Python code
        code = extract_python_code(text)

        if not code:
            return False

        # Execute the code with the test input
        if not self.test_data:
            return False

        test_input = self.test_data[0].input_grid.grid
        expected_output = self.test_data[0].output_grid.grid

        result = execute_code_with_grid(code, test_input)

        if not result.get('success', False) or 'grid' not in result:
            return False

        # Compare the output with the expected result
        output_grid = result.get('grid')

        # Convert to list of lists if it's a numpy array
        if isinstance(output_grid, np.ndarray):
            output_grid = output_grid.tolist()

        return output_grid == expected_output
