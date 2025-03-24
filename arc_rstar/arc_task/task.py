import json
import os
from typing import Optional

from arc_rstar.tools.python_tool import execute_code_with_grid
from config import Config


class Grid:
    """Class representing a 2D grid of integers"""

    def __init__(self, grid: list[list[int]]):
        self.grid: list[list[int]] = grid
        self.rows: int = len(self.grid)
        self.columns: int = len(self.grid[0])

    def __eq__(self, other: "Grid"):
        return self.grid == other.grid

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)


class Example:
    """Class representing an input-output pair in an ARC task"""

    def __init__(self, input_grid: Grid, output_grid: Optional[Grid] = None):
        self.input_grid = input_grid
        self.output_grid = output_grid

    def __eq__(self, other: "Example"):
        input_match = self.input_grid == other.input_grid
        if self.output_grid is None or other.output_grid is None:
            return input_match
        return input_match and self.output_grid == other.output_grid

    def __str__(self):
        result = "Input:\n\n"
        result += str(self.input_grid)
        if self.output_grid:
            result += "\n\nOutput: \n\n"
            result += str(self.output_grid)
        else:
            result += "\n\nOutput: Not available"
        return result


class ARCTask:
    def __init__(self, config: Config, path):
        self.config = config

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]

        self.training_examples = []
        self.test_examples = []

        self._load_data()

    def _load_data(self):
        """Load task data from the provided path"""
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)

            # Process training data
            if 'train' in data:
                for item in data['train']:
                    self.training_examples.append(Example(
                        input_grid=Grid(item['input']),
                        output_grid=Grid(item['output'])
                    ))

            # Process test data
            if 'test' in data:
                for item in data['test']:
                    # Handle test examples that might not have output
                    if 'output' in item:
                        self.test_examples.append(Example(
                            input_grid=Grid(item['input']),
                            output_grid=Grid(item['output'])
                        ))
                    else:
                        self.test_examples.append(Example(
                            input_grid=Grid(item['input'])
                        ))

        except Exception as e:
            print(f"Error loading task data: {e}")

    def __eq__(self, other: "ARCTask"):
        """Check if two ARCTask objects are equal by comparing their train and test data"""
        # Compare training data
        if len(self.training_examples) != len(other.training_examples):
            return False

        for i in range(len(self.training_examples)):
            if self.training_examples[i] != other.training_examples[i]:
                return False

        # Compare test data
        if len(self.test_examples) != len(other.test_examples):
            return False

        for i in range(len(self.test_examples)):
            if self.test_examples[i] != other.test_examples[i]:
                return False

        return True

    def to_prompt(self) -> str:
        """Generate the initial prompt for the task to feed into the LLM."""
        prompt = [f"# ARC Task: {self.name}\n", "## Training Examples\n"]

        for i, example in enumerate(self.training_examples):
            prompt.append(f"### Training Example {i + 1}")
            prompt.append("Input:")
            prompt.append("```")
            prompt.append(str(example.input_grid))
            prompt.append("```")
            prompt.append("Output:")
            prompt.append("```")
            prompt.append(str(example.output_grid))
            prompt.append("```\n")

        prompt.append("## Test Examples\n")

        for i, example in enumerate(self.test_examples):
            prompt.append(f"### Test Example {i + 1}")
            prompt.append("Input:")
            prompt.append("```")
            prompt.append(str(example.input_grid))
            prompt.append("```")
            prompt.append("Output:")
            prompt.append("```")
            prompt.append("To be predicted!")
            prompt.append("```\n")

        return "\n".join(prompt)

    def run_training_examples(self, code: str) -> tuple[bool, bool, list[list[int]]]:
        error = False
        passed = True
        outputs = []

        for i, example in enumerate(self.training_examples):
            test_input = example.input_grid.grid
            expected_output = example.output_grid.grid

            actual_output = execute_code_with_grid(code, test_input, self.config.temporary_path)

            if actual_output is None:
                error = True
                passed = False
                break

            if actual_output != expected_output:
                passed = False

            outputs.append(actual_output)

        return error, passed, outputs

    def run_test_examples(self, code: str) -> tuple[bool, bool, list[list[int]]]:
        error = False
        passed = True
        outputs = []

        for i, example in enumerate(self.test_examples):
            test_input = example.input_grid.grid
            expected_output = example.output_grid.grid

            actual_output = execute_code_with_grid(code, test_input, self.config.temporary_path)

            if actual_output is None:
                error = True
                passed = False
                break

            if actual_output != expected_output:
                passed = False

            outputs.append(actual_output)

        return error, passed, outputs

    def predict_test_examples(self, code: str) -> tuple[bool, list[list[int]]]:
        """Generate predictions for test examples without validation."""
        err = False
        predictions = []

        for i, example in enumerate(self.test_examples):
            test_input = example.input_grid.grid
            output = execute_code_with_grid(code, test_input, self.config.temporary_path)
            if output is None:
                err = True
                break
            else:
                predictions.append(output)

        return err, predictions
