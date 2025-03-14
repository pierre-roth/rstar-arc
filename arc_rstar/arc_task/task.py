import json
from typing import Any, Type
from config import Config
import numpy as np

from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid


class Grid:
    """Class representing a 2D grid of integers"""

    def __init__(self, grid: list[list[int]]):
        self.grid: list[list[int]] = grid
        self.rows: int = len(self.grid)
        self.columns: int = len(self.grid[0])

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


class ARCTask:
    def __init__(self, path, config: Config):
        self.path = path
        self.config = config
        self.name = path.split("/")[-1].split(".")[0]
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
                    self.test_examples.append(Example(
                        input_grid=Grid(item['input']),
                        output_grid=Grid(item['output'])
                    ))
        except Exception as e:
            print(f"Error loading task data: {e}")

    def __eq__(self, other: 'ARCTask'):
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

    def __str__(self):
        """String representation of the ARCTask"""
        result = [f"ARCTask {self.name}", f"Number of training examples: {len(self.training_examples)}",
                  f"Number of test examples: {len(self.test_examples)}", "\nTraining examples: "]

        # Add details for each training example
        for i, example in enumerate(self.training_examples):
            result.append(f"\nTraining Example {i + 1}:")
            result.append(str(example))

        result.append("\nTest examples: ")

        # Add details for each test example
        for i, example in enumerate(self.test_examples):
            result.append(f"\nTest Example {i + 1}:")
            result.append(str(example))

        return "\n".join(result)

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

    def run_training_examples(self, code: str) -> (bool, list):
        passed = True
        outputs = []
        
        if self.config.verbose:
            print(f"\nValidating code on {len(self.training_examples)} training examples")

        for i, example in enumerate(self.training_examples):
            test_input = example.input_grid.grid
            expected_output = example.output_grid.grid
            
            if self.config.verbose:
                print(f"\nRunning training example {i+1}:")
                print(f"Input shape: {len(test_input)}x{len(test_input[0]) if test_input else 0}")
                print(f"Expected output shape: {len(expected_output)}x{len(expected_output[0]) if expected_output else 0}")

            actual_output = execute_code_with_grid(code, test_input, self.config.verbose)

            if actual_output == expected_output:
                if self.config.verbose:
                    print(f"✓ Example {i+1} passed")
            else:
                passed = False
                outputs.append(actual_output)
                if self.config.verbose:
                    print(f"✗ Example {i+1} failed")
                    if actual_output:
                        print(f"Actual output shape: {len(actual_output)}x{len(actual_output[0]) if actual_output and actual_output[0] else 0}")
                    else:
                        print("No output generated (None returned)")

        if self.config.verbose:
            if passed:
                print(f"All {len(self.training_examples)} training examples passed!")
            else:
                print(f"Failed {len(outputs)}/{len(self.training_examples)} examples")
                
        return passed, outputs

    def run_test_examples(self, code: str) -> (bool, list):
        passed = True
        outputs = []
        
        if self.config.verbose:
            print(f"\nEvaluating code on {len(self.test_examples)} test examples")

        for i, example in enumerate(self.test_examples):
            test_input = example.input_grid.grid
            expected_output = example.output_grid.grid
            
            if self.config.verbose:
                print(f"\nRunning test example {i+1}:")
                print(f"Input shape: {len(test_input)}x{len(test_input[0]) if test_input else 0}")
                print(f"Expected output shape: {len(expected_output)}x{len(expected_output[0]) if expected_output else 0}")

            actual_output = execute_code_with_grid(code, test_input, self.config.verbose)

            if actual_output == expected_output:
                if self.config.verbose:
                    print(f"✓ Test example {i+1} passed")
            else:
                passed = False
                outputs.append(actual_output)
                if self.config.verbose:
                    print(f"✗ Test example {i+1} failed")
                    if actual_output:
                        print(f"Actual output shape: {len(actual_output)}x{len(actual_output[0]) if actual_output and actual_output[0] else 0}")
                    else:
                        print("No output generated (None returned)")

        if self.config.verbose:
            if passed:
                print(f"All {len(self.test_examples)} test examples passed!")
            else:
                print(f"Failed {len(outputs)}/{len(self.test_examples)} test examples")
                
        return passed, outputs

