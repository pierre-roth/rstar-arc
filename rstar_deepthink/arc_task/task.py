import json
import logging
import os
import sys
from typing import Optional

from rstar_deepthink.config import Config

logger = logging.getLogger(__name__)


class Grid:
    """Class representing a 2D grid of integers"""

    def __init__(self, grid: any):
        if self._is_valid_grid(grid):
            self.grid: list[list[int]] = grid
            self.rows: int = len(grid)
            # Safe to use grid[0] because _is_valid_grid ensures at least one row
            self.columns: int = len(grid[0])
        else:
            self.grid = [[]]
            self.rows = 0
            self.columns = 0

    @staticmethod
    def _is_valid_grid(grid: any) -> bool:
        # Must be a list and non-empty.
        if not isinstance(grid, list) or not grid:
            return False

        # Check that each row is a list and that they all have the same length.
        # Also verify that every cell in each row is an integer.
        expected_length = None
        for row in grid:
            if not isinstance(row, list):
                return False
            # Set the expected row length from the first row.
            if expected_length is None:
                expected_length = len(row)
            # Each row must be the same length.
            elif len(row) != expected_length:
                return False
            # Every cell must be an integer.
            for cell in row:
                if not isinstance(cell, int):
                    return False
        return True

    def __eq__(self, other: "Grid"):
        return self.grid == other.grid

    def __str__(self):
        return "\n".join("|".join(str(cell) for cell in row) for row in self.grid)


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
    def __init__(self, config: Config, path: str):
        self.config = config

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.json_data: Optional[dict] = None

        self.training_examples = []
        self.test_examples = []

        self._load_data()

    def _load_data(self):
        """Load task data from the provided path"""
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
                self.json_data = data

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
            logger.critical(f"Error loading task data: {e}")
            sys.exit(1)

    @classmethod
    def from_dict(cls, task_data: dict, task_name: str = "task_from_dict") -> "ARCTask":
        """
        Creates an ARCTask instance directly from a dictionary.

        Bypasses the need for a file path and Config object during initialization.

        Args:
            cls: The class itself (ARCTask).
            task_data: A dictionary representing the ARC task JSON structure.
                       Expected format: {"train": [...], "test": [...]}, where
                       each item is a dict {"input": grid, "output": grid}.
            task_name: An optional name to assign to this task instance.

        Returns:
            An initialized ARCTask instance.

        Raises:
            ValueError: If task_data is not a dictionary or if the data
                        within the dictionary is malformed (e.g., invalid grids).
        """
        if not isinstance(task_data, dict):
            raise ValueError("Input 'task_data' must be a dictionary.")

        # Create an instance without calling __init__ directly
        instance = cls.__new__(cls)

        # Manually set attributes
        instance.config = None  # No config object provided
        instance.path = None  # No file path provided
        instance.name = task_name
        instance.json_data = task_data  # Store the raw data if needed
        instance.training_examples = []
        instance.test_examples = []

        try:
            # Process training data using Grid and Example classes
            for item in task_data.get('train', []):
                input_grid = Grid(item['input'])
                output_grid = Grid(item['output'])
                # Optional: Add checks here if Grid/Example don't validate sufficiently
                # if not Grid._is_valid_grid(item['input']) or not Grid._is_valid_grid(item['output']):
                #     raise ValueError("Invalid grid found in training data")
                instance.training_examples.append(Example(input_grid, output_grid))

            # Process test data
            for item in task_data.get('test', []):
                input_grid = Grid(item['input'])
                # Optional: Add check
                # if not Grid._is_valid_grid(item['input']):
                #    raise ValueError("Invalid input grid found in test data")
                output_grid = None
                if 'output' in item:
                    output_grid = Grid(item['output'])
                    # Optional: Add check
                    # if not Grid._is_valid_grid(item['output']):
                    #     raise ValueError("Invalid output grid found in test data")
                instance.test_examples.append(Example(input_grid, output_grid))

        except KeyError as e:
            # Catch missing 'input'/'output' keys if Grid doesn't handle None gracefully
            logger.error(f"Missing key {e} in task data for task '{task_name}'.")
            raise ValueError(f"Malformed task data dictionary: missing key {e}") from e
        except Exception as e:
            # Catch other errors (e.g., from Grid validation if it raises)
            logger.error(f"Error processing task data dictionary for task '{task_name}': {e}")
            raise ValueError(f"Failed to process task data dictionary: {e}") from e

        return instance  # Return the fully populated instance

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
