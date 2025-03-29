import json
import logging
import os
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
        self.json_data = None

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
            logger.error(f"Error loading task data: {e}")

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
