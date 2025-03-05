import json
from arc_rstar.config import Config


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


class ARCTask:
    def __init__(self, path, config: Config):
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
        result = []

        result.append(f"ARCTask {self.name}")
        result.append(f"Number of training examples: {len(self.train_data)}")
        result.append(f"Number of test examples: {len(self.test_data)}")

        result.append("\nTraining examples: ")

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


