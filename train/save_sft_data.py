import json
import os

from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from rstar_deepthink.tools import test_correct


def save_sft(config: Config, nodes: list[Node]):
    """
        Save successful task solutions to a JSONL file for supervised fine-tuning.

        Args:
            config: The configuration object
            nodes: List of nodes to check for valid solutions
        """
    # Create the directory path if it doesn't exist
    output_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "raw.jsonl")

    task_name = nodes[0].task.name

    solutions = []

    for node in nodes:
        if node.is_valid_final_answer_node():
            # Test if it passes test examples
            error, passed_test, _ = test_correct(node)
            if not error and passed_test:
                # Store the task name and code
                solution_code = node.collect_code()
                solutions.append(solution_code)

    with open(file_path, 'a', encoding="utf-8") as f:
        for solution_code in solutions:
            entry = json.dumps({"task_name": task_name, "solution_code": solution_code})
            f.write(entry + '\n')
