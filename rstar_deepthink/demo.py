"""Run candidate validation and trace visualization without a language model."""

import argparse
import json
from pathlib import Path

from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.constants import (
    CODE_END,
    CODE_PREFIX,
    NET_SCRATCH_PATH,
    PROJECT_PATH,
    STEP_END,
)
from rstar_deepthink.node import Node
from rstar_deepthink.tools.python_tool import test_correct
from rstar_deepthink.utils import serialize_nodes
from rstar_deepthink.visualization import visualize_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(NET_SCRATCH_PATH) / "demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    task_path = Path(PROJECT_PATH) / "data/very_easy/6150a2bd.json"
    task = ARCTask.from_dict(json.loads(task_path.read_text()), "6150a2bd")
    config = Config(execute_in_subprocess=True, max_depth=2)
    root = Node(config)
    root.task = task
    root.valid = True
    root.state["code"] = CODE_PREFIX

    candidates = [
        ("Identity", "return I"),
        ("Mirror", "return [row[::-1] for row in I]"),
        ("Rotate 180 degrees", "return [row[::-1] for row in I[::-1]]"),
    ]
    print("Handwritten candidates; no language model is used.")
    for name, code in candidates:
        child = root.add_child(code + "\n" + STEP_END + "\n" + CODE_END, 0.0, name)
        child.update_recursive(1.0 if child.passes_training else 0.0)
        print(
            f"{name}: {'passes' if child.passes_training else 'fails'} the training examples"
        )

    solved = [
        node for node in root.children if node.passes_training and test_correct(node)[1]
    ]
    assert len(solved) == 1
    print(f"{solved[0].example_name} also passes the held-out test example.")
    trace = args.output / "6150a2bd_nodes.json"
    trace.write_text(json.dumps(serialize_nodes([root, *root.children]), indent=2))
    output = visualize_tree(
        trace, output_html=args.output / "tree.html", theme="light", step_only=True
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
