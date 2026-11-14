import json
import subprocess
import sys
from pathlib import Path

import pytest

from constants import CODE_END, CODE_PREFIX, PROJECT_PATH
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from tree_visualizer import build_graph_from_nodes, visualize_tree
from utils import load_nodes, serialize_nodes


@pytest.fixture
def trace(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    config = Config(execute_in_subprocess=False)
    root = Node(config)
    root.task = ARCTask.from_dict(
        {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[1]], "output": [[1]]}],
        },
        "identity",
    )
    root.valid = True
    root.state["code"] = CODE_PREFIX
    child = root.add_child("return I if 1 < 2 else []\n" + CODE_END, 0.7, "example")
    child.update_recursive(1.0)
    path = tmp_path / "identity_nodes.json"
    path.write_text(json.dumps(serialize_nodes([root, child])))
    return path


def test_loaded_tree_keeps_node_behavior(trace):
    root, child = load_nodes(trace)
    assert child.is_valid_final_answer_node()
    assert child.puct() > 0
    assert root.children == [child]
    graph = build_graph_from_nodes([root, child])
    assert graph.ecount() == 1
    assert "&lt;" in graph.vs[1]["normal_hovertext"]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_html_rendering(trace, tmp_path, theme):
    output = tmp_path / theme / "tree.html"
    assert (
        visualize_tree(trace, output_html=output, theme=theme, step_only=True) == output
    )
    html = output.read_text()
    assert "plotly-div" in html
    assert "Plotly.restyle" in html


def test_visualizer_directory_cli(trace, tmp_path):
    output = tmp_path / "rendered"
    subprocess.run(
        [
            sys.executable,
            str(Path(PROJECT_PATH) / "tree_visualizer.py"),
            "--input",
            str(trace.parent),
            "--output",
            str(output),
            "--theme",
            "light",
        ],
        check=True,
    )
    assert (output / "identity_nodes.html").is_file()
