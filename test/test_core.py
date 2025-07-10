import sys
import numpy as np

import pytest

from rstar_deepthink.config import Config
from rstar_deepthink.arc_task import ARCTask, Grid, Example
from rstar_deepthink.node import Node
from rstar_deepthink.tools import python_tool
import utils
from utils import batch, serialize_nodes, load_nodes
from constants import CODE_PREFIX, CODE_END


@pytest.fixture(autouse=True)
def _patch_sys_argv(monkeypatch):
    """Ensure Config does not parse pytest arguments."""
    monkeypatch.setattr(sys, "argv", ["pytest"])


def test_batch_basic():
    items = list(range(10))
    result = list(batch(items, 3))
    assert result == [items[0:3], items[3:6], items[6:9], items[9:10]]


def test_batch_single_when_n_nonpositive():
    items = list(range(5))
    assert list(batch(items, 0)) == [items]


def test_grid_and_example_equality():
    g1 = Grid([[1, 2], [3, 4]])
    g2 = Grid([[1, 2], [3, 4]])
    g3 = Grid([[0, 1], [2, 3]])

    assert g1 == g2
    assert g1 != g3

    ex1 = Example(g1, g2)
    ex2 = Example(Grid([[1, 2], [3, 4]]), Grid([[1, 2], [3, 4]]))
    assert ex1 == ex2


def test_arctask_from_dict_and_equality():
    data = {
        "train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[2]]}],
    }
    t1 = ARCTask.from_dict(data, "t")
    t2 = ARCTask.from_dict(data, "t")
    assert t1 == t2
    assert len(t1.training_examples) == 1
    assert len(t1.test_examples) == 1


def _make_simple_task():
    data = {
        "train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[1]], "output": [[1]]}],
    }
    return ARCTask.from_dict(data, "simple")


def test_node_validity_and_terminal(monkeypatch, tmp_path):
    cfg = Config()
    cfg.local_job_dir = tmp_path.as_posix()
    cfg.execute_in_subprocess = False
    task = _make_simple_task()

    root = Node(cfg)
    root.task = task
    root.state["code"] = CODE_PREFIX

    child = root.add_child("return I\n" + CODE_END, 0.0, None)

    assert child.valid
    assert child.passes_training
    assert child.is_terminal()
    assert child.terminal_reason == "Code end marker"
    assert child.execution_outputs == [[[1]], [[1]]]


def test_node_invalid_solution(monkeypatch):
    cfg = Config()
    cfg.execute_in_subprocess = False
    task = _make_simple_task()
    root = Node(cfg)
    root.task = task
    root.state["code"] = CODE_PREFIX

    child = root.add_child("return [[0]]\n" + CODE_END, 0.0, None)

    assert child.valid
    assert not child.passes_training


def test_serialize_and_load_nodes(tmp_path, monkeypatch):
    cfg = Config()
    cfg.local_job_dir = tmp_path.as_posix()
    cfg.execute_in_subprocess = False
    task = _make_simple_task()

    root = Node(cfg)
    root.task = task
    root.state["code"] = CODE_PREFIX
    child = root.add_child("return I\n" + CODE_END, 0.0, None)

    nodes = [root, child]
    data = serialize_nodes(nodes)
    file_path = tmp_path / "nodes.json"
    with open(file_path, "w") as f:
        import json
        json.dump(data, f)

    loaded = load_nodes(file_path)
    # find loaded root and child by tags
    loaded_dict = {n.tag: n for n in loaded}
    assert set(loaded_dict.keys()) == {"0", "0.0"}
    assert loaded_dict["0.0"].parent is loaded_dict["0"]
    assert loaded_dict["0"].children[0] is loaded_dict["0.0"]


def test_node_updates_and_metadata(monkeypatch):
    cfg = Config()
    cfg.execute_in_subprocess = False
    task = _make_simple_task()

    root = Node(cfg)
    root.task = task
    root.state["code"] = CODE_PREFIX
    root.example_name = "root"
    root.temperature = 0.5

    child = root.add_child("return I\n" + CODE_END, 0.7, "child")

    # update root and child with values
    root.update(2.0)
    child.update_recursive(1.0)

    assert root.visit_count == 2
    assert child.visit_count == 1

    assert root.q_value() == pytest.approx(1.5)
    assert child.q_value() == pytest.approx(1.0)

    expected_puct = child.q_value() + cfg.c_puct * np.sqrt(
        np.log(root.visit_count) / child.visit_count
    )
    assert child.puct() == pytest.approx(expected_puct)

    # Collect code and prompt
    collected_code = child.collect_code()
    assert collected_code.startswith(CODE_PREFIX)
    assert collected_code.endswith(CODE_END)

    root.state.update(
        {
            "system_prompt": "sys1",
            "example_prompt": "ex1",
            "task_prompt": "t1",
            "hint": "h1",
        }
    )
    child.state.update(
        {
            "system_prompt": "sys2",
            "example_prompt": "ex2",
            "task_prompt": "t2",
            "hint": "h2",
        }
    )
    collected_prompt_code = child.collect_prompt_and_code()
    assert collected_prompt_code == (
        "sys1ex1t1h1" + CODE_PREFIX + "sys2ex2t2h2return I\n" + CODE_END
    )

    meta = child.collect_metadata()
    assert meta == {
        "q_values": [root.q_value(), child.q_value()],
        "examples": ["root", "child"],
        "temperatures": [0.5, 0.7],
    }

    assert child.is_valid_final_answer_node()


def test_run_examples_and_correctness(monkeypatch):
    cfg = Config()
    cfg.execute_in_subprocess = False
    task = _make_simple_task()

    code = CODE_PREFIX + "return I\n" + CODE_END
    err, passed, outputs = python_tool.run_examples(
        task, python_tool.remove_markers(code), test_test=True, config=cfg
    )
    assert not err
    assert passed
    assert outputs == [[[1]], [[1]]]

    node = Node(cfg)
    node.task = task
    node.valid = True
    node.execution_outputs = outputs

    err, passed_test, test_outs = python_tool.test_correct(node)
    assert not err
    assert passed_test
    assert test_outs == [[[1]]]


def test_make_serializable_with_object():
    class Dummy:
        def __init__(self) -> None:
            self.x = 1

    assert serialize_nodes([Node(Config())])  # smoke for coverage
    assert utils.make_serializable(Dummy()) == {"x": 1}
