import json
import subprocess
import sys


def test_cpu_demo(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rstar_deepthink.demo", "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    nodes = json.loads((tmp_path / "6150a2bd_nodes.json").read_text())
    children = [node for node in nodes.values() if node["parent"] == "0"]
    assert len(children) == 3
    assert sum(bool(node["passes_training"]) for node in children) == 1
    assert "also passes the held-out test example" in result.stdout
    assert "Plotly.restyle" in (tmp_path / "tree.html").read_text()
