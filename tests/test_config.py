import sys
from pathlib import Path

import pytest

from constants import NET_SCRATCH_TASK_DATA_DIR, PROJECT_PATH
from rstar_deepthink.config import Config


@pytest.fixture(autouse=True)
def clean_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])


@pytest.mark.parametrize(
    "path",
    sorted((Path(PROJECT_PATH) / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_supplied_configurations_load(path):
    config = Config(config_file=str(path))
    assert config.search_mode in {"bs", "mcts", "custom", "bootstrap"}


def test_yaml_options_and_relative_paths(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text(
        "search-mode: mcts\nnum-rollouts: 3\ndata-folder: data_sample/easy\nsort-by-length: true\n"
    )
    config = Config(config_file=str(path))
    assert config.num_rollouts == 3
    assert config.sort_by_length
    assert Path(config.data_folder) == Path(PROJECT_PATH) / "data_sample/easy"


def test_dataset_root_expansion(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text('data-folder: "${RSTAR_DATA_DIR}/evaluation"\n')
    assert Config(config_file=str(path)).data_folder == str(
        Path(NET_SCRATCH_TASK_DATA_DIR) / "evaluation"
    )


def test_unknown_options_are_reported(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text("num-simulations: 12\n")
    with pytest.raises(ValueError, match="num_simulations"):
        Config(config_file=str(path))


def test_independent_configs_and_reproducible_examples():
    first = Config(seed=17)
    second = Config(seed=17)
    first.model_initialization_times["policy"] = 1
    assert second.model_initialization_times["policy"] is None
    assert first.example_names == second.example_names
