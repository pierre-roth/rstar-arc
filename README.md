# rStar-ARC

Bachelor's thesis · Pierre Roth, 2025. Development is complete.

Applying rStar-Math to the Abstraction and Reasoning Corpus (ARC): language models
build Python programs through tree search, with execution feedback and a learned
reward model guiding the search. Successful traces and preference pairs are used
to train the policy and reward models.

The implementation includes beam search, MCTS, description-guided bootstrapping,
task augmentation, supervised fine-tuning, and a search-tree viewer.

## Setup

Python 3.12. Search inference requires Linux and an NVIDIA GPU for vLLM.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU tests and visualization, install `requirements-core.txt` instead.

## Run

```bash
python main.py --config-file configs/bs.yaml
python tree_visualizer.py --input path/to/task_nodes.json --output tree.html
```

The viewer supports `--theme light`, `--step-only`, and `--open`. Hover for code;
hold Shift for scores and search metadata. Green nodes pass the task's training
examples, which does not guarantee a correct test prediction.

[`configs/`](configs/) contains the search and ablation experiments. Options are
defined in [`Config`](rstar_deepthink/config.py); paths in YAML are relative to the
checkout. `python run.py` launches jobs on the ETH TIK SLURM cluster.

## Data and storage

Sample tasks and bootstrap descriptions are included. Trained models, generated
training datasets, and full experiment outputs are not included.

Generated files default to `artifacts/`. Set these variables to use other storage:

| Variable | Default |
| --- | --- |
| `RSTAR_STORAGE_DIR` | `<checkout>/artifacts` |
| `RSTAR_SCRATCH_DIR` | `<storage>/scratch` |
| `RSTAR_DATA_DIR` | `<storage>/task_data` |

Fine-tuned configurations expect checkpoints under `<storage>/models/policy/`
and `<storage>/models/reward/`. Search traces go to `<scratch>/job_<id>/`;
summaries go to `<storage>/outputs/detailed_logs/job_<id>/`.

## Code

- [`rstar_deepthink/`](rstar_deepthink/): solver, agents, tasks, prompts, and model wrappers.
- [`train/`](train/): dataset preparation and policy/reward training. Use
  `train_policy.py` or `train_reward.py` for Trainer-based training,
  `train_policy_distributed.py` with `accelerate launch`, and
  `train_reward_full.py` for full reward training with separate validation splits.
  The latter takes reward filenames through `training-dataset-name` and
  `validation-dataset-name` in the config.
- [`scripts/`](scripts/): direct inference and LoRA adapter merging.
- [`Research notes`](docs/research-notes.md): the original development journal.

## Tests

```bash
pip install -r requirements-core.txt
python -m pytest -q
```

Tests run on the CPU without downloading models.

## License

Original code: [MIT](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for sources,
acknowledgments, and third-party terms.
