# rStar-ARC

**Tree-search program synthesis for ARC**

Bachelor's thesis · Pierre Roth · 2025

Language models build Python solutions step by step, guided by execution feedback
and a learned reward model. Search traces train the policy and reward models.
The implementation includes beam search, MCTS, bootstrapping, and task augmentation.

![ARC task 00d62c1b: input grids above, output grids below](docs/images/00d62c1b.png)

This research project is complete.

## Run

Python 3.12. Search inference requires Linux and an NVIDIA GPU for vLLM.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/research.txt
python -m rstar_deepthink --config-file configs/bs.yaml
```

For CPU tests and visualization, install `requirements/core.txt` instead.

```bash
python -m rstar_deepthink.visualization --input path/to/task_nodes.json --output tree.html
python -m pytest -q
```

The viewer supports `--theme light`, `--step-only`, and `--open`.
Hover for code; hold Shift for search metadata.

## Layout

| Directory | Contents |
| --- | --- |
| [`rstar_deepthink/`](rstar_deepthink/) | Solver, agents, models, and training pipelines |
| [`configs/`](configs/) | Search and ablation experiments |
| [`data/`](data/) | Sample tasks, prompt examples, and bootstrap descriptions |
| [`scripts/`](scripts/) | Inference, adapter merging, and SLURM submission |
| [`tests/`](tests/) | CPU tests |
| [`docs/`](docs/) | [Research notes](docs/research-notes.md), figures, and [sources](docs/sources.md) |

Run training pipelines as `python -m rstar_deepthink.training.<module>`.
The distributed policy trainer uses `accelerate launch --module`.
`python -m scripts.submit` submits jobs to the ETH TIK cluster.

## Data and storage

Trained checkpoints, generated datasets, and full experiment outputs are not
included. Generated files default to `artifacts/`.

| Variable | Default |
| --- | --- |
| `RSTAR_STORAGE_DIR` | `<checkout>/artifacts` |
| `RSTAR_SCRATCH_DIR` | `<storage>/scratch` |
| `RSTAR_DATA_DIR` | `<storage>/task_data` |

Fine-tuned configurations expect models under `<storage>/models/policy/` and
`<storage>/models/reward/`. YAML paths are relative to the checkout.

## License

Original code: [MIT](LICENSE). [Third-party sources and terms](docs/sources.md).
