# rStar-ARC

🚧 **Status: active work in progress** – expect rapid iteration, rough edges, and breaking changes while the project grows toward a fully automated ARC solver.

## Overview
`rStar-ARC` experiments with bringing **Self-play muTuAl Reasoning (rStar)** to the **Abstraction and Reasoning Corpus (ARC)**. The system coordinates small language models through search agents that iteratively draft and grade Python code reasoning steps until a task’s test cases are solved. 

The code base focuses on:
- search-driven program synthesis with interchangeable agents (beam search, MCTS variants, bootstrap hints)
- tooling for dataset curation, supervised fine-tuning (SFT), and qualitative inspection of the agent search process

The repository currently targets local workstation and SLURM-based research workflows. Many components are prototypes.

## Repository layout
```
├── configs/                # YAML configurations for search, ablations, and training
├── data_sample/            # Small ARC task subsets and prompt examples for development
├── rstar_deepthink/        # Core library: solver, agents, ARC task helpers, prompt builders, LLM wrappers
├── train/                  # Scripts for generating datasets and fine-tuning policy/reward adapters
├── test/                   # Smoke tests, inference utilities, and LoRA merge helpers
├── tree_visualizer.py      # Converts saved search traces into interactive HTML trees
├── run.py                  # Interactive SLURM job generator for batch experiments
└── weekly_progress_report.md
```

Key modules worth exploring first:

- [`rstar_deepthink/solver.py`](rstar_deepthink/solver.py) – orchestrates multi-rollout search, batching, and scoring between policy/reward models.
- [`rstar_deepthink/agents/`](rstar_deepthink/agents) – implementations of the shared `Agent` interface (`BS`, `MCTS`, `Custom`, `Bootstrap`) plus utilities for temperature control.
- [`rstar_deepthink/config.py`](rstar_deepthink/config.py) – unified dataclass that loads CLI/YAML settings, tracks resource usage, and exposes training knobs.

## Getting started
1. **Install dependencies** (a CUDA-capable GPU is recommended for `vllm`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **(Optional) Download ARC data.** The repository ships with `data_sample/` for quick experiments. Point `--config-file` entries at the full ARC dataset if available.
3. **Configure credentials.** Paths in [`constants.py`](constants.py) default to ETH Zürich DISCO group infrastructure; adjust them to match your environment.

## Running a search experiment
1. Choose a configuration from `configs/` (e.g., `custom.yaml`).
2. Launch the main loop:
   ```bash
   python main.py --config-file configs/custom.yaml
   ```
   `main.py` loads ARC tasks, instantiates the requested agent, iterates over batches, and records summaries/visualizations depending on the config flags.
3. Inspect the logs and saved artifacts under the configured output directory (defaults to scratch paths defined in `Config`).

### Visualizing rollouts
Set `save_for_visualization: true` in your YAML config, then render the generated `nodes.json` with the built-in visualizer:
```bash
python tree_visualizer.py --input <path/to/nodes.json> --output tree.html
```
The HTML output highlights branching structure, reasoning steps, and final program validity.

### Generating supervised fine-tuning (SFT) data
If `save_sft_data` is enabled, `main.py` writes intermediate trajectories that can be post-processed with scripts in `train/`. Use the helpers in that directory to augment data, build training/validation splits, and launch adapter fine-tuning runs (e.g., `train/train_policy.py`).

### Submitting jobs via SLURM (optional)
`run.py` provides an interactive CLI that bundles script selection, SLURM resource choices, and configuration overrides into a temporary submission script. This is useful when scheduling multiple experiments on shared clusters.

## Testing and linting
A lightweight `pytest` suite exercises utility functions and integration points. Run it from the repository root:
```bash
pytest -q
```
(Some tests may require ARC sample data or optional dependencies; skip or adjust as needed.)

## Roadmap & open questions
The project is still stabilizing. Immediate priorities include:

- making testing scripts more rigorous and collecting additional data to enable statistically sound analyses suitable for eventual publication
- expanding automated tests to cover new agent behaviors and data pipelines
- validating policy/reward model choices on the full ARC evaluation set
- tightening resource configuration for both local and SLURM runs
- polishing the visualization tooling and documentation

For a narrative view of progress and hurdles, see [`weekly_progress_report.md`](weekly_progress_report.md).

## License
This project is released under the MIT License. See [`LICENSE`](LICENSE) if/when it is added to the repository. Until then, assume MIT terms as stated here.
