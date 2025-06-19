# Agents Guide for rStar meets ARC

This document provides a high-level overview of the agent-driven architecture in the **rStar meets ARC** project, including how to configure and run the different search agents used to solve ARC tasks.

## Project Goal
The aim of this project is to apply **Self-play muTuAl Reasoning (rStar)** to the **Abstraction and Reasoning Corpus (ARC)**, using small language models (SLMs) orchestrated by search agents to iteratively build reasoning-and-code solutions for ARC tasks.

## Repository Structure
```text
.
├── configs/                 # YAML experiment configurations (bootstrap, training, evaluation, custom)
├── data_sample/             # Example ARC tasks and bootstrap data for development
├── rstar_deepthink/         # Core library
│   ├── agents/              # Agent implementations (Beam Search, MCTS, Custom, Bootstrap)
│   ├── arc_task/            # ARC task loader and data structures
│   ├── config.py            # Unified configuration class (loads YAML)
│   ├── llms/                # PolicyModel and RewardModel wrappers
│   ├── node.py              # Tree node abstraction for reasoning steps
│   ├── prompt/              # Prompt-building utilities
│   ├── solver.py            # Orchestrates multi-step search using agents and LLMs
│   └── tools/               # Helper tools (code execution, similarity tests)
├── train/                   # Scripts to generate and train policy/reward models
├── test/                    # Unit tests, helper scripts and scripts for merging fine-tuned adapters (LoRA)
├── main.py                  # Entry point for running ARC-solving experiments
├── run.py                   # Helper for interactive or SLURM job submission
├── tree_visualizer.py       # Visualize agent search-tree outputs
├── utils.py                 # I/O, batching, logging, and serialization helpers
└── requirements.txt         # Python dependencies
```

## Agent Architecture
All search agents inherit from the common `Agent` interface defined in **rstar_deepthink/agents/base_agent.py**:
```python
class Agent:
    def __init__(self, config: Config, task: ARCTask): ...
    def update(self, rollout_idx: int, current_temperature: float) -> None: ...
    def should_generate_next(self) -> bool: ...
    def has_expanded(self) -> bool: ...
    def create_prompts(self, is_value_only: bool = False) -> list[str]: ...
    def generate_next_step(self, outputs) -> None: ...
    def select_next_step(
        self,
        scores: Optional[list[float]] = None,
        from_root: bool = False,
    ) -> None: ...
    def get_nodes(self) -> list[Node]: ...
```
【F:rstar_deepthink/agents/base_agent.py†L16-L24】【F:rstar_deepthink/agents/base_agent.py†L51-L68】【F:rstar_deepthink/agents/base_agent.py†L79-L88】【F:rstar_deepthink/agents/base_agent.py†L90-L97】【F:rstar_deepthink/agents/base_agent.py†L98-L110】【F:rstar_deepthink/agents/base_agent.py†L112-L118】

## Available Agents
- **Beam Search (BS)**: maintains a fixed-width beam of top hypotheses at each step.
  Implemented in `rstar_deepthink/agents/beam_search.py`.
  【F:rstar_deepthink/agents/beam_search.py†L9-L40】
- **Monte Carlo Tree Search (MCTS)**: uses PUCT to balance exploration and exploitation.
  Implemented in `rstar_deepthink/agents/mcts.py`.
  【F:rstar_deepthink/agents/mcts.py†L10-L18】【F:rstar_deepthink/agents/mcts.py†L38-L47】【F:rstar_deepthink/agents/mcts.py†L59-L67】
- **Custom Agent**: MCTS variant with grid-similarity rewards for intermediate steps.
  Implemented in `rstar_deepthink/agents/custom.py`.
  【F:rstar_deepthink/agents/custom.py†L12-L29】【F:rstar_deepthink/agents/custom.py†L55-L89】
- **Bootstrap Agent**: hint-guided MCTS bootstrapping with task-description prompts.
  Implemented in `rstar_deepthink/agents/bootstrap.py`.
  【F:rstar_deepthink/agents/bootstrap.py†L13-L25】【F:rstar_deepthink/agents/bootstrap.py†L39-L57】【F:rstar_deepthink/agents/bootstrap.py†L59-L78】

## Selecting and Running Agents
The active agent is chosen by the `search_mode` parameter in your YAML config:
```yaml
# configs/custom.yaml
search-mode: "custom"
num-simulations: 16
branching-factor: 8
max-depth: 16
... (other hyperparameters)
```
【F:configs/custom.yaml†L1-L8】【F:configs/custom.yaml†L12-L15】

In `main.py`, the mapping looks like:
```python
agent_cls = {"bs": BS,
             "mcts": MCTS,
             "custom": Custom,
             "bootstrap": Bootstrap}
agent = agent_cls.get(config.search_mode, BS)
```
【F:main.py†L31-L33】

Run a configured experiment:
```bash
python main.py --config-file configs/custom.yaml
```
Or submit via SLURM:
```bash
python run.py --config-file configs/custom.yaml
```
【F:run.py†L38-L50】


## Training and Evaluation
Training scripts live in the `train/` folder and use the same `Config` system as the solver. For example, to fine‑tune the policy model with LoRA adapters:
```bash
python train/train_policy.py --config-file configs/train_policy_full_1.yaml
```
Validation datasets and LoRA merging utilities are provided under `test/`.


## Visualization and Debugging
- Enable `save_for_visualization: true` in your config to dump `nodes.json` files.
- Visualize saved trees using:
  ```bash
  python tree_visualizer.py --input <nodes.json> --output <tree.html>
  ```
  【F:tree_visualizer.py†L261-L270】

## Running the Unit Tests
A small `pytest` suite ensures that utilities behave as expected. Run the tests from the repository root:
```bash
pytest -q
```
All tests should pass (see `test/test_core.py`).
If you come across functionality that is not covered, please add a test case to `test/test_core.py` or create a new test file in the `test/` directory.


## Extending the Agent Framework
To add a new search agent:
1. Create a subclass of `Agent` in `rstar_deepthink/agents/`.
2. Override the core interface methods (`select_next_step`, `should_generate_next`, etc.).
3. Register your new agent in the `main.py` mapping.

---
For general project information see `README.md`. Historical notes are kept in `weekly_progress_report.md`.
