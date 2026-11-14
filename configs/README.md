# Experiments

| Directory | Comparison |
| --- | --- |
| `search/` | Beam search, custom search, and description-guided bootstrapping |
| `baselines/` | Qwen coder without fine-tuning |
| `policy/` | Fine-tuned policy without learned reward scoring |
| `reward/` | Fine-tuned policy with a reward model of the same size |
| `ablations/` | Policy/reward size combinations, one-shot prompts, and branching factor |

Names identify the model size and dataset split. `branches-32` changes the
branching factor from 8 to 32. The original `round-number` values and
experiment settings are unchanged.

```bash
python -m rstar_deepthink --config-file configs/search/beam.yaml
```
