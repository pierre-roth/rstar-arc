# Training

- `datasets/`: trajectory collection, augmentation, conversion, and dataset preparation.
- `policy/`: Trainer-based and distributed policy training.
- `reward/`: reward training, including full training with separate validation splits.
- `utils.py`: shared collators and weighting helpers.

```bash
python -m rstar_deepthink.training.policy.train --config-file path/to/training.yaml
accelerate launch --module rstar_deepthink.training.policy.distributed --config-file path/to/training.yaml
```

Imported grid helpers and task generators are kept in `rstar_deepthink/vendor/`.
