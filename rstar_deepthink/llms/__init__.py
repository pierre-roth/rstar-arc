"""
Lazy import of PolicyModel and RewardModel to avoid heavy dependencies at module import.
Heavy frameworks (vllm, torch, transformers) only load when these classes are accessed.
"""
import importlib

# noinspection PyUnresolvedReferences
__all__ = ['PolicyModel', 'RewardModel']


def __getattr__(name: str):
    if name == 'PolicyModel':
        mod = importlib.import_module(f"{__name__}.policy")
        return mod.PolicyModel
    if name == 'RewardModel':
        mod = importlib.import_module(f"{__name__}.reward")
        return mod.RewardModel
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
