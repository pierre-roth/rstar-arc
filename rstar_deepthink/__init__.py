"""
Lazy import of Solver and Config to avoid heavy dependencies at package import time.
Heavy modules (e.g., vllm) are only loaded when Solver or Config are accessed.
"""
import importlib

# noinspection PyUnresolvedReferences
__all__ = ["Solver", "Config"]


def __getattr__(name: str):
    if name == "Solver":
        module = importlib.import_module(f"{__name__}.solver")
        return module.Solver
    if name == "Config":
        module = importlib.import_module(f"{__name__}.config")
        return module.Config
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
