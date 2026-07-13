"""Training-side modules for deepcell-types.

This subpackage holds the training pipeline — losses, the zarr-backed
dataset, annotation extraction, training utilities, etc. None of its
contents are imported by the inference path (``deepcell_types.predict``);
importing it requires the ``[train]`` extra:

    pip install deepcell-types[train]

Inference users do not need any of this. The split is enforced by a CI
guard (``tests/test_inference_deps.py``) that fails if the inference path
ever transitively imports a training-only dependency (zarr, scikit-learn,
pandas, ...).
"""

# Convenience re-exports of the most commonly used training symbols, so
# `from deepcell_types.training import TissueNetConfig` works. These are
# resolved lazily (PEP 562) to avoid eagerly importing the dataloader/dataset
# modules — which cross-import each other — at package import time.
__all__ = [
    "TissueNetConfig",
    "FullImageDataset",
    "FocalLoss",
    "HierarchicalLoss",
    "create_dataloader",
]

_LAZY = {
    "TissueNetConfig": "deepcell_types.training.config",
    "FullImageDataset": "deepcell_types.training.dataset",
    "create_dataloader": "deepcell_types.training.dataloader",
    "FocalLoss": "deepcell_types.training.losses",
    "HierarchicalLoss": "deepcell_types.training.losses",
}


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)
