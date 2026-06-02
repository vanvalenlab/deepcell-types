"""Pytest configuration for the deepcell-types test suite.

Several test modules exercise the training pipeline and need the ``[train]``
install extra (``zarr``, ``pandas``, ``torchmetrics``, ``sklearn``, ...).
On an inference-only checkout (``pip install -e .``) those tests should be
*skipped*, not raise collection errors.

We achieve this with ``collect_ignore_glob`` by attempting to import the
required modules and skipping the corresponding tests when imports fail.
"""

from __future__ import annotations

import importlib


# Map of test files to the [train]-extra modules they need at import time.
# When any required module is missing, the test file is excluded from
# collection. Add entries here when a new test file gains a bare top-level
# import of a [train]-only dependency.
_REQUIRES = {
    "test_baseline_feature_splits.py": ("zarr",),
    "test_channel_aliases.py": ("zarr",),
    "test_ct_abstention_cli.py": ("pandas",),
    "test_dataset_cache.py": ("pandas",),
    "test_dataset_celltypes.py": ("pandas",),
    "test_dataset_splits.py": ("zarr",),
    "test_embeddings_load.py": ("pandas",),
    "test_hierarchical_eval.py": ("pandas",),
    "test_hierarchy_one_way.py": ("pandas",),
    "test_losses.py": ("pandas",),
    "test_min_channels.py": ("zarr",),
    "test_predlogger_dataframe.py": ("pandas",),
    "test_samplers.py": ("zarr",),
    "test_stratified_splits.py": ("zarr",),
    "test_v2.py": ("zarr",),
}


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
    except ImportError:
        return False
    return True


collect_ignore = [
    filename
    for filename, required in _REQUIRES.items()
    if not all(_have(m) for m in required)
]
