"""Regression: training submodules must import in any order.

``deepcell_types.training.dataset`` re-exports ``dataloader``'s symbols for
back-compat, so a module-level ``from .dataset import ...`` in ``dataloader``
made ``import deepcell_types.training.dataloader`` (before ``dataset``) raise a
circular ImportError. Each import is checked in its own subprocess because
``sys.modules`` caches modules within a single interpreter.
"""

import subprocess
import sys


def _import_ok(statement):
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_dataloader_importable_first():
    _import_ok(
        "import deepcell_types.training.dataloader as d; "
        "assert d.create_dataloader and d.DataLoaderConfig"
    )


def test_dataset_importable_first():
    _import_ok(
        "import deepcell_types.training.dataset as ds; "
        "assert ds.create_dataloader and ds.FullImageDataset"
    )


def test_dataloader_reexported_from_dataset():
    _import_ok(
        "from deepcell_types.training.dataset import create_dataloader, DataLoaderConfig"
    )
