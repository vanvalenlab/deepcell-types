"""Test min_channels filtering in FullImageDataset."""

import os
from pathlib import Path

import pytest


def _archive_path():
    candidates = []
    if os.environ.get("DATA_DIR"):
        candidates.append(
            Path(os.environ["DATA_DIR"]) / "expanded-tissuenet.zarr"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip("No zarr archive")


def test_min_channels_includes_repaired_fan_uf_panel():
    from deepcell_types.training.dataset import FullImageDataset
    from deepcell_types.training.config import TissueNetConfig

    config = TissueNetConfig(_archive_path())

    # fan_uf was formerly a two-channel preprocessed panel, but the repaired
    # archive has enough model-visible markers to pass the default filter.
    ds = FullImageDataset(
        zarr_dir=str(config.zarr_path),
        dct_config=config,
        keep_fovs={"fan_uf_codex_HBM568.NGPL.345-654418415bed5ecb9596b17a0320a2c6"},
        min_channels=3,
    )
    assert len(ds.indices) > 0


def test_min_channels_high_threshold_excludes_panel():
    from deepcell_types.training.dataset import FullImageDataset
    from deepcell_types.training.config import TissueNetConfig

    config = TissueNetConfig(_archive_path())

    ds = FullImageDataset(
        zarr_dir=str(config.zarr_path),
        dct_config=config,
        keep_fovs={"fan_uf_codex_HBM568.NGPL.345-654418415bed5ecb9596b17a0320a2c6"},
        min_channels=100,
    )
    assert len(ds.indices) == 0


def test_min_channels_zero_includes_all():
    from deepcell_types.training.dataset import FullImageDataset
    from deepcell_types.training.config import TissueNetConfig

    config = TissueNetConfig(_archive_path())

    ds = FullImageDataset(
        zarr_dir=str(config.zarr_path),
        dct_config=config,
        keep_fovs={"fan_uf_codex_HBM568.NGPL.345-654418415bed5ecb9596b17a0320a2c6"},
        min_channels=0,
    )
    assert len(ds.indices) > 0, "fan_uf should be included with min_channels=0"
