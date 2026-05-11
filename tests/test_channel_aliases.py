"""Tests for the strict canonical-channel contract.

The training-time path uses a direct ``marker2idx`` lookup with no alias
resolution and no case-insensitive fallback. Source-data variants must
be canonicalized at ingestion (``hubmap-to-zarr/apply_canonicalization.py``).
"""

from pathlib import Path

import zarr

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import FullImageDataset


def _make_archive_stub(path):
    """Build a minimal zarr v3 archive with the attrs TissueNetConfig reads."""
    root = zarr.open_group(str(path), mode="w")
    root.attrs["all_standardized_channels"] = ["CD45", "HLA-G", "Ki67"]
    root.attrs["all_standardized_cell_types"] = ["Tcell", "Tumor"]
    root.attrs["cell_type_mapping"] = {"Tcell": 0, "Tumor": 1}


def test_resolve_channel_index_canonical_match():
    dataset = FullImageDataset.__new__(FullImageDataset)
    dataset.marker2idx = {"HO1": 0, "Galectin9": 1, "PanCK": 2}
    dataset._idx2marker = {v: k for k, v in dataset.marker2idx.items()}

    assert dataset._resolve_channel_index("HO1") == (0, "HO1")
    assert dataset._resolve_channel_index("Galectin9") == (1, "Galectin9")
    assert dataset._resolve_channel_index("PanCK") == (2, "PanCK")


def test_resolve_channel_index_returns_minus_one_for_non_canonical():
    # Strict contract: source-data variants like "HO-1", "GALECTIN9", "panck"
    # are not canonical and must NOT resolve at runtime. They get masked
    # downstream. Canonicalization is the ingestion step's responsibility.
    dataset = FullImageDataset.__new__(FullImageDataset)
    dataset.marker2idx = {"HO1": 0, "Galectin9": 1, "PanCK": 2}
    dataset._idx2marker = {v: k for k, v in dataset.marker2idx.items()}

    assert dataset._resolve_channel_index("HO-1") == (-1, "HO-1")
    assert dataset._resolve_channel_index("GALECTIN9") == (-1, "GALECTIN9")
    assert dataset._resolve_channel_index("panck") == (-1, "panck")
    assert dataset._resolve_channel_index("CD45") == (-1, "CD45")  # not in this stub


def test_tissuenet_config_does_not_expose_channel_mapping(tmp_path):
    # The runtime alias plumbing was intentionally removed — TissueNetConfig
    # exposes only the canonical registry. This test guards against
    # accidental reintroduction of the channel_mapping property.
    archive = tmp_path / "stub.zarr"
    _make_archive_stub(archive)

    config = TissueNetConfig(str(archive))
    assert not hasattr(config, "channel_mapping")
    # The registry itself is still exposed.
    assert "CD45" in config.marker2idx
    assert "HLA-G" in config.marker2idx


def test_train_checkpoint_bundles_canonical_channels():
    # Documents the contract: scripts/train.py::build_checkpoint includes
    # canonical_channels (= list(dct_config.marker2idx.keys())) so inference
    # can size marker2idx without consulting a vendored YAML or the archive.
    train_src = (
        Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    ).read_text()
    assert '"canonical_channels": list(dct_config.marker2idx.keys())' in train_src
