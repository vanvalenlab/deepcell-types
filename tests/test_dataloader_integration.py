"""End-to-end tests for ``FullImageDataset.__getitem__`` and
``create_dataloader`` against a real (synthetic) training-shaped zarr
archive.

Previously ``FullImageDataset`` was only exercised via ``__new__``-stubs
(``tests/test_channel_aliases.py``) or reimplemented fragments
(``tests/test_zero_channel_masking.py``); no test constructed a real instance
end-to-end or drove a real ``DataLoader`` batch. This builds a tiny archive
(2 FOVs, 4 cells, 3 classes; one FOV has an all-zero acquisition channel) via
the real ``zarr`` library — the layout ``TissueNetConfig``/``FullImageDataset``
read in production.
"""

import numpy as np
import pytest
import torch
import zarr

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import FullImageDataset, create_dataloader
from deepcell_types.training.utils import BatchData


def _make_archive(root_path):
    """2 FOVs, 4 cells, 3 classes.

    ``fov_a`` (IMC/lung): CD68 (channel index 2) is all-zero across the whole
    FOV — acquisition-empty, exercising the FOV-zero-channel masking path in
    ``__getitem__``. ``fov_b`` (CODEX/liver): all 3 channels carry signal.
    """
    root = zarr.open_group(str(root_path), mode="w")
    root.attrs["cell_type_mapping"] = {"Bcell": 0, "Tumor": 1, "Myeloid": 2}
    root.attrs["all_standardized_channels"] = ["CD45", "PanCK", "CD68"]
    root.attrs["all_standardized_cell_types"] = ["Bcell", "Tumor", "Myeloid"]

    rng = np.random.default_rng(0)

    g = root.create_group("fov_a")
    g.attrs["modality"] = "IMC"
    g.attrs["tissue"] = "lung"
    ann = g.create_group("cell_types").create_group("annotations")
    ann.attrs["standardized_source"] = {"Bcell": [1], "Tumor": [2]}
    preproc = g.create_group("preprocessed")
    preproc.attrs["channel_names"] = ["CD45", "PanCK", "CD68"]
    preproc.attrs["centroids"] = {"1": [16, 16], "2": [16, 48]}
    raw = rng.random((3, 64, 64), dtype=np.float64).astype(np.float32) + 0.1
    raw[2] = 0.0  # CD68 acquisition-empty on this FOV
    preproc["raw"] = raw
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:22, 10:22] = 1
    mask[10:22, 42:54] = 2
    preproc["mask"] = mask

    g2 = root.create_group("fov_b")
    g2.attrs["modality"] = "CODEX"
    g2.attrs["tissue"] = "liver"
    ann2 = g2.create_group("cell_types").create_group("annotations")
    ann2.attrs["standardized_source"] = {"Tumor": [1], "Myeloid": [2]}
    preproc2 = g2.create_group("preprocessed")
    preproc2.attrs["channel_names"] = ["CD45", "PanCK", "CD68"]
    preproc2.attrs["centroids"] = {"1": [16, 16], "2": [16, 48]}
    raw2 = rng.random((3, 64, 64), dtype=np.float64).astype(np.float32) + 0.1
    preproc2["raw"] = raw2
    mask2 = np.zeros((64, 64), dtype=np.int32)
    mask2[10:22, 10:22] = 1
    mask2[10:22, 42:54] = 2
    preproc2["mask"] = mask2

    return root_path


@pytest.fixture
def archive_and_config(tmp_path):
    archive_path = tmp_path / "train.zarr"
    _make_archive(archive_path)
    config = TissueNetConfig(archive_path)
    return archive_path, config


def test_config_reads_training_archive_contract(archive_and_config):
    archive_path, config = archive_and_config
    assert config.ct2idx == {"Bcell": 0, "Tumor": 1, "Myeloid": 2}
    assert config.marker2idx == {"CD45": 0, "PanCK": 1, "CD68": 2}
    assert config.domain2idx == {"CODEX": 0, "IMC": 1}
    assert config.tissue2idx == {"liver": 0, "lung": 1}


def test_full_image_dataset_indexes_all_cells(archive_and_config):
    archive_path, config = archive_and_config
    dataset = FullImageDataset(archive_path, dct_config=config)

    assert len(dataset) == 4
    fov_names = {rec.fov_name for rec in dataset.indices}
    assert fov_names == {"fov_a", "fov_b"}
    ct_labels = sorted(rec.ct_label_standard for rec in dataset.indices)
    assert ct_labels == ["Bcell", "Myeloid", "Tumor", "Tumor"]


def test_getitem_returns_correctly_shaped_and_typed_tensors(archive_and_config):
    archive_path, config = archive_and_config
    dataset = FullImageDataset(archive_path, dct_config=config)

    (
        sample,
        spatial_context,
        ch_idx,
        attn_mask,
        ct_idx,
        domain_idx,
        mp,
        vm,
        cell_idx,
        dataset_name,
        fov_name,
        tissue_idx,
    ) = dataset[0]

    max_channels = dataset.max_channels  # 80 (dct_config.MAX_NUM_CHANNELS)
    crop = dataset.output_size  # 32

    assert sample.shape == (max_channels, 1, crop, crop)
    assert sample.dtype == torch.float32
    assert spatial_context.shape == (3, crop, crop)
    assert spatial_context.dtype == torch.float32
    assert ch_idx.shape == (max_channels,)
    assert ch_idx.dtype == torch.int64
    assert attn_mask.shape == (max_channels,)
    assert attn_mask.dtype == torch.bool
    assert mp.shape == (max_channels,)
    assert mp.dtype == torch.float32
    assert vm.shape == (max_channels,)
    assert vm.dtype == torch.bool
    assert isinstance(ct_idx, int)
    assert isinstance(domain_idx, int)
    assert isinstance(cell_idx, int)
    assert isinstance(dataset_name, str)
    assert isinstance(fov_name, str)
    assert tissue_idx.shape == ()
    assert tissue_idx.dtype == torch.int64

    # Only the first 3 channel slots are real (CD45, PanCK, CD68); the rest
    # are padding (ch_idx == -1).
    assert ch_idx[:3].tolist() == [0, 1, 2]
    assert (ch_idx[3:] == -1).all()
    assert attn_mask[3:].all()  # padding is always attention-masked


def test_getitem_masks_all_zero_channel_for_affected_fov_only(archive_and_config):
    """fov_a's CD68 (channel slot 2) is all-zero across the FOV and must be
    attention-masked (attn_mask[2] == True) for BOTH of its cells, while
    fov_b's CD68 carries real signal and is unmasked."""
    archive_path, config = archive_and_config
    dataset = FullImageDataset(archive_path, dct_config=config)

    for i, record in enumerate(dataset.indices):
        item = dataset[i]
        attn_mask = item[3]
        sample = item[0]
        if record.fov_name == "fov_a":
            assert bool(attn_mask[0]) is False  # CD45: real signal
            assert bool(attn_mask[1]) is False  # PanCK: real signal
            assert bool(attn_mask[2]) is True  # CD68: acquisition-empty -> masked
            # Masked-out channel's sample is cleared to the -1.0 padding sentinel.
            assert torch.all(sample[2] == -1.0)
        else:
            assert record.fov_name == "fov_b"
            assert bool(attn_mask[0]) is False
            assert bool(attn_mask[1]) is False
            assert bool(attn_mask[2]) is False  # CD68 has real signal here


def test_getitem_self_mask_is_nonempty(archive_and_config):
    """spatial_context[0] is the self-mask; a labeled cell must have a
    non-empty self-mask (the production tripwire in __getitem__)."""
    archive_path, config = archive_and_config
    dataset = FullImageDataset(archive_path, dct_config=config)
    for i in range(len(dataset)):
        spatial_context = dataset[i][1]
        assert float(spatial_context[0].sum()) > 0


def test_create_dataloader_only_test_yields_correctly_shaped_batch(archive_and_config):
    archive_path, config = archive_and_config

    train_loader, test_loader, metadata = create_dataloader(
        zarr_dir=archive_path,
        dct_config=config,
        batch_size=2,
        num_workers=0,
        only_test=True,
    )

    assert train_loader is None
    assert metadata["num_samples"] == 4

    batch = next(iter(test_loader))
    batch_data = BatchData(*batch)

    B = 2
    max_channels = config.MAX_NUM_CHANNELS
    crop = config.OUTPUT_SIZE

    assert batch_data.sample.shape == (B, max_channels, 1, crop, crop)
    assert batch_data.spatial_context.shape == (B, 3, crop, crop)
    assert batch_data.ch_idx.shape == (B, max_channels)
    assert batch_data.mask.shape == (B, max_channels)
    assert batch_data.ct_idx.shape == (B,)
    assert batch_data.domain_idx.shape == (B,)
    assert batch_data.marker_positivity.shape == (B, max_channels)
    assert batch_data.marker_positivity_mask.shape == (B, max_channels)
    assert batch_data.cell_index.shape == (B,)
    assert len(batch_data.dataset_name) == B
    assert len(batch_data.fov_name) == B
    assert batch_data.tissue_idx.shape == (B,)

    # Every batch element assembled without error and cell types/domains
    # resolve to valid indices.
    assert set(batch_data.ct_idx.tolist()) <= set(config.ct2idx.values())
    assert set(batch_data.domain_idx.tolist()) <= set(config.domain2idx.values())


def test_create_dataloader_only_test_covers_all_cells_across_batches(archive_and_config):
    archive_path, config = archive_and_config

    _, test_loader, metadata = create_dataloader(
        zarr_dir=archive_path,
        dct_config=config,
        batch_size=2,
        num_workers=0,
        only_test=True,
    )

    seen_cell_index = []
    seen_dataset = []
    for batch in test_loader:
        batch_data = BatchData(*batch)
        seen_cell_index.extend(batch_data.cell_index.tolist())
        seen_dataset.extend(batch_data.dataset_name)

    assert len(seen_cell_index) == metadata["num_samples"] == 4
    assert sorted(seen_dataset) == ["fov_a", "fov_a", "fov_b", "fov_b"]
