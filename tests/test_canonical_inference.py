import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from numcodecs import Zstd

from deepcell_types.model import create_model
from deepcell_types.dataset import PatchDataset
from deepcell_types.dct_kit.config import DCTConfig
from deepcell_types.predict import (
    _InferenceResultBuffer,
    _build_model,
    _excluded_celltype_indices,
    _model_path,
    predict,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_v3_1d_array(array_dir, values):
    array_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if values.dtype.kind in {"U", "O"}:
        max_len = max((len(str(value)) for value in values), default=1)
        dtype = np.dtype(f"<U{max_len}")
        values = values.astype(dtype)
        data_type = {
            "name": "fixed_length_utf32",
            "configuration": {"length_bytes": dtype.itemsize},
        }
        fill_value = ""
    else:
        values = values.astype(np.int64)
        data_type = "int64"
        fill_value = 0

    meta = {
        "shape": [int(values.shape[0])],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    _write_json(array_dir / "zarr.json", meta)

    zstd = Zstd(level=0)
    chunk_root = array_dir / "c"
    chunk_root.mkdir(parents=True, exist_ok=True)
    for idx, value in enumerate(values):
        (chunk_root / str(idx)).write_bytes(zstd.encode(np.asarray([value]).tobytes()))


def _make_archive(tmp_path):
    root = tmp_path / "mini-tissuenet.zarr"
    _write_json(
        root / "zarr.json",
        {
            "attributes": {
                "schema_version": "v8.0",
                "archive_version": "v8.0",
                "cell_type_mapping": {"Bcell": 0, "Tumor": 1},
                "all_standardized_channels": ["CD45", "PanCK", "CD107A"],
            }
        },
    )

    fovs = [
        ("IMC", "lung", "cohort_a", "sample_a", "fov_0", ["Bcell", "Tumor"]),
        ("CODEX", "liver", "cohort_b", "sample_b", "fov_0", ["Tumor"]),
    ]
    for modality, tissue, cohort, sample, fov, cell_types in fovs:
        fov_dir = root / modality / tissue / cohort / sample / fov
        _write_json(
            fov_dir / "zarr.json",
            {"attributes": {"modality": modality, "tissue": tissue}},
        )
        _write_v3_1d_array(
            fov_dir / "preprocessed" / "cell_type_info" / "cell_type",
            cell_types,
        )

    return root


def test_canonical_config_reads_contract_from_archive(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)

    assert config.MAX_NUM_CHANNELS == 80
    assert config.CROP_SIZE == 32
    assert config.NUM_DOMAINS == 2
    assert config.ct2idx == {"Bcell": 0, "Tumor": 1}
    assert config.marker2idx == {"CD45": 0, "PanCK": 1, "CD107A": 2}
    assert config.get_tct_mapping() == {
        "liver": ["Tumor"],
        "lung": ["Bcell", "Tumor"],
    }


def test_patch_dataset_can_emit_archive_backed_canonical_batch(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    raw = np.ones((2, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    dataset = PatchDataset(raw, mask, ["cd45", "Pan-Cytokeratin"], 0.5, config)
    sample, spatial_context, ch_idx, attn_mask, cell_index = next(iter(dataset))

    assert sample.shape == (80, 1, 32, 32)
    assert spatial_context.shape == (3, 32, 32)
    assert ch_idx.shape == (80,)
    assert attn_mask.shape == (80,)
    assert ch_idx[0].item() == config.marker2idx["CD45"]
    assert ch_idx[1].item() == config.marker2idx["PanCK"]
    assert attn_mask[0].item() is False
    assert attn_mask[2:].all().item() is True
    assert cell_index == 1


def test_patch_dataset_shards_iterable_workers_without_duplicates(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    raw = np.ones((1, 80, 80), dtype=np.float32)
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[8:20, 8:20] = 1
    mask[32:44, 32:44] = 2
    mask[56:68, 56:68] = 3

    dataset = PatchDataset(raw, mask, ["CD45"], 0.5, config)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=2)

    cell_indices = []
    for *_, cell_index in data_loader:
        cell_indices.extend(cell_index.tolist())

    assert sorted(cell_indices) == [1, 2, 3]
    assert len(cell_indices) == len(set(cell_indices)) == len(dataset)


def test_model_path_treats_dotted_model_names_as_cached_models():
    model_cache = Path.home() / ".deepcell" / "models"

    assert _model_path("specific_ct_v0.1") == model_cache / "specific_ct_v0.1.pt"
    assert (
        _model_path("deepcell-types_specific_ct_v0.1")
        == model_cache / "deepcell-types_specific_ct_v0.1.pt"
    )
    assert _model_path("checkpoint.pt") == Path("checkpoint.pt")


def test_pred_logger_returns_results_ordered_by_cell_index(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    logger = _InferenceResultBuffer(config)

    logger.log(
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([2, 1], dtype=np.int64),
    )

    cell_types, top_probs, cell_index = logger.get_result()

    assert list(cell_index) == [1, 2]
    assert list(top_probs) == [1.0, 1.0]
    assert cell_types == ["Bcell", "Tumor"]


def test_predict_accepts_archive_backed_canonical_checkpoint_path(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    marker_embeddings = np.zeros((len(config.marker2idx), 8), dtype=np.float32)
    model = create_model(
        config,
        marker_embeddings,
        d_model=32,
        n_heads=8,
        n_layers=1,
        resnet_base_channels=4,
    )
    checkpoint_path = tmp_path / "canonical.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    cell_types = predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        str(checkpoint_path),
        "cpu",
        batch_size=1,
        num_workers=0,
        zarr_path=archive_path,
    )

    assert len(cell_types) == 1
    assert cell_types[0] in config.ct2idx


def _build_checkpoint(config, tmp_path, **overrides):
    marker_embeddings = np.zeros((len(config.marker2idx), 8), dtype=np.float32)
    model = create_model(
        config,
        marker_embeddings,
        d_model=32,
        n_heads=8,
        n_layers=1,
        resnet_base_channels=4,
        **overrides,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    return ckpt_path


def test_build_model_raises_on_marker_count_mismatch(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)

    # Hand the checkpoint a config whose marker2idx is missing one entry.
    config._marker2idx = {k: v for k, v in list(config.marker2idx.items())[:-1]}
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    with pytest.raises(ValueError, match="markers"):
        _build_model(checkpoint, config, torch.device("cpu"))


def test_build_model_raises_on_celltype_count_mismatch(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)

    config._ct2idx = {k: v for k, v in list(config.ct2idx.items())[:-1]}
    config.NUM_CELLTYPES = len(config._ct2idx)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    with pytest.raises(ValueError, match="cell types"):
        _build_model(checkpoint, config, torch.device("cpu"))


def test_excluded_celltype_indices_rejects_unknown_tissue(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    with pytest.raises(ValueError, match="Unknown tissue_exclude"):
        _excluded_celltype_indices(config, "not-a-real-tissue", batch_size=2)


def test_excluded_celltype_indices_restricts_to_allowed(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    # liver fixture only annotates "Tumor" — the excluded set must include
    # every index that is NOT Tumor.
    excluded = _excluded_celltype_indices(config, "liver", batch_size=3)
    assert excluded is not None
    assert len(excluded) == 3
    tumor_idx = config.ct2idx["Tumor"]
    for row in excluded:
        assert tumor_idx not in row
        assert set(row) | {tumor_idx} == set(range(len(config.ct2idx)))


def test_excluded_celltype_indices_none_passthrough(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    assert _excluded_celltype_indices(config, None, batch_size=4) is None


def test_patch_dataset_rejects_only_unknown_channel_names(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1
    with pytest.raises(ValueError, match="No input channels matched"):
        PatchDataset(raw, mask, ["FAKE_MARKER_XYZ_000"], 0.5, config)


def test_dct_and_tissuenet_config_agree_on_shared_constants(tmp_path):
    """DCTConfig (inference) and TissueNetConfig (training) coexist by
    design but must agree on the constants that affect patch geometry —
    otherwise inference runs on patches sized differently than training.
    """
    zarr = pytest.importorskip("zarr")  # noqa: F841 — training extra gate
    from deepcell_types.training.config import TissueNetConfig

    archive_path = _make_archive(tmp_path)
    dct = DCTConfig(zarr_path=archive_path)
    tnc = TissueNetConfig(zarr_path=archive_path)

    assert dct.MAX_NUM_CHANNELS == tnc.MAX_NUM_CHANNELS
    assert dct.CROP_SIZE == tnc.CROP_SIZE
    assert dct.STANDARD_MPP_RESOLUTION == tnc.STANDARD_MPP_RESOLUTION
    assert dct.marker2idx == tnc.marker2idx
    assert dct.ct2idx == tnc.ct2idx
