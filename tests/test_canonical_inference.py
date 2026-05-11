import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from numcodecs import Zstd

from deepcell_types.model import create_model
from deepcell_types.dataset import PatchDataset
from deepcell_types.dct_kit.config import DCTConfig
from deepcell_types.predict import (
    _InferenceResultBuffer,
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
