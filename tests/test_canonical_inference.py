import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from numcodecs import Zstd

from deepcell_types.model import create_model
from deepcell_types.dataset import PatchDataset
from deepcell_types.config import DCTConfig
from deepcell_types.predict import (
    _CANONICAL_CT2IDX_SHA256,
    _InferenceResultBuffer,
    _build_model,
    _ct2idx_ordering_sha256,
    _model_path,
    predict,
    validate_checkpoint_vocabulary,
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

    cell_types, top_probs, cell_index, probs = logger.get_result()

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
    # Self-describing checkpoint (matches scripts/train.py): bundle the
    # vocabulary so the ordering guard validates against it.
    torch.save(
        {
            "model": model.state_dict(),
            "ct2idx": dict(config.ct2idx),
            "canonical_channels": list(config.marker2idx.keys()),
        },
        checkpoint_path,
    )

    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cell_types = predict(
            raw,
            mask,
            ["CD45"],
            0.5,
            model_name=str(checkpoint_path),
            device="cpu",
            batch_size=1,
            num_workers=0,
            zarr_path=archive_path,
        )

    assert len(cell_types) == 1
    assert cell_types[0] in config.ct2idx
    # The checkpoint records no n_heads / compat_marker0_zero, so the fallback
    # warning fires and (stacklevel=3) points at this caller, not predict.py.
    fallback_warning = next(
        rec for rec in caught if "Checkpoint config does not record" in str(rec.message)
    )
    assert Path(fallback_warning.filename).name == Path(__file__).name


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
    # Self-describing checkpoint (matches scripts/train.py): bundle the
    # vocabulary so the ordering guard validates against it.
    torch.save(
        {
            "model": model.state_dict(),
            "ct2idx": dict(config.ct2idx),
            "canonical_channels": list(config.marker2idx.keys()),
        },
        ckpt_path,
    )
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


def test_patch_dataset_rejects_only_unknown_channel_names(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1
    with pytest.raises(ValueError, match="No input channels matched"):
        PatchDataset(raw, mask, ["FAKE_MARKER_XYZ_000"], 0.5, config)


def test_predict_returns_prediction_result_when_requested(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)

    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    from deepcell_types import PredictionResult, predict as top_predict

    result = top_predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        model_name=str(ckpt_path),
        device="cpu",
        batch_size=1,
        num_workers=0,
        zarr_path=archive_path,
        return_probabilities=True,
    )
    assert isinstance(result, PredictionResult)
    assert len(result.cell_types) == 1
    assert result.probabilities.shape == (1, len(config.ct2idx))
    assert result.cell_indices.tolist() == [1]
    assert np.isclose(result.probabilities.sum(), 1.0)


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


# ---------------------------------------------------------------------------
# Gap coverage: multi-chunk zarr decode, archive resolution, abstention
# wiring, and the empty-mask contract.
# ---------------------------------------------------------------------------


def _write_v3_1d_array_chunked(array_dir, values, chunk_len):
    """Like _write_v3_1d_array but with an arbitrary chunk length (the helper
    above hard-codes chunk_shape=[1]); exercises the multi-element-per-chunk
    decode path of ``_read_v3_1d_array``, including a short final chunk."""
    array_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values).astype(np.int64)
    n = int(values.shape[0])
    meta = {
        "shape": [n],
        "data_type": "int64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [chunk_len]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
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
    for ci, start in enumerate(range(0, n, chunk_len)):
        chunk = values[start : start + chunk_len]  # final chunk may be short
        (chunk_root / str(ci)).write_bytes(zstd.encode(chunk.tobytes()))


@pytest.mark.parametrize("chunk_len", [1, 2, 4, 16])
def test_read_v3_1d_array_multichunk_roundtrip(tmp_path, chunk_len):
    from deepcell_types.config import _read_v3_1d_array

    values = list(range(10))  # 10 % 4 != 0 -> exercises a short final chunk
    array_dir = tmp_path / "arr"
    _write_v3_1d_array_chunked(array_dir, values, chunk_len)
    assert _read_v3_1d_array(array_dir).tolist() == values


def test_config_resolves_archive_from_env_var(tmp_path, monkeypatch):
    archive_path = _make_archive(tmp_path)
    monkeypatch.setenv("DEEPCELL_TYPES_ZARR_PATH", str(archive_path))
    config = DCTConfig()  # no explicit zarr_path -> must resolve via env var
    assert config.ct2idx == {"Bcell": 0, "Tumor": 1}


def test_config_raises_when_explicit_archive_missing(tmp_path, monkeypatch):
    # An explicitly-passed zarr_path that doesn't exist is an error (not a
    # silent fall-through to the packaged vocab).
    monkeypatch.delenv("DEEPCELL_TYPES_ZARR_PATH", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="zarr archive"):
        DCTConfig(zarr_path=tmp_path / "does-not-exist.zarr")


def test_config_falls_back_to_packaged_vocab(monkeypatch):
    # No archive anywhere -> DCTConfig loads the packaged vocab.json snapshot,
    # so predict() works without the (large) TissueNet archive.
    monkeypatch.delenv("DEEPCELL_TYPES_ZARR_PATH", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    config = DCTConfig()  # no zarr_path, no env -> packaged vocab
    assert config.zarr_path is None
    assert len(config.ct2idx) == 51
    assert len(config.marker2idx) == 278
    first = config.master_channels[0]
    assert config.resolve_channel_name(first) == first
    assert config.resolve_channel_name(first.lower()) == first
    # the tissue->celltype mapping is archive-only
    with pytest.raises(RuntimeError, match="zarr archive"):
        config.get_tct_mapping()


def _four_cell_mask():
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[6:18, 6:18] = 1
    mask[6:18, 40:52] = 2
    mask[40:52, 6:18] = 3
    mask[40:52, 40:52] = 4
    return mask


def test_predict_abstention_wiring(tmp_path):
    from deepcell_types.abstention import ABSTENTION_LABEL

    torch.manual_seed(0)
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)
    raw = np.ones((1, 80, 80), dtype=np.float32)

    res = predict(
        raw,
        _four_cell_mask(),
        ["CD45"],
        0.5,
        model_name=str(ckpt_path),
        device="cpu",
        zarr_path=archive_path,
        ct_abstention_k=0.2,
        return_probabilities=True,
        num_workers=0,
    )

    n = len(res.cell_types)
    assert n == 4
    assert res.abstained.shape == (n,)
    # A cell carries the "Unknown" sentinel iff it was flagged abstained.
    for i in range(n):
        assert (res.cell_types[i] == ABSTENTION_LABEL) == bool(res.abstained[i])
    # The pre-abstention labels are never the sentinel (no ct is named "Unknown").
    assert ABSTENTION_LABEL not in res.cell_types_raw


def test_predict_abstention_disabled_with_k_zero(tmp_path):
    from deepcell_types.abstention import ABSTENTION_LABEL

    torch.manual_seed(0)
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)
    raw = np.ones((1, 80, 80), dtype=np.float32)

    res = predict(
        raw,
        _four_cell_mask(),
        ["CD45"],
        0.5,
        model_name=str(ckpt_path),
        device="cpu",
        zarr_path=archive_path,
        ct_abstention_k=0,
        return_probabilities=True,
        num_workers=0,
    )
    assert not res.abstained.any()
    assert res.cell_types == res.cell_types_raw
    assert ABSTENTION_LABEL not in res.cell_types


def test_predict_empty_mask_returns_empty(tmp_path):
    archive_path = _make_archive(tmp_path)
    config = DCTConfig(zarr_path=archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)
    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)  # all background -> no cells

    out = predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        model_name=str(ckpt_path),
        device="cpu",
        zarr_path=archive_path,
        num_workers=0,
    )
    assert out == []
    res = predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        model_name=str(ckpt_path),
        device="cpu",
        zarr_path=archive_path,
        num_workers=0,
        return_probabilities=True,
    )
    assert len(res.cell_types) == 0
    assert res.probabilities.shape == (0, len(config.ct2idx))


# --- Vocabulary ordering guard (validate_checkpoint_vocabulary) -------------


def test_legacy_checkpoint_requires_canonical_ordering():
    """A checkpoint without a bundled ct2idx (the released v0.1.0 artifact)
    must be paired with the canonical cell-type ordering; a permuted vocabulary
    is rejected rather than silently mislabeling cells."""
    config = DCTConfig()  # packaged vocab.json -> canonical 51-class ordering
    legacy_ckpt = {"canonical_channels": list(config.marker2idx.keys())}

    # Canonical ordering passes; the hash is order-independent (sorted by index).
    validate_checkpoint_vocabulary(legacy_ckpt, config.ct2idx, config.marker2idx)
    assert _ct2idx_ordering_sha256(config.ct2idx) == _CANONICAL_CT2IDX_SHA256
    assert (
        _ct2idx_ordering_sha256(dict(reversed(list(config.ct2idx.items()))))
        == _CANONICAL_CT2IDX_SHA256
    )

    # Swapping two cell types' indices (same count) is rejected.
    permuted = dict(config.ct2idx)
    names = list(permuted)
    permuted[names[0]], permuted[names[1]] = permuted[names[1]], permuted[names[0]]
    with pytest.raises(ValueError, match="canonical v0.1.0 ordering"):
        validate_checkpoint_vocabulary(legacy_ckpt, permuted, config.marker2idx)


def test_bundled_ct2idx_mismatch_is_rejected():
    """A self-describing checkpoint whose bundled ct2idx disagrees with the
    inference vocabulary is rejected (ordering, not just count)."""
    config = DCTConfig()
    permuted = dict(config.ct2idx)
    names = list(permuted)
    permuted[names[0]], permuted[names[1]] = permuted[names[1]], permuted[names[0]]
    ckpt = {"ct2idx": permuted}
    with pytest.raises(ValueError, match="ct2idx ordering does not match"):
        validate_checkpoint_vocabulary(ckpt, config.ct2idx, config.marker2idx)


def test_marker_ordering_uses_indices_not_dict_insertion():
    checkpoint = {"ct2idx": {"A": 0, "B": 1}, "canonical_channels": ["M1", "M2"]}

    # marker2idx is compared by numeric index, so dict insertion order does not
    # matter — only the index a name maps to.
    validate_checkpoint_vocabulary(
        checkpoint,
        {"A": 0, "B": 1},
        {"M2": 1, "M1": 0},
    )

    with pytest.raises(ValueError, match="marker ordering"):
        validate_checkpoint_vocabulary(
            checkpoint,
            {"A": 0, "B": 1},
            {"M1": 1, "M2": 0},
        )
