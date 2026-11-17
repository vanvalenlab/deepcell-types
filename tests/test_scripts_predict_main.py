"""End-to-end tests for ``scripts/predict.py::main`` — the evaluation CLI that
generates the metrics/predictions reported in the paper. Previously only a
pure helper (``_resolve_marker_embeddings``, see ``test_scripts_predict.py``)
was covered; ``main`` itself (checkpoint loading, dataloader construction,
prediction, CSV assembly, optional CT abstention) had zero direct coverage.

Builds a tiny synthetic training-shaped zarr archive (2 FOVs, 4 cells, 3
classes) via the real ``zarr`` library — the same archive layout
``FullImageDataset``/``TissueNetConfig`` read in production — plus a matching
checkpoint, then drives ``main()`` with ``click.testing.CliRunner``.
"""

import numpy as np
import pytest
import torch
import zarr
from click.testing import CliRunner

from deepcell_types.model import create_model
from deepcell_types.training.config import TissueNetConfig
from scripts.predict import main


def _make_archive(root_path):
    """A minimal but real training-shaped archive: 2 FOVs, 4 cells, 3 classes.

    Layout mirrors production: root attrs carry the vocab, each FOV group has
    modality/tissue attrs, a ``cell_types/annotations`` group with
    ``standardized_source`` (cell-index-keyed labels), and a ``preprocessed``
    group with ``raw``/``mask`` arrays plus ``centroids``.
    """
    root = zarr.open_group(str(root_path), mode="w")
    root.attrs["cell_type_mapping"] = {"Bcell": 0, "Tumor": 1, "Myeloid": 2}
    root.attrs["all_standardized_channels"] = ["CD45", "PanCK", "CD68"]
    root.attrs["all_standardized_cell_types"] = ["Bcell", "Tumor", "Myeloid"]

    rng = np.random.default_rng(0)

    def _add_fov(name, modality, tissue, labels):
        g = root.create_group(name)
        g.attrs["modality"] = modality
        g.attrs["tissue"] = tissue
        ann = g.create_group("cell_types").create_group("annotations")
        ann.attrs["standardized_source"] = labels
        preproc = g.create_group("preprocessed")
        preproc.attrs["channel_names"] = ["CD45", "PanCK", "CD68"]
        preproc.attrs["centroids"] = {"1": [16, 16], "2": [16, 48]}
        raw = rng.random((3, 64, 64), dtype=np.float64).astype(np.float32) + 0.1
        preproc["raw"] = raw
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:22, 10:22] = 1
        mask[10:22, 42:54] = 2
        preproc["mask"] = mask

    _add_fov("fov_a", "IMC", "lung", {"Bcell": [1], "Tumor": [2]})
    _add_fov("fov_b", "CODEX", "liver", {"Tumor": [1], "Myeloid": [2]})
    return root_path


def _build_checkpoint(config, tmp_path):
    """Build a small self-describing checkpoint matching scripts/train.py's
    bundling contract (model + config + ct2idx + canonical_channels), sized
    small (resnet_base_channels=4, ct_head_width=32, ct_head_depth=1) for a
    fast CPU forward pass. ``n_layers``/``d_model`` are NOT stored in
    "config" because ``scripts/predict.py::main`` hardcodes d_model=256 and
    does not read n_layers from the checkpoint, so the checkpoint's model
    must be built with the same (default) values main() will use.
    """
    marker_embeddings = np.zeros((len(config.marker2idx), 8), dtype=np.float32)
    model = create_model(
        config,
        marker_embeddings,
        d_model=256,
        n_heads=8,
        n_layers=4,
        resnet_base_channels=4,
        spatial_pool_size=1,
        use_conditioned_mp_head=True,
        compat_marker0_zero=True,
        ct_head_width=32,
        ct_head_depth=1,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "resnet_channels": 4,
                "spatial_pool_size": 1,
                "n_heads": 8,
                "use_conditioned_mp_head": True,
                "compat_marker0_zero": True,
            },
            "ct2idx": dict(config.ct2idx),
            "canonical_channels": list(config.marker2idx.keys()),
        },
        ckpt_path,
    )
    return ckpt_path


def _run_main(archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=()):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = [
        "--model_path", str(ckpt_path),
        "--zarr_dir", str(archive_path),
        "--device_num", "cpu",
        "--batch_size", "2",
        "--num_workers", "0",
        "--model_name", "test_model",
        *extra_args,
    ]
    return runner.invoke(main, args, catch_exceptions=True)


@pytest.fixture
def archive_and_checkpoint(tmp_path):
    archive_path = tmp_path / "train.zarr"
    _make_archive(archive_path)
    config = TissueNetConfig(archive_path)
    ckpt_path = _build_checkpoint(config, tmp_path)
    return archive_path, ckpt_path, config


def test_main_runs_to_completion_and_writes_expected_csv(
    archive_and_checkpoint, tmp_path, monkeypatch
):
    archive_path, ckpt_path, config = archive_and_checkpoint

    result = _run_main(
        archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=["--ct_abstention_k", "0"]
    )

    assert result.exit_code == 0, result.output + "\n" + repr(result.exception)

    out_csv = tmp_path / "output" / "test_model_prediction.csv"
    assert out_csv.exists()

    import pandas as pd

    df = pd.read_csv(out_csv)
    class_cols = sorted(config.ct2idx, key=config.ct2idx.get)
    expected_cols = class_cols + [
        "cell_type_actual",
        "cell_index",
        "dataset_name",
        "fov_name",
    ]
    assert df.columns.tolist() == expected_cols
    # 2 FOVs x 2 cells each.
    assert len(df) == 4
    # Every row is a valid probability distribution over the class columns.
    np.testing.assert_allclose(df[class_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-4)
    assert set(df["cell_type_actual"]) <= set(config.ct2idx)
    assert set(df["dataset_name"]) == {"fov_a", "fov_b"}


def test_main_ct_abstention_k_zero_omits_abstention_columns(
    archive_and_checkpoint, tmp_path, monkeypatch
):
    """k<=0 (the default) writes the frame as-is: no predicted_ct/abstained
    columns, matching the historical disabled-path output."""
    archive_path, ckpt_path, config = archive_and_checkpoint

    result = _run_main(
        archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=["--ct_abstention_k", "0"]
    )
    assert result.exit_code == 0, result.output

    import pandas as pd

    df = pd.read_csv(tmp_path / "output" / "test_model_prediction.csv")
    assert "predicted_ct" not in df.columns
    assert "abstained" not in df.columns
    assert "predicted_ct_raw" not in df.columns
    assert "CT abstention enabled" not in result.output


def test_main_ct_abstention_k_nonzero_adds_abstention_columns(
    archive_and_checkpoint, tmp_path, monkeypatch
):
    """k>0 enables the IQR-fence abstention path: predicted_ct/abstained/
    predicted_ct_raw columns are added and coverage/macro-F1 are reported."""
    archive_path, ckpt_path, config = archive_and_checkpoint

    result = _run_main(
        archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=["--ct_abstention_k", "0.2"]
    )
    assert result.exit_code == 0, result.output

    import pandas as pd

    df = pd.read_csv(tmp_path / "output" / "test_model_prediction.csv")
    assert "predicted_ct" in df.columns
    assert "abstained" in df.columns
    assert "predicted_ct_raw" in df.columns
    assert set(df["predicted_ct"]) <= set(config.ct2idx)
    assert df["abstained"].dtype == bool

    assert "CT abstention enabled (k=0.2)" in result.output
    assert "Coverage:" in result.output


def test_main_reports_hierarchical_macro_f1_pre_and_post_abstention(
    archive_and_checkpoint, tmp_path, monkeypatch
):
    """When abstention is enabled, main() prints the pre-abstention (full
    coverage) and post-abstention (kept cells) hierarchical macro-F1 so a
    human running the CLI can see the abstention/accuracy tradeoff."""
    archive_path, ckpt_path, config = archive_and_checkpoint

    result = _run_main(
        archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=["--ct_abstention_k", "0.2"]
    )
    assert result.exit_code == 0, result.output

    assert "Macro F1 on kept cells:" in result.output
    assert "with no abstention" in result.output


def test_main_default_cli_abstention_is_disabled(archive_and_checkpoint, tmp_path, monkeypatch):
    """--ct_abstention_k defaults to 0.0 (disabled) when the flag is omitted
    entirely — guards against silently re-enabling the benchmark-tuned
    default at the CLI layer."""
    archive_path, ckpt_path, config = archive_and_checkpoint

    result = _run_main(archive_path, ckpt_path, tmp_path, monkeypatch, extra_args=[])
    assert result.exit_code == 0, result.output

    import pandas as pd

    df = pd.read_csv(tmp_path / "output" / "test_model_prediction.csv")
    assert "predicted_ct" not in df.columns
