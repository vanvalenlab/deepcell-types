"""Tests for PredLogger.to_dataframe() / write_csv_atomic() / save()."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepcell_types.training.utils import PredLogger


def _logger_with_two_cells() -> PredLogger:
    ct2idx = {"CD4T": 0, "CD8T": 1, "Bcell": 2}
    pl = PredLogger(ct2idx)
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    pl.log(
        labels=np.array([0, 1]),
        probs=probs,
        cell_index=np.array([10, 11]),
        dataset_name=np.array(["ds_a", "ds_a"]),
        fov_name=np.array(["fov0", "fov0"]),
    )
    return pl


def test_to_dataframe_columns_and_values():
    pl = _logger_with_two_cells()
    df = pl.to_dataframe()
    assert list(df.columns) == [
        "CD4T",
        "CD8T",
        "Bcell",
        "cell_type_actual",
        "cell_index",
        "dataset_name",
        "fov_name",
    ]
    assert df["cell_type_actual"].tolist() == ["CD4T", "CD8T"]
    assert df["cell_index"].tolist() == [10, 11]
    np.testing.assert_allclose(
        df[["CD4T", "CD8T", "Bcell"]].to_numpy(),
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        rtol=1e-6,
    )


def test_save_roundtrips_to_dataframe(tmp_path):
    pl = _logger_with_two_cells()
    out = tmp_path / "pred.csv"
    pl.save(out)
    back = pd.read_csv(out)
    pd.testing.assert_frame_equal(back, pl.to_dataframe(), check_dtype=False)


def test_write_csv_atomic_writes_given_frame(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "frame.csv"
    PredLogger.write_csv_atomic(df, out)
    back = pd.read_csv(out)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
