"""Guard: abstention is a DCT-only capability — no baseline ever abstains.

Two enforced invariants:
1. ``save_baseline_predictions`` emits full-coverage output: only the
   probability columns plus (cell_type_actual, cell_index, dataset_name,
   fov_name). No ``abstained`` / ``predicted_ct_raw`` columns, and no cell
   carrying the abstention sentinel.
2. No module under ``deepcell_types/baselines/`` references abstention at all
   (so a future edit that wires a baseline to abstention fails CI).
"""

from __future__ import annotations

import pathlib

import numpy as np

import deepcell_types.baselines as baselines_pkg
from deepcell_types.abstention import ABSTENTION_LABEL
from deepcell_types.training.baseline_features import save_baseline_predictions


def test_save_baseline_predictions_is_full_coverage(tmp_path):
    ct2idx = {"Bcell": 0, "Tcell": 1, "Tumor": 2}
    n = 6
    rng = np.random.default_rng(0)
    y_prob = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    out = tmp_path / "baseline_pred.csv"

    save_baseline_predictions(
        y_true=y_true,
        y_prob=y_prob,
        cell_indices=list(range(n)),
        dataset_names=["ds"] * n,
        fov_names=["fov0"] * n,
        ct2idx=ct2idx,
        output_path=out,
    )

    import pandas as pd

    df = pd.read_csv(out)
    assert "abstained" not in df.columns
    assert "predicted_ct_raw" not in df.columns
    # The abstention sentinel must not appear anywhere in the baseline output.
    assert not (df == ABSTENTION_LABEL).any().any()
    # Exactly the probability columns + 4 metadata columns.
    assert set(df.columns) == set(ct2idx) | {
        "cell_type_actual",
        "cell_index",
        "dataset_name",
        "fov_name",
    }


def test_no_baseline_module_references_abstention():
    pkg_dir = pathlib.Path(baselines_pkg.__file__).parent
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "absten" in text.lower():
            offenders.append(str(py.relative_to(pkg_dir.parent)))
    assert offenders == [], (
        "Baselines must never reference abstention (DCT-only capability); "
        f"found references in: {offenders}"
    )
