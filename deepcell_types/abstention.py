"""IQR-fence post-hoc abstention (inference-side public utilities).

This module's **import-time dependencies are numpy-only** so that
``deepcell_types.predict`` can import it without pulling in any
``[train]``-extra dependency (pandas, scikit-learn, etc.). The batched
``apply_abstention`` below imports pandas lazily, inside the function,
preserving that contract.

Algorithm (mirrors ``analysis/ct_abstention_iqr.py``):

1. For each unique group key, look at the max-softmax distribution of cells
   in that group.
2. If fewer than 4 cells are in the group, the IQR is undefined — keep all
   of them.
3. Otherwise compute Q1, Q3, IQR = Q3 - Q1, fence = Q1 - k * IQR.
4. Cells with ``max_softmax < fence`` are abstained.

Operating points:

* ``k = 0.2`` (default; paper headline): chosen to widen macro_F1
  separation over the strongest baseline; substantial macro_F1 lift on
  kept cells.
* ``k = 1.5`` (canonical Tukey): near-no-op (~0.23% abstained,
  +0.02pp macro).
* ``k <= 0``: callers skip abstention. (Passing ``k = 0`` to
  :func:`compute_iqr_fence` directly gives ``fence = Q1``, not a no-op.)

:func:`apply_abstention` (this module) applies the same fence to a pandas
DataFrame of DCT predictions, grouped by FOV — the batched form used by the
DCT eval CLI (``scripts/predict.py``). Abstention is DCT-owned: it is never
applied to baseline predictions.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


ABSTENTION_LABEL = "Unknown"
"""Sentinel cell-type name used for cells flagged as abstained.

Both the Python API (:func:`deepcell_types.predict`) and the CLI
(``scripts/predict.py``) write this string into the predicted-cell-type
output for cells whose max-softmax falls below the IQR fence.
"""


def compute_iqr_fence(max_softmax: np.ndarray, k: float) -> Optional[float]:
    """Compute the Tukey lower fence ``Q1 - k * IQR`` for a 1D array.

    Returns ``None`` when fewer than 4 values are supplied (the IQR is
    undefined for tiny samples; mirrors the analysis script's
    ``len(vals) < 4`` guard).
    """
    arr = np.asarray(max_softmax, dtype=np.float64)
    # Drop non-finite values: a single NaN would otherwise propagate through
    # np.quantile and yield fence=NaN, which silently disables abstention
    # (``x < NaN`` is always False) for the whole group.
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr)


def apply_abstention(
    df,
    k: float,
    group_cols: Sequence[str] = ("tissue", "modality"),
    max_softmax_col: str = "_max_softmax",
    pred_col: str = "predicted_ct",
    sentinel=ABSTENTION_LABEL,
):
    """Apply DeepCellTypes' IQR-fence abstention to a frame of DCT predictions.

    This is the batched form of the per-FOV abstention that
    :func:`deepcell_types.predict` applies at inference time: it computes the
    Tukey lower fence ``Q1 - k * IQR`` per ``group_cols`` group on the
    max-softmax column and relabels below-fence cells to ``sentinel``.

    OWNERSHIP: abstention is a DeepCellTypes capability. This function must
    only ever be applied to **DCT model predictions** — never to baseline
    predictions. Baselines are scored at full coverage (see
    ``tests/baselines/test_no_baseline_abstention.py``).

    Adds two columns to ``df``:
      - ``abstained`` (bool): True iff max_softmax < fence within the cell's group.
      - ``predicted_ct_raw``: a copy of the original ``predicted_ct`` column.
    Mutates ``predicted_ct`` so that abstained cells take the ``sentinel`` value
    (default ``"Unknown"``, matching ``deepcell_types.predict``). Downstream
    metric code that ignores the sentinel cleanly drops abstained cells from
    the macro/weighted denominator.

    Per-group fences are computed independently. Groups with fewer than 4 cells
    have no fence (no abstention fires for them — see :func:`compute_iqr_fence`).
    Groups whose max_softmax distribution is degenerate (all identical) yield
    fence == Q1, so ``vals >= fence`` is all True (no abstention fires).
    """
    # pandas is a [train]-extra dependency; imported lazily so this module
    # keeps its numpy-only import-time contract (deepcell_types.predict imports
    # this module without [train] extras).
    import pandas as pd  # noqa: F401

    if "abstained" in df.columns:
        raise ValueError(
            "`abstained` column already exists; refusing to overwrite. "
            "Pass a frame that has not had abstention applied."
        )
    df = df.copy()
    df["abstained"] = False
    df["predicted_ct_raw"] = df[pred_col].to_numpy().copy()

    max_p = df[max_softmax_col].to_numpy(dtype=np.float64)
    abstained = np.zeros(len(df), dtype=bool)

    # group_cols may contain a single string for convenience
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    grouped = df.groupby(group_cols, sort=False, dropna=False).indices
    for _gkey, idx in grouped.items():
        if len(idx) < 4:
            continue
        fence = compute_iqr_fence(max_p[idx], k)
        if fence is None:
            continue
        abstained[idx] = max_p[idx] < fence

    df["abstained"] = abstained
    # Sentinel out the predicted_ct of abstained cells. The column dtype may
    # be string-like (PyArrow-backed under pandas 2.x infers ``str``) or
    # numeric; cast to ``object`` before assignment so a mixed sentinel
    # alongside class-name strings doesn't trip dtype-strict setitem.
    if abstained.any():
        df[pred_col] = df[pred_col].astype(object)
        df.loc[abstained, pred_col] = sentinel
    return df
