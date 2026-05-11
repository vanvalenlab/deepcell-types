"""IQR-fence-based CT abstention.

Production wire-up of the post-hoc analysis in `analysis/ct_abstention_iqr.py`.
Given per-cell max-softmax confidences and (tissue, modality) group keys, build
the Tukey lower fence `Q1 - k * IQR` per group and abstain on cells below the
fence.

Algorithm (mirrors `analysis/ct_abstention_iqr.py::apply_iqr_abstention`):
1. For each unique group key, look at the max-softmax distribution of cells in
   that group.
2. If fewer than 4 cells in the group, the IQR is undefined — keep all of them.
3. Otherwise compute Q1, Q3, IQR = Q3 - Q1, fence = Q1 - k * IQR.
4. Cells with max_softmax < fence are abstained (kept[i] = False).

See `docs/audits/ct_abstention_iqr_signal_2026-04-28.md` for the Pareto sweep:
  - k = 1.5 (canonical Tukey): near-no-op (~0.23% abstained, +0.02pp macro)
  - k = 0.5 (aggressive): ~10.5% abstained, +3.22pp macro on kept cells
  - k = None (off): default; predict.py emits no `abstained` column.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def compute_iqr_fence(max_softmax: np.ndarray, k: float) -> Optional[float]:
    """Compute the Tukey lower fence Q1 - k * IQR for a 1D array.

    Returns None when fewer than 4 values are supplied (IQR undefined for tiny
    samples; mirrors the analysis script's `len(vals) < 4` guard).
    """
    arr = np.asarray(max_softmax, dtype=np.float64)
    if arr.size < 4:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr)


def apply_abstention(
    df: pd.DataFrame,
    k: float,
    group_cols: Sequence[str] = ("tissue", "modality"),
    max_softmax_col: str = "_max_softmax",
    pred_col: str = "predicted_ct",
    sentinel: int = -1,
) -> pd.DataFrame:
    """Apply IQR-fence abstention in-place-style and return the modified frame.

    Adds two columns to `df`:
      - `abstained` (bool): True iff max_softmax < fence within the cell's group.
      - `predicted_ct_raw`: a copy of the original `predicted_ct` column.
    Mutates `predicted_ct` so that abstained cells take the integer `sentinel`
    value (default -1) — downstream metric code that ignores -1 will then drop
    abstained cells from the macro/weighted denominator cleanly.

    Per-group fences are computed independently. Groups with fewer than 4 cells
    have no fence (no abstention fires for them — see compute_iqr_fence guard).
    Groups whose max_softmax distribution is degenerate (all identical) yield
    fence == Q1, so `vals >= fence` is all True (no abstention fires).
    """
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
    # be string-like (PyArrow-backed under pandas 2.x infers `str`) or numeric;
    # cast to `object` before assignment so a mixed sentinel (e.g. -1 alongside
    # class-name strings) doesn't trip dtype-strict setitem.
    if abstained.any():
        df[pred_col] = df[pred_col].astype(object)
        df.loc[abstained, pred_col] = sentinel
    return df


def hierarchical_correct(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    hierarchy: dict[str, Iterable[str]],
) -> np.ndarray:
    """Per-cell binary correctness with parent->child credit.

    Mirrors `analysis/ct_abstention_iqr.py::hierarchical_correct` and
    `deepcell_types.training.utils.adjust_conf_mat_hierarchy`: a prediction of a child
    cell type when the true label is its declared parent counts as correct.
    """
    correct = (true_labels == pred_labels)
    for parent, children in hierarchy.items():
        children_list = list(children)
        forgive = (true_labels == parent) & np.isin(pred_labels, children_list)
        correct = correct | forgive
    return correct


def macro_weighted_accuracy(
    true_labels: np.ndarray,
    correct: np.ndarray,
    classes: Sequence[str],
) -> tuple[float, float]:
    """Macro = mean per-class accuracy (only over classes with support).
    Weighted = overall mean accuracy. Mirrors the analysis script.
    """
    accs: list[float] = []
    for c in classes:
        mask = (true_labels == c)
        n = int(mask.sum())
        if n == 0:
            continue
        accs.append(float(correct[mask].mean()))
    macro = float(np.mean(accs)) if accs else 0.0
    weighted = float(correct.mean()) if len(correct) else 0.0
    return macro, weighted
