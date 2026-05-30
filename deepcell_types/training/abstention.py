"""IQR-fence-based CT abstention (training/eval-side helpers).

Production wire-up of the post-hoc analysis in ``analysis/ct_abstention_iqr.py``.
Given per-cell max-softmax confidences and (tissue, modality) group keys, build
the Tukey lower fence ``Q1 - k * IQR`` per group and abstain on cells below the
fence.

The numpy-only fence computation lives in
:mod:`deepcell_types.abstention` so that the inference path can import it
without pulling in ``pandas`` or any other ``[train]``-extra dependency.
This module adds the pandas DataFrame plumbing and the hierarchy-aware
macro-F1 helper used by the CLI evaluator (``scripts/predict.py``).

Cell-type quality is reported with a single metric throughout this repo:
**macro-F1** (mean per-class F1 over classes with support). It is robust to
the heavy class imbalance in TissueNet, where overall/weighted accuracy can be
inflated by over-predicting majority classes. See
:func:`deepcell_types.training.baseline_features._conf_mat_summary` for the
canonical formula shared by the main model and the baselines.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Public re-export so existing callers of
# ``from deepcell_types.training.abstention import compute_iqr_fence``
# keep working. The implementation has moved to the inference-side module
# so ``deepcell_types.predict`` can import it without [train] extras.
from ..abstention import ABSTENTION_LABEL as ABSTENTION_LABEL  # noqa: F401
from ..abstention import compute_iqr_fence as compute_iqr_fence


def apply_abstention(
    df,
    k: float,
    group_cols: Sequence[str] = ("tissue", "modality"),
    max_softmax_col: str = "_max_softmax",
    pred_col: str = "predicted_ct",
    sentinel=ABSTENTION_LABEL,
):
    """Apply IQR-fence abstention in-place-style and return the modified frame.

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
    # can still be imported (for ``compute_iqr_fence`` etc.) without it.
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


def hierarchical_macro_f1(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    classes: Sequence[str],
    hierarchy: dict,
) -> float:
    """Macro-F1 with parent->child credit.

    Builds a confusion matrix over ``classes``, applies the same
    hierarchy forgiveness as the main model (a prediction of a child cell
    type when the true label is its declared parent is moved onto the
    diagonal via :func:`deepcell_types.training.utils.adjust_conf_mat_hierarchy`),
    then reduces it with the canonical
    :func:`deepcell_types.training.baseline_features._conf_mat_summary` so the
    number is directly comparable to ``ct_macro_f1`` from
    ``LossesAndMetrics.compute()`` and the baseline reports.

    Labels outside ``classes`` (e.g. the abstention ``"Unknown"`` sentinel) are
    dropped by ``sklearn.metrics.confusion_matrix(labels=classes)``; callers that
    want abstained cells excluded should pass only the kept rows.
    """
    from sklearn.metrics import confusion_matrix

    from .baseline_features import _conf_mat_summary
    from .utils import adjust_conf_mat_hierarchy

    classes = list(classes)
    if len(true_labels) == 0:
        return 0.0
    ct2idx = {c: i for i, c in enumerate(classes)}
    conf_mat = confusion_matrix(true_labels, pred_labels, labels=classes)
    if hierarchy:
        conf_mat = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)
    return _conf_mat_summary(conf_mat)["macro_f1"]
