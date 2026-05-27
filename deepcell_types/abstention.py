"""IQR-fence post-hoc abstention (inference-side public utilities).

This module is intentionally **numpy-only** so that
``deepcell_types.predict`` can import it without pulling in any
``[train]``-extra dependency (pandas, scikit-learn, etc.).

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
* ``k = 0`` (off): no cells are abstained.

The training-side helper :func:`deepcell_types.training.abstention.apply_abstention`
applies this fence to a pandas DataFrame (used by ``scripts/predict.py``).
"""

from __future__ import annotations

from typing import Optional

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
    if arr.size < 4:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr)
