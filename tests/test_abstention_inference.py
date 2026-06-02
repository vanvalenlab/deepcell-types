"""Direct coverage for the inference-side IQR abstention module.

``deepcell_types.abstention`` is the numpy-only kernel that ``predict()`` uses.
``deepcell_types.abstention`` is also where the batched ``apply_abstention``
lives, so the CLI tests exercise the same kernel; these tests pin the inference
module directly (including the non-finite-input guard) and run in the
inference-only CI job (no ``[train]`` dependency).
"""

import numpy as np

from deepcell_types.abstention import ABSTENTION_LABEL, compute_iqr_fence


def test_returns_none_below_four_values():
    assert compute_iqr_fence(np.array([0.4, 0.6, 0.9]), 1.5) is None
    assert compute_iqr_fence(np.array([]), 0.2) is None


def test_known_fence_value():
    vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    assert compute_iqr_fence(vals, 0.2) == float(q1 - 0.2 * (q3 - q1))


def test_k_zero_gives_q1():
    vals = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_iqr_fence(vals, 0.0) == float(np.quantile(vals, 0.25))


def test_degenerate_iqr_returns_q1():
    # All-equal confidences => IQR == 0 => fence == Q1, regardless of k.
    assert compute_iqr_fence(np.array([0.5, 0.5, 0.5, 0.5]), 1.5) == 0.5


def test_non_finite_values_are_dropped():
    finite = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    with_bad = np.array([0.1, 0.3, np.nan, 0.5, 0.7, np.inf, 0.9])
    # Dropping NaN/inf must reproduce the finite-only fence, not poison it to NaN.
    assert compute_iqr_fence(with_bad, 0.2) == compute_iqr_fence(finite, 0.2)


def test_all_non_finite_returns_none():
    assert compute_iqr_fence(np.array([np.nan, np.inf, -np.inf, np.nan]), 0.2) is None


def test_abstention_label_sentinel():
    assert ABSTENTION_LABEL == "Unknown"
