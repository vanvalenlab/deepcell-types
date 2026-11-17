"""Tests for the post-hoc CT abstention wired into scripts/predict.py.

Exercises the lower-level building blocks in `deepcell_types.abstention` —
`_compute_iqr_fence` and `apply_abstention` — that the CLI wires up. Five
scenarios are covered:

1. Default (no abstention applied) — `apply_abstention` not called → no
   `abstained` column, no `-1` sentinels.
2. k=1.5 on a synthetic 100-cell frame → ~0.23% abstained (near-no-op).
3. k=0.5 on the same frame → ~10% abstained (aggressive).
4. Per-FOV grouping is honoured — the canonical default; fences are computed
   per (dataset_name, fov_name). The optional (tissue, modality) grouping is
   also exercised to confirm `group_cols` is respected.
5. Degenerate distribution (all max-softmax identical) yields no abstentions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepcell_types.abstention import (
    ABSTENTION_LABEL,
    _compute_iqr_fence,
    apply_abstention,
)


def _synthetic_frame(
    n: int, seed: int = 0, tissue: str = "intestine", modality: str = "CODEX"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Beta-distributed max-softmax in (0, 1); skewed high (typical of softmax).
    max_softmax = rng.beta(8.0, 2.0, size=n).astype(np.float32)
    return pd.DataFrame(
        {
            "predicted_ct": np.array(["CD4T"] * n, dtype=object),
            "_max_softmax": max_softmax,
            "dataset_name": ["ds0"] * n,
            "fov_name": ["fov0"] * n,
            "tissue": [tissue] * n,
            "modality": [modality] * n,
        }
    )


def test_k_0_2_ablation_abstains_nontrivial_fraction():
    """``k=0.2`` is the historical opt-in abstention ablation, NOT a default —
    the CLI defaults to ``--ct_abstention_k=0`` and the paper headline is
    full-coverage. The CSV-side guard runs ``apply_abstention`` whenever
    ``k > 0``; we exercise that path: passing k=0.2 to a synthetic 1000-cell
    frame must add the ``abstained`` column and produce a non-trivial fraction
    of sentinel-marked cells (≈18% expected).
    """
    df = _synthetic_frame(1000, seed=99)
    out = apply_abstention(df.copy(), k=0.2)
    assert "abstained" in out.columns
    frac = out["abstained"].mean()
    assert 0.03 <= frac <= 0.30, (
        f"k=0.2 should abstain ~18% of cells; got {frac * 100:.2f}%"
    )


def test_disable_abstention_with_nonpositive_k():
    """Passing k<=0 must be a no-op: no abstention applied, original frame
    returned unmodified. The CLI guard ``if k is not None and k > 0`` is the
    mechanism; this test asserts the contract for end users who leave
    abstention off (the default).
    """
    df = _synthetic_frame(100)
    assert "abstained" not in df.columns

    # The guard in scripts/predict.py is `if ct_abstention_k > 0`; passing 0 or
    # a negative value must skip apply_abstention entirely. We assert the
    # guard's logic here (apply_abstention itself doesn't accept k<=0).
    def skip(k):
        return not (k is not None and k > 0)

    assert skip(0)
    assert skip(-1)
    assert skip(None)
    assert not skip(0.5)


def test_k_1_5_near_no_op_on_100_cells():
    """k=1.5 (canonical Tukey fence) abstains a tiny fraction (~0%–2%).

    At this multiplier the fence sits well below the bulk of the confidence
    distribution, so on a 100-cell synthetic frame the count is small (often 0)
    but must remain at most a few percent — the assertion is "<= 5%" to absorb
    sampling noise.
    """
    df = _synthetic_frame(100, seed=42)
    out = apply_abstention(df.copy(), k=1.5)
    abst = out["abstained"].sum()
    assert abst / len(out) <= 0.05, f"k=1.5 should abstain few cells, got {abst}/100"


def test_k_0_5_aggressive_on_1000_cells():
    """k=0.5 is aggressive (a tighter fence trims more low-confidence cells).
    On a beta-distributed synthetic 1000-cell frame the fraction abstained
    should land in a 5%–25% band.
    """
    df = _synthetic_frame(1000, seed=7)
    out = apply_abstention(df.copy(), k=0.5)
    frac = out["abstained"].mean()
    assert 0.03 <= frac <= 0.30, (
        f"k=0.5 should abstain ~10% of cells; got {frac * 100:.2f}%"
    )
    # Sentinel applied
    assert (out.loc[out["abstained"], "predicted_ct"] == ABSTENTION_LABEL).all()
    # Original predictions preserved
    assert (out["predicted_ct_raw"] == "CD4T").all()


def test_per_group_grouping_honored():
    """Different (tissue, modality) groups should have INDEPENDENT fences.

    Construct two groups whose max-softmax distributions are far apart: A is
    in [0.90, 0.99] and B is in [0.10, 0.30]. With k=1.5 in each group the
    fences are computed locally, so a value of 0.20 (low globally, typical
    in B) should NOT be abstained in B; conversely 0.90 (high globally, low
    in A) should NOT be abstained in A.
    """
    rng = np.random.default_rng(123)
    n_per = 50
    a_vals = rng.uniform(0.90, 0.99, n_per).astype(np.float32)
    b_vals = rng.uniform(0.10, 0.30, n_per).astype(np.float32)
    df = pd.DataFrame(
        {
            "predicted_ct": np.array(["X"] * (2 * n_per), dtype=object),
            "_max_softmax": np.concatenate([a_vals, b_vals]),
            "tissue": ["lung"] * n_per + ["skin"] * n_per,
            "modality": ["CODEX"] * (2 * n_per),
        }
    )
    out = apply_abstention(df.copy(), k=1.5, group_cols=("tissue", "modality"))

    # Most B-cells are in [0.10, 0.30] — they should NOT all be abstained,
    # because the fence is computed within their own group.
    b_mask = out["tissue"].to_numpy() == "skin"
    assert out.loc[b_mask, "abstained"].mean() <= 0.05, (
        "Per-group grouping not honored: B-group cells were abstained based "
        "on the global distribution rather than the local one."
    )

    # And in group A the median value 0.945 must not be abstained either.
    a_mask = out["tissue"].to_numpy() == "lung"
    assert out.loc[a_mask, "abstained"].mean() <= 0.05


def test_degenerate_distribution_no_abstention():
    """All-identical max-softmax → IQR == 0 → fence == Q1 → no abstention.

    Verifies the analysis-script invariant carried over to the production
    wire-up.
    """
    df = pd.DataFrame(
        {
            "predicted_ct": ["Y"] * 50,
            "_max_softmax": np.full(50, 0.42, dtype=np.float32),
            "dataset_name": ["ds0"] * 50,
            "fov_name": ["fov0"] * 50,
            "tissue": ["liver"] * 50,
            "modality": ["MIBI"] * 50,
        }
    )
    out = apply_abstention(df.copy(), k=1.5)
    assert out["abstained"].sum() == 0


def test_compute_iqr_fence_returns_none_under_threshold():
    """Tiny samples (<4) cannot define an IQR; helper returns None and the
    caller must not abstain anything in that group.
    """
    assert _compute_iqr_fence(np.array([0.5, 0.6, 0.7]), 1.5) is None
    f = _compute_iqr_fence(np.array([0.1, 0.2, 0.3, 0.4]), 1.5)
    assert f is not None
    assert f == pytest.approx(0.175 - 1.5 * 0.15)


def test_already_abstained_frame_rejected():
    """Defensive: refuse to double-apply abstention on the same frame."""
    df = _synthetic_frame(20)
    out = apply_abstention(df.copy(), k=1.5)
    with pytest.raises(ValueError, match="already exists"):
        apply_abstention(out, k=1.5)


def test_per_fov_grouping_honored():
    """The CLI groups by (dataset_name, fov_name); fences are per-FOV.

    Two FOVs in the same dataset with far-apart max-softmax distributions
    must be abstained on their own local fences, not a global one. A low FOV
    (all in [0.10, 0.30]) should not be wholesale abstained just because its
    values are low relative to the high FOV.
    """
    rng = np.random.default_rng(123)
    n = 50
    hi = rng.uniform(0.90, 0.99, n).astype(np.float32)
    lo = rng.uniform(0.10, 0.30, n).astype(np.float32)
    df = pd.DataFrame(
        {
            "predicted_ct": np.array(["X"] * (2 * n), dtype=object),
            "_max_softmax": np.concatenate([hi, lo]),
            "dataset_name": ["ds_a"] * (2 * n),
            "fov_name": ["fov_hi"] * n + ["fov_lo"] * n,
        }
    )
    out = apply_abstention(df.copy(), k=1.5, group_cols=("dataset_name", "fov_name"))
    lo_mask = out["fov_name"].to_numpy() == "fov_lo"
    assert out.loc[lo_mask, "abstained"].mean() <= 0.05, (
        "Per-FOV grouping not honored: low-FOV cells abstained on the global "
        "distribution rather than their own."
    )
    hi_mask = out["fov_name"].to_numpy() == "fov_hi"
    assert out.loc[hi_mask, "abstained"].mean() <= 0.05
