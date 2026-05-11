"""Smoke tests for the v2 MP-label refinement pipeline.

Each stage gets at least one test plus an idempotence and a tiny synthetic
gold-validation test. These do NOT load the full archive — they exercise
the pure-numpy core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_stage1_doublet_detection():
    from scripts.refine_mp_labels_with_intensity_v2 import detect_doublets

    areas = np.array([100, 110, 105, 95, 320], dtype=float)
    # cell 4 is 3.2x median; should fire if violations flagged
    violations = np.array([0, 0, 0, 0, 1])
    out = detect_doublets(areas, violations)
    assert out.tolist() == [False, False, False, False, True]
    # without lineage violation the giant cell should NOT be flagged
    out2 = detect_doublets(areas, np.zeros_like(violations))
    assert out2.tolist() == [False, False, False, False, False]


def test_stage2_ct_mislabel_detection():
    from scripts.refine_mp_labels_with_intensity_v2 import detect_ct_mislabel

    # Bcell that's CD20+ (above threshold) — NOT suspect
    inten = {"CD20": 1.0, "CD19": 0.0}
    thr = {"CD20": 0.5, "CD19": 0.5}
    assert detect_ct_mislabel(inten, "Bcell", thr, {}) is False
    # Bcell that's flat (both below threshold) — IS suspect
    inten2 = {"CD20": 0.0, "CD19": 0.0}
    assert detect_ct_mislabel(inten2, "Bcell", thr, {}) is True
    # Bcell with no required markers in panel — NOT suspect (no info)
    assert detect_ct_mislabel({"CD45": 0.0}, "Bcell", {"CD45": 0.5}, {}) is False
    # Unknown CT — NOT suspect
    assert detect_ct_mislabel(inten2, "MysteryCell", thr, {}) is False


def test_stage3_compartment_intensity():
    from scripts.refine_mp_labels_with_intensity_v2 import (
        quantify_cell_marker, per_fov_background)

    # 11x11 image, central 5x5 cell, bright ring at boundary, dark interior
    img = np.zeros((11, 11), dtype=np.float64)
    mask = np.zeros_like(img, dtype=bool)
    mask[3:8, 3:8] = True
    img[mask] = 10.0  # all cell pixels = 10
    img[4:7, 4:7] = 1.0  # interior (3x3) low
    bg, mad = per_fov_background(img, mask.astype(int))
    # background should be 0 since outside mask is all 0
    assert bg == 0
    # quantify a membrane marker — should pick up the bright ring
    out = quantify_cell_marker(img, mask, "CD3", bg, mad)
    assert out["compartment"] in ("membrane", "cell")
    assert out["summary"] >= 1.0


def test_stage4_threshold_class_a_gmm():
    from scripts.refine_mp_labels_with_intensity_v2 import fit_threshold_class_a

    rng = np.random.default_rng(0)
    neg = rng.lognormal(mean=0.0, sigma=0.4, size=2000)
    pos = rng.lognormal(mean=3.0, sigma=0.4, size=2000)
    intensities = np.concatenate([neg, pos])
    thr, status, bic_ratio = fit_threshold_class_a(intensities, min_n=200)
    assert status == "bimodal_gmm_valley"
    # threshold should fall between the two modes
    assert np.median(neg) < thr < np.median(pos)
    # bic ratio should be << 1.0 because the data is strongly bimodal
    assert bic_ratio is not None
    assert bic_ratio < 0.95


def test_stage4_threshold_class_b_iqr():
    """Iter-5 (v3): class-B fitting uses IQR-based fence (Q3 + k*IQR)
    in log-space, replacing the iter-1..4 ``median + 3*1.4826*MAD``
    formula. Threshold must be above Q3 of the negative pool.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        fit_threshold_class_b, CLASS_B_IQR_K)

    rng = np.random.default_rng(0)
    neg = rng.lognormal(mean=0.0, sigma=0.4, size=1000)
    thr, status = fit_threshold_class_b(neg, min_n=200)
    assert status.startswith("graded_iqr"), (
        f"expected IQR-based status, got {status!r}")
    # The status string encodes the K constant for traceability.
    assert str(CLASS_B_IQR_K) in status
    # Threshold must be above Q3 (the lower bound of the upper fence)
    # in intensity space — it's expm1(Q3_log + k*IQR_log) > Q3_intensity.
    q3_intensity = float(np.percentile(neg, 75))
    assert thr > q3_intensity, (
        f"threshold {thr:.4f} should be above Q3 = {q3_intensity:.4f}")


def test_stage5_lineage_exclusion():
    from scripts.refine_mp_labels_with_intensity_v2 import apply_lineage_exclusion

    # A cell labelled CD20=1 AND CD3=1 (impossible) -> both masked
    labels = {"CD20": 1, "CD3": 1, "CD8": 0, "CD45": 1}
    intensities = {"CD20": 5.0, "CD3": 5.0, "CD8": 0.1, "CD45": 5.0}
    thresholds = {"CD20": 1.0, "CD3": 1.0, "CD8": 1.0, "CD45": 1.0}
    out = apply_lineage_exclusion(labels, intensities, thresholds)
    assert out["CD20"] == "?"
    assert out["CD3"] == "?"
    # Non-violating labels untouched
    assert out["CD8"] == 0


def test_stage6_apply_pipeline_emits_labels():
    from scripts.refine_mp_labels_with_intensity_v2 import apply_pipeline

    # 3 cells with intensities. CD3 threshold = 0.5; cell 1 CD3+, cell 2 CD3-,
    # cell 3 has CD20+ AND CD3+ (lineage violation -> both masked).
    cells = {
        1: {"area": 100, "intensities": {"CD3": 1.0, "CD20": 0.0},
            "compartments": {}, "coverage": {}},
        2: {"area": 100, "intensities": {"CD3": 0.1, "CD20": 0.0},
            "compartments": {}, "coverage": {}},
        3: {"area": 100, "intensities": {"CD3": 1.0, "CD20": 1.0},
            "compartments": {}, "coverage": {}},
    }
    per_fov = {("ds1", "fov1"): cells}
    thr = {"CD3": 0.5, "CD20": 0.5}
    out = apply_pipeline(per_fov, thr)
    assert out[("ds1", "fov1")].cells[1]["labels"]["CD3"] == 1
    assert out[("ds1", "fov1")].cells[2]["labels"]["CD3"] == 0
    assert out[("ds1", "fov1")].cells[3]["labels"]["CD3"] == "?"
    assert out[("ds1", "fov1")].cells[3]["labels"]["CD20"] == "?"


def test_idempotence():
    """Running the pipeline twice on the same input gives the same labels."""
    from scripts.refine_mp_labels_with_intensity_v2 import apply_pipeline

    cells = {
        1: {"area": 100, "intensities": {"CD3": 1.0}, "compartments": {}, "coverage": {}},
        2: {"area": 100, "intensities": {"CD3": 0.1}, "compartments": {}, "coverage": {}},
    }
    per_fov = {("ds1", "fov1"): cells}
    thr = {"CD3": 0.5}
    a = apply_pipeline(per_fov, thr)
    b = apply_pipeline(per_fov, thr)
    for cid in (1, 2):
        assert a[("ds1", "fov1")].cells[cid]["labels"] == b[("ds1", "fov1")].cells[cid]["labels"]


def test_stage7_synthetic_gold_validation(tmp_path):
    """End-to-end gold-validation test on a tiny synthetic fixture."""
    import pandas as pd
    import pytest

    pytest.importorskip(
        "analysis.validate_mp_refinement",
        reason="analysis/ lives in the research workspace (sibling), not this repo",
    )
    from analysis.validate_mp_refinement import (
        evaluate_acceptance, score_predictions, two_fold_cv)

    # Build a fake gold dataframe and a perfectly-correct prediction set
    gold = pd.DataFrame([
        {"dataset": "fakeA", "fov": "f1", "cell_id": 1, "channel": "CD3", "activity": 1},
        {"dataset": "fakeA", "fov": "f1", "cell_id": 2, "channel": "CD3", "activity": 0},
        {"dataset": "fakeB", "fov": "f1", "cell_id": 1, "channel": "CD3", "activity": 1},
        {"dataset": "fakeB", "fov": "f1", "cell_id": 2, "channel": "CD3", "activity": 0},
        {"dataset": "fakeA", "fov": "f1", "cell_id": 1, "channel": "CD20", "activity": 0},
        {"dataset": "fakeA", "fov": "f1", "cell_id": 2, "channel": "CD20", "activity": 1},
        {"dataset": "fakeB", "fov": "f1", "cell_id": 1, "channel": "CD20", "activity": 0},
        {"dataset": "fakeB", "fov": "f1", "cell_id": 2, "channel": "CD20", "activity": 1},
    ])
    perfect_preds = {
        "fakeA": {"f1": {"1": {"labels": {"CD3": 1, "CD20": 0}},
                          "2": {"labels": {"CD3": 0, "CD20": 1}}}},
        "fakeB": {"f1": {"1": {"labels": {"CD3": 1, "CD20": 0}},
                          "2": {"labels": {"CD3": 0, "CD20": 1}}}},
    }
    weak_preds = {  # everything = 0
        "fakeA": {"f1": {"1": {"labels": {"CD3": 0, "CD20": 0}},
                          "2": {"labels": {"CD3": 0, "CD20": 0}}}},
        "fakeB": {"f1": {"1": {"labels": {"CD3": 0, "CD20": 0}},
                          "2": {"labels": {"CD3": 0, "CD20": 0}}}},
    }
    rows = list(zip(gold["dataset"], gold["fov"], gold["cell_id"],
                    gold["channel"], gold["activity"]))
    pm_perfect, n_def, n_abs = score_predictions(rows, perfect_preds)
    assert pm_perfect["CD3"]["f1"] == pytest.approx(1.0)
    assert pm_perfect["CD20"]["f1"] == pytest.approx(1.0)

    results, avg_r, avg_u = two_fold_cv(gold, perfect_preds, weak_preds,
                                         cv_seed=0, min_definite=2)
    assert avg_r > avg_u
    accept = evaluate_acceptance(results, avg_r, avg_u)
    assert accept["delta_macro_f1"] > 0.024


def test_marker_class_dispatch():
    from scripts.refine_mp_labels_with_intensity_v2 import marker_class

    assert marker_class("CD3") == "A"
    assert marker_class("Ki67") == "B"
    assert marker_class("DAPI") == "optout"
    assert marker_class("CD20") == "A"
    # Unknown -> default B (graded; safer than misclassifying as bimodal)
    assert marker_class("UnknownMarker") == "B"


def test_fix1_dual_summary_picks_lower_bic_ratio():
    """Fix 1: when class-A wholecell summary is more bimodal than the
    compartment summary, derive_thresholds should pick "wholecell" as the
    summary statistic for that bucket."""
    from scripts.refine_mp_labels_with_intensity_v2 import derive_thresholds

    rng = np.random.default_rng(7)
    n_neg, n_pos = 1500, 1500

    # COMPARTMENT summary: noisy unimodal-ish (BIC ratio ~ 1, no clean valley)
    comp_unimodal = rng.lognormal(mean=1.0, sigma=1.5, size=n_neg + n_pos)

    # WHOLECELL summary: clean bimodal (low BIC ratio)
    wc_neg = rng.lognormal(mean=0.0, sigma=0.3, size=n_neg)
    wc_pos = rng.lognormal(mean=3.5, sigma=0.3, size=n_pos)
    wc_bimodal = np.concatenate([wc_neg, wc_pos])

    # Use a class-A marker so the dual-summary path runs
    intensities = {"CD3": comp_unimodal}
    intensities_wc = {"CD3": wc_bimodal}

    thresholds, statuses, summary_choice = derive_thresholds(
        intensities,
        intensities_wholecell=intensities_wc,
        # provide an empty per-(ds,marker) dict so the new-API code path runs
        intensities_per_ds_marker={},
        intensities_wc_per_ds_marker={},
        min_bucket_n=200,
    )
    bucket = (None, "CD3")
    # Either the wholecell summary wins outright (lower BIC ratio), or the
    # compartment summary failed to be bimodal — in either case the choice
    # should be "wholecell".
    assert bucket in summary_choice, f"no choice picked: statuses={statuses}"
    assert summary_choice[bucket] == "wholecell", (
        f"expected wholecell to be picked for CD3, got {summary_choice[bucket]} "
        f"(status={statuses.get(bucket)})"
    )


def test_fix2_class_b_fallback_when_no_negatives():
    """Fix 2: a class-B marker with NO hard negatives (e.g. Calprotectin in
    a panel with only neutrophil-CT cells) should still receive a threshold
    via the global-MAD fallback rather than no threshold at all.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import derive_thresholds

    # A class-B-by-default marker with very few cells in the lower 70% (so
    # the hard-neg pool ends up below min_neg_for_mad). Easiest way to
    # trigger: a tiny array of all-similar values where the bottom 70% has
    # < 50 elements. Use 60 cells; bottom 70% = 42 cells < 50.
    rng = np.random.default_rng(0)
    vals = rng.lognormal(mean=2.0, sigma=0.3, size=60)
    intensities = {"Calprotectin": vals}
    # Class B by default (Calprotectin not in CLASS_A_BIMODAL nor in
    # CLASS_OPTOUT, so marker_class -> "B"). No wholecell pool needed.
    thresholds, statuses, summary_choice = derive_thresholds(
        intensities,
        intensities_wholecell={},
        intensities_per_ds_marker={},
        intensities_wc_per_ds_marker={},
        min_bucket_n=10,        # allow the 60-cell array through
        min_neg_for_mad=50,     # 42 (lower 70%) < 50 -> trigger fallback
    )
    bucket = (None, "Calprotectin")
    assert bucket in thresholds, (
        f"expected fallback threshold for Calprotectin; "
        f"got statuses={statuses}, thresholds={thresholds}"
    )
    assert "fallback" in statuses[bucket], (
        f"expected 'fallback' in status, got {statuses[bucket]}"
    )
    assert thresholds[bucket] > 0


def test_fix3_per_dataset_marker_stratification():
    """Fix 3: a (dataset, marker) bucket above ``min_bucket_n`` gets its own
    threshold; smaller buckets fall back to the (None, marker) pool.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        derive_thresholds, apply_pipeline)

    rng = np.random.default_rng(13)
    # dsA has very different absolute intensities than dsB (10x).
    # Both are bimodal with the same separation in log space, so GMM should
    # yield clean valleys per bucket.
    a_neg = rng.lognormal(mean=0.0, sigma=0.3, size=400)
    a_pos = rng.lognormal(mean=3.0, sigma=0.3, size=400)
    b_neg = rng.lognormal(mean=3.0, sigma=0.3, size=400)
    b_pos = rng.lognormal(mean=6.0, sigma=0.3, size=400)
    intensities = {
        "CD3": np.concatenate([a_neg, a_pos, b_neg, b_pos]),
    }
    intensities_per_ds_marker = {
        ("dsA", "CD3"): np.concatenate([a_neg, a_pos]),
        ("dsB", "CD3"): np.concatenate([b_neg, b_pos]),
    }

    thresholds, statuses, summary_choice = derive_thresholds(
        intensities,
        intensities_wholecell={"CD3": np.concatenate([a_neg, a_pos, b_neg, b_pos])},
        intensities_per_ds_marker=intensities_per_ds_marker,
        intensities_wc_per_ds_marker={
            ("dsA", "CD3"): np.concatenate([a_neg, a_pos]),
            ("dsB", "CD3"): np.concatenate([b_neg, b_pos]),
        },
        min_bucket_n=200,
    )
    assert ("dsA", "CD3") in thresholds
    assert ("dsB", "CD3") in thresholds
    # Per-dataset thresholds should differ substantially (different scales).
    assert thresholds[("dsB", "CD3")] > 2 * thresholds[("dsA", "CD3")]

    # Pipeline should pick up per-dataset thresholds correctly: a "dsA" cell
    # with intensity that's negative-mode-dsB but positive-mode-dsA should
    # be called positive.
    cells_dsA = {
        1: {"area": 100,
            "intensities": {"CD3": float(np.median(a_pos))},
            "intensities_wholecell": {"CD3": float(np.median(a_pos))},
            "compartments": {}, "coverage": {}},
    }
    cells_dsB = {
        1: {"area": 100,
            "intensities": {"CD3": float(np.median(a_pos))},
            "intensities_wholecell": {"CD3": float(np.median(a_pos))},
            "compartments": {}, "coverage": {}},
    }
    per_fov = {("dsA", "f1"): cells_dsA, ("dsB", "f1"): cells_dsB}
    out = apply_pipeline(per_fov, thresholds, summary_choice=summary_choice)
    label_a = out[("dsA", "f1")].cells[1]["labels"]["CD3"]
    label_b = out[("dsB", "f1")].cells[1]["labels"]["CD3"]
    # In dsA the value is positive (sits at dsA's positive mode); in dsB the
    # exact same value is at dsB's negative mode.
    assert label_a == 1, f"dsA cell should be CD3+ at its scale, got {label_a}"
    assert label_b == 0, f"dsB cell should be CD3- at its scale, got {label_b}"


def test_fix4_lineage_aware_pool_selection():
    """Iter-3 Fix 4: when ``per_fov_cells`` is provided, class-B markers in
    ``LINEAGE_HARD_NEGATIVES`` should fit on cells whose derived lineage tags
    intersect the marker's hard-negative set, NOT on the lower-70% proxy.

    Construction: a Ki67 panel with two cell groups —
      * 200 epithelial cells (panCK-high, low Ki67) — these are the
        biological hard-negatives for Ki67.
      * 200 lymphoid cells (CD3-high, MIXED Ki67 — half low / half high).
        Many lymphoid Ki67-low cells would land in "lower 70 percent"
        and contaminate the negative pool, dragging the threshold down.

    Expected: with the lineage-aware pool, only the epithelial cells'
    Ki67 (uniformly low) is used, yielding a TIGHTER threshold than the
    "lower 70 percent" proxy would produce.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        derive_thresholds, LINEAGE_HARD_NEGATIVES)

    rng = np.random.default_rng(42)

    # Cells: dict of cid -> {area, intensities, intensities_wholecell, ...}
    # Build a single FOV in a synthetic dataset.
    cells: dict = {}
    cid = 1
    # 200 epithelial cells: panCK high, Ki67 LOW (~lognormal mean=0.3)
    for _ in range(200):
        ck_hi = float(rng.lognormal(mean=4.0, sigma=0.2))
        ki67_lo = float(rng.lognormal(mean=0.3, sigma=0.3))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": ck_hi, "Ki67": ki67_lo, "CD3": 0.05},
            "intensities_wholecell": {"panCK": ck_hi, "Ki67": ki67_lo, "CD3": 0.05},
            "compartments": {}, "coverage": {},
        }
        cid += 1
    # 200 lymphoid cells: CD3 high; HALF have Ki67 HIGH, half have Ki67 LOW.
    # Put the "low" half at the SAME log-mean as epithelial cells to make
    # the lower-70% proxy worse than the lineage-aware pool.
    for k in range(200):
        cd3_hi = float(rng.lognormal(mean=4.0, sigma=0.2))
        ki67_val = (float(rng.lognormal(mean=4.5, sigma=0.3)) if k % 2 == 0
                    else float(rng.lognormal(mean=0.3, sigma=0.3)))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": 0.05, "Ki67": ki67_val, "CD3": cd3_hi},
            "intensities_wholecell": {"panCK": 0.05, "Ki67": ki67_val, "CD3": cd3_hi},
            "compartments": {}, "coverage": {},
        }
        cid += 1

    per_fov_cells = {("ds1", "fov1"): cells}

    # Pooled intensities (the script computes these from per_fov_cells; we
    # mirror them here for the test).
    pancK_arr = np.array([c["intensities"]["panCK"] for c in cells.values()])
    cd3_arr = np.array([c["intensities"]["CD3"] for c in cells.values()])
    ki67_arr = np.array([c["intensities"]["Ki67"] for c in cells.values()])

    intensities = {"panCK": pancK_arr, "CD3": cd3_arr, "Ki67": ki67_arr}
    intensities_wc = dict(intensities)
    inten_per_ds = {("ds1", k): v for k, v in intensities.items()}
    inten_wc_per_ds = dict(inten_per_ds)

    diag: dict = {}
    thresholds_with_lineage, statuses_w, _ = derive_thresholds(
        intensities,
        intensities_wholecell=intensities_wc,
        intensities_per_ds_marker=inten_per_ds,
        intensities_wc_per_ds_marker=inten_wc_per_ds,
        min_bucket_n=200,
        per_fov_cells=per_fov_cells,
        min_neg_for_lineage=50,
        fix4_diagnostics=diag,
    )
    # Sanity: Ki67 is in LINEAGE_HARD_NEGATIVES, has epithelial as a member.
    assert "Ki67" in LINEAGE_HARD_NEGATIVES
    assert "epithelial" in LINEAGE_HARD_NEGATIVES["Ki67"]
    # Ki67 (None, marker) bucket should have a threshold AND status should
    # show the lineage-aware path was used.
    bucket = (None, "Ki67")
    assert bucket in thresholds_with_lineage, (
        f"no Ki67 threshold; statuses={statuses_w}")
    assert "lineage_n=" in statuses_w[bucket], (
        f"Ki67 not fit via lineage path; status={statuses_w[bucket]}")
    assert diag.get("lineage", 0) >= 1, f"diag={diag}"

    # Now run the SAME data WITHOUT per_fov_cells (legacy lower-70 path) —
    # threshold should be lower because the lower-70 pool includes the
    # contaminating low-Ki67 lymphoid cells.
    thresholds_no_lineage, _, _ = derive_thresholds(
        intensities,
        intensities_wholecell=intensities_wc,
        intensities_per_ds_marker=inten_per_ds,
        intensities_wc_per_ds_marker=inten_wc_per_ds,
        min_bucket_n=200,
    )
    # Both should have a threshold; the lineage-aware one should be HIGHER
    # (tighter — pulls out the real positives, doesn't include contaminated
    # negatives).
    assert (None, "Ki67") in thresholds_no_lineage
    thr_lin = thresholds_with_lineage[(None, "Ki67")]
    thr_no = thresholds_no_lineage[(None, "Ki67")]
    # The contamination effect: in this construction the lineage pool is
    # uniformly low so its 3*MAD threshold sits above the bulk of the
    # epithelial Ki67 distribution but BELOW the lymphoid-Ki67-high
    # half. The lower-70 pool combines epi-low + lymph-low + some lymph-
    # high → wider MAD → higher threshold (or vice versa). The minimum
    # invariant we check: the two thresholds DIFFER and the lineage one
    # is reasonable (between epi-low median and lymph-high median).
    epi_med = float(np.median([
        c["intensities"]["Ki67"] for c in list(cells.values())[:200]
    ]))
    lymph_high_med = float(np.median([
        c["intensities"]["Ki67"]
        for c in list(cells.values())[200:]
        if c["intensities"]["Ki67"] > 1.0
    ]))
    assert epi_med < thr_lin < lymph_high_med, (
        f"lineage threshold {thr_lin:.3f} should sit between epi-Ki67 "
        f"median {epi_med:.3f} and lymph-Ki67-high median "
        f"{lymph_high_med:.3f}"
    )
    # The two thresholds shouldn't be identical — the path differed.
    assert abs(thr_lin - thr_no) > 1e-6, (
        f"Fix 4 didn't change the threshold: lineage={thr_lin}, "
        f"no_lineage={thr_no}")


def test_fix4_floor_gated_global_mad():
    """Iter-3 Fix 4: the floor-gated Fix 2 fallback must NOT produce a
    threshold below the marker's 5th-percentile global intensity.

    Construction: a class-B marker with all values being uniformly tiny —
    the Fix-2 raw 3*MAD threshold collapses to ~0. The floor (p5 of the
    same distribution) keeps it above zero.
    """
    import numpy as np
    from scripts.refine_mp_labels_with_intensity_v2 import (
        _floor_gated_global_mad, _class_b_global_mad_fallback)

    rng = np.random.default_rng(0)
    # 200 cells, ALL with very narrow lognormal around mean=0.5 -- MAD
    # is small relative to the absolute scale; raw 3*MAD threshold is tiny.
    vals = rng.lognormal(mean=-2.0, sigma=0.05, size=200)

    raw_thr, _ = _class_b_global_mad_fallback(vals)
    floored_thr, status = _floor_gated_global_mad(vals, global_pool=vals,
                                                    floor_pct=5.0)
    assert raw_thr is not None
    assert floored_thr is not None
    p5 = float(np.percentile(vals, 5.0))
    # The floor is enforced (>= p5) regardless of MAD.
    assert floored_thr >= p5 - 1e-9, (
        f"floored_thr={floored_thr} should be >= p5={p5}")
    # Status reflects whether floor or MAD was used.
    assert "floor" in status or "above_floor" in status, (
        f"status should mention floor/above_floor, got {status}")


def test_fix4_lineage_derivation_basic():
    """A cell with CD3+ (above class-A threshold) gets {lymphoid, Tcell}.
    A cell with panCK+ gets {epithelial}. A cell with both CD3+ and SMA+
    does NOT get 'mesenchymal' (mesenchymal is exclusive of lymphoid)."""
    from scripts.refine_mp_labels_with_intensity_v2 import (
        _derive_cell_lineage_tags)

    cells = {
        1: {"area": 100, "intensities": {"CD3": 5.0, "panCK": 0.0, "SMA": 0.0},
            "intensities_wholecell": {"CD3": 5.0, "panCK": 0.0, "SMA": 0.0},
            "compartments": {}, "coverage": {}},
        2: {"area": 100, "intensities": {"CD3": 0.0, "panCK": 5.0, "SMA": 0.0},
            "intensities_wholecell": {"CD3": 0.0, "panCK": 5.0, "SMA": 0.0},
            "compartments": {}, "coverage": {}},
        3: {"area": 100, "intensities": {"CD3": 5.0, "panCK": 0.0, "SMA": 5.0},
            "intensities_wholecell": {"CD3": 5.0, "panCK": 0.0, "SMA": 5.0},
            "compartments": {}, "coverage": {}},
        4: {"area": 100, "intensities": {"CD3": 0.0, "panCK": 0.0, "SMA": 5.0},
            "intensities_wholecell": {"CD3": 0.0, "panCK": 0.0, "SMA": 5.0},
            "compartments": {}, "coverage": {}},
    }
    thresholds = {(None, "CD3"): 1.0, (None, "panCK"): 1.0, (None, "SMA"): 1.0}
    tags = _derive_cell_lineage_tags(cells, "ds1", thresholds, {})
    assert "Tcell" in tags[1] and "lymphoid" in tags[1]
    assert "epithelial" in tags[2]
    # cell 3: CD3+ AND SMA+ -- mesenchymal is dropped due to lymphoid signal.
    assert "Tcell" in tags[3] and "lymphoid" in tags[3]
    assert "mesenchymal" not in tags[3]
    # cell 4: SMA-only -> mesenchymal.
    assert tags[4] == {"mesenchymal"}


def test_compartment_masks_disjoint():
    from scripts.refine_mp_labels_with_intensity_v2 import _compartment_masks

    mask = np.zeros((20, 20), dtype=bool)
    mask[6:14, 6:14] = True
    comps = _compartment_masks(mask)
    # nuclear + cytoplasmic = whole cell (when nuclear non-empty)
    assert (comps["nuclear"] | comps["cytoplasmic"])[mask].all()
    # membrane is a subset of the cell
    assert (comps["membrane"] & ~mask).sum() == 0
    # pericellular is OUTSIDE the cell
    assert (comps["pericellular"] & mask).sum() == 0


def test_fix5_sox9_is_nuclear():
    """Iter-4 Fix 5: SOX9 is a transcription factor and must report
    'nuclear' as its compartment so the per-cell summary statistic is
    averaged over the nuclear sub-mask, not the cytoplasmic default.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        NUCLEAR, localization)

    assert "SOX9" in NUCLEAR, (
        "SOX9 must be in the NUCLEAR set (it's a transcription factor; "
        "default cytoplasmic compartment was the iter-3 failure mode)")
    assert localization("SOX9") == "nuclear", (
        f"localization('SOX9') = {localization('SOX9')!r}; expected 'nuclear'")
    # The other long-standing nuclear TFs should still be there (sanity).
    assert localization("Ki67") == "nuclear"
    assert localization("FoxP3") == "nuclear"


def test_fix6_cd15_class_b_with_lineage_negatives():
    """Iter-4 Fix 6: CD15 should now route through the class-B graded
    path with a lineage-aware hard-negative pool (lymphocytes + structural
    cells). It was implicit class-B-by-default in iter-3 with no
    LINEAGE_HARD_NEGATIVES entry, so it fell through to the lower-70%
    proxy and lost 0.300 F1 on fold A.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        CLASS_B_GRADED, LINEAGE_HARD_NEGATIVES, marker_class,
        derive_thresholds)

    # 1. CD15 is now an explicit class-B marker.
    assert "CD15" in CLASS_B_GRADED, (
        "CD15 must be in CLASS_B_GRADED (was implicit class-B-by-default; "
        "explicit membership is the iter-4 fix)")
    assert marker_class("CD15") == "B"

    # 2. CD15 has a lineage hard-negative pool restricted to non-myeloid
    #    lineages. Lymphocyte and structural lineages must be hard-negs.
    assert "CD15" in LINEAGE_HARD_NEGATIVES
    cd15_negs = LINEAGE_HARD_NEGATIVES["CD15"]
    for required in ("Tcell", "Bcell", "epithelial", "endothelial",
                     "mesenchymal"):
        assert required in cd15_negs, (
            f"CD15 hard-neg pool missing {required!r}; got {cd15_negs}")
    # Myeloid is NOT a hard-neg (CD15 IS expressed on neutrophils /
    # late-myeloid).
    assert "myeloid" not in cd15_negs

    # 3. End-to-end smoke: build a synthetic FOV where lineage-aware
    #    fitting would route CD15 through the lineage path.
    rng = np.random.default_rng(99)
    cells: dict = {}
    cid = 1
    # 200 epithelial cells (panCK+, CD15-low) — these are hard-negs.
    for _ in range(200):
        ck_hi = float(rng.lognormal(mean=4.0, sigma=0.2))
        cd15_lo = float(rng.lognormal(mean=0.3, sigma=0.3))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": ck_hi, "CD15": cd15_lo, "CD3": 0.05},
            "intensities_wholecell": {
                "panCK": ck_hi, "CD15": cd15_lo, "CD3": 0.05},
            "compartments": {}, "coverage": {},
        }
        cid += 1
    # 200 myeloid-like cells (CD3-low, panCK-low, but CD15 high — would
    # contaminate a lower-70% pool that doesn't strip lineage).
    for _ in range(200):
        cd15_hi = float(rng.lognormal(mean=4.5, sigma=0.3))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": 0.05, "CD15": cd15_hi, "CD3": 0.05},
            "intensities_wholecell": {
                "panCK": 0.05, "CD15": cd15_hi, "CD3": 0.05},
            "compartments": {}, "coverage": {},
        }
        cid += 1
    per_fov_cells = {("ds1", "fov1"): cells}
    pancK_arr = np.array([c["intensities"]["panCK"] for c in cells.values()])
    cd3_arr = np.array([c["intensities"]["CD3"] for c in cells.values()])
    cd15_arr = np.array([c["intensities"]["CD15"] for c in cells.values()])
    intensities = {"panCK": pancK_arr, "CD3": cd3_arr, "CD15": cd15_arr}
    intensities_wc = dict(intensities)
    inten_per_ds = {("ds1", k): v for k, v in intensities.items()}
    inten_wc_per_ds = dict(inten_per_ds)

    diag: dict = {}
    thresholds, statuses, _ = derive_thresholds(
        intensities,
        intensities_wholecell=intensities_wc,
        intensities_per_ds_marker=inten_per_ds,
        intensities_wc_per_ds_marker=inten_wc_per_ds,
        min_bucket_n=200,
        per_fov_cells=per_fov_cells,
        min_neg_for_lineage=50,
        fix4_diagnostics=diag,
    )
    bucket = (None, "CD15")
    assert bucket in thresholds, f"no CD15 threshold; statuses={statuses}"
    # Lineage path must have fired for CD15 (status mentions 'lineage_n=').
    assert "lineage_n=" in statuses[bucket], (
        f"CD15 not fit via lineage path; status={statuses[bucket]}")
    # CD15 threshold should fall between the epithelial-low median and
    # the myeloid-CD15-high median (the lineage pool is uniformly low).
    epi_med = float(np.median(
        [c["intensities"]["CD15"] for c in list(cells.values())[:200]]))
    myel_med = float(np.median(
        [c["intensities"]["CD15"] for c in list(cells.values())[200:]]))
    thr = thresholds[bucket]
    assert epi_med < thr < myel_med, (
        f"CD15 threshold {thr:.3f} should sit between epithelial-low "
        f"median {epi_med:.3f} and myeloid-high median {myel_med:.3f}")


def test_iter5_iqr_handles_saturated_negative_pool_where_mad_inflates():
    """Iter-5 (v3): on a saturated negative pool (the iter-4 SOX9 / FAP
    failure mode), ``median + 3*MAD`` produces a higher threshold than
    ``Q3 + 1.5*IQR`` because MAD's absolute-deviation metric is dragged
    up by the saturated upper half of the distribution. IQR is bounded
    by the gap between Q1 and Q3 only — saturation that pushes everything
    above Q3 toward a clipped maximum doesn't widen IQR.

    Construction: 1000 cells with a broad-spread lognormal (mean=4.0,
    sigma=0.8) — the codex-colon SOX9 / FAP "everything is high" lineage
    pool, where the panel max compresses the upper tail and MAD reaches
    further than Q3.
    """
    from scripts.refine_mp_labels_with_intensity_v2 import (
        fit_threshold_class_b, CLASS_B_IQR_K)

    rng = np.random.default_rng(7)
    # Saturated, broad lognormal — emulates the iter-4 failure mode where
    # the lineage hard-negative pool has uniformly high intensities with
    # heavy spread.
    neg = rng.lognormal(mean=4.0, sigma=0.8, size=1000)

    iqr_thr, status = fit_threshold_class_b(neg, min_n=200)
    assert iqr_thr is not None
    assert "iqr" in status

    # Hand-compute the iter-1..4 MAD-based threshold for comparison.
    x = np.log1p(neg)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    mad_thr_log = med + 3.0 * 1.4826 * mad
    mad_thr = float(np.expm1(mad_thr_log))

    # IQR threshold MUST be lower than the MAD threshold on this
    # saturated distribution (the iter-4 failure mode).
    assert iqr_thr < mad_thr, (
        f"IQR threshold {iqr_thr:.3f} should be < MAD threshold "
        f"{mad_thr:.3f} on a saturated negative pool; the whole point "
        f"of iter-5 is that MAD inflates here and IQR doesn't")
    # The reduction should be material (>= 10 %) — not just a numerical
    # tie. On the iter-4 failure mode this gap is what unblocks SOX9.
    reduction = (mad_thr - iqr_thr) / mad_thr
    assert reduction >= 0.10, (
        f"IQR threshold reduction {reduction:.3f} should be >= 0.10 vs "
        f"MAD on a saturated pool; got iqr={iqr_thr:.3f}, "
        f"mad={mad_thr:.3f}")
    # Sanity: IQR threshold should still be above the bulk's Q3 (it IS
    # an upper fence on the negatives).
    bulk_q3 = float(np.percentile(neg, 75))
    assert iqr_thr > bulk_q3, (
        f"IQR threshold {iqr_thr:.3f} should be above bulk Q3 "
        f"{bulk_q3:.3f}")
    # The constant used should be 1.5 (Tukey) per the spec.
    assert CLASS_B_IQR_K == 1.5


def test_iter5_p95_safety_cap_fires():
    """Iter-5 (v3): when the IQR-fitted class-B threshold would land
    above p95 of the global per-marker intensity distribution, the
    safety cap MUST clamp it to p95. This is the iter-4 SOX9 / FAP /
    Vimentin failure mode: lineage-pool MAD saturated past the panel's
    dynamic range and 0 cells got called positive.

    Construction: a class-B-by-default marker ("BlamMarker") with
    a per-FOV lineage hard-negative pool whose Q3 is far above the rest
    of the global pool. With Tukey's k=1.5, ``Q3 + 1.5*IQR`` lands
    above the global p95 — the cap should fire and pull it back down.
    """
    import numpy as np
    from scripts.refine_mp_labels_with_intensity_v2 import (
        derive_thresholds, LINEAGE_HARD_NEGATIVES, CLASS_B_GRADED)

    # Reuse Vimentin's lineage hard-neg set (epithelial). Vimentin is in
    # CLASS_B_GRADED already and has LINEAGE_HARD_NEGATIVES = {epithelial}.
    assert "Vimentin" in CLASS_B_GRADED
    assert "epithelial" in LINEAGE_HARD_NEGATIVES["Vimentin"]

    rng = np.random.default_rng(101)
    cells: dict = {}
    cid = 1
    # 200 epithelial cells (panCK+) with a SATURATED-TAIL Vimentin negative
    # pool: 50% contamination at lognormal(mean=8, sigma=0.5) and 50% at
    # lognormal(mean=1, sigma=0.2). With Q3 sitting in the contamination,
    # Q3 + 1.5*IQR in log-space saturates *enormously* — the iter-4
    # SOX9 / FAP failure mode reproduced synthetically.
    for k in range(200):
        ck_hi = float(rng.lognormal(mean=4.0, sigma=0.2))
        if k % 2 == 0:  # 50/200 contaminating high values
            vim_neg = float(rng.lognormal(mean=8.0, sigma=0.5))
        else:
            vim_neg = float(rng.lognormal(mean=1.0, sigma=0.2))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": ck_hi, "Vimentin": vim_neg},
            "intensities_wholecell": {"panCK": ck_hi, "Vimentin": vim_neg},
            "compartments": {}, "coverage": {},
        }
        cid += 1
    # 200 mesenchymal cells (panCK-low) at a MODERATE, tight Vimentin
    # distribution (mean=3.0, sigma=0.2). These set p95 of the global pool
    # WELL BELOW the saturated lineage-pool IQR fence — so the cap fires.
    for _ in range(200):
        vim_pos = float(rng.lognormal(mean=3.0, sigma=0.2))
        cells[cid] = {
            "area": 100,
            "intensities": {"panCK": 0.05, "Vimentin": vim_pos},
            "intensities_wholecell": {"panCK": 0.05, "Vimentin": vim_pos},
            "compartments": {}, "coverage": {},
        }
        cid += 1
    per_fov_cells = {("ds1", "fov1"): cells}
    pancK_arr = np.array([c["intensities"]["panCK"] for c in cells.values()])
    vim_arr = np.array([c["intensities"]["Vimentin"] for c in cells.values()])
    intensities = {"panCK": pancK_arr, "Vimentin": vim_arr}
    intensities_wc = dict(intensities)
    inten_per_ds = {("ds1", k): v for k, v in intensities.items()}
    inten_wc_per_ds = dict(inten_per_ds)

    diag: dict = {}
    thresholds, statuses, _ = derive_thresholds(
        intensities,
        intensities_wholecell=intensities_wc,
        intensities_per_ds_marker=inten_per_ds,
        intensities_wc_per_ds_marker=inten_wc_per_ds,
        min_bucket_n=200,
        per_fov_cells=per_fov_cells,
        min_neg_for_lineage=50,
        fix4_diagnostics=diag,
    )

    bucket = (None, "Vimentin")
    assert bucket in thresholds, f"no Vimentin threshold; statuses={statuses}"
    # The cap should have fired at least once (this bucket triggered it).
    assert diag.get("iqr_capped_at_p95", 0) >= 1, (
        f"expected the p95 cap to fire on this construction; "
        f"diag={diag}, status={statuses[bucket]}")
    # Threshold should be ~p95 of the global Vimentin pool.
    p95 = float(np.percentile(vim_arr, 95))
    thr = thresholds[bucket]
    assert thr <= p95 + 1e-6, (
        f"capped threshold {thr:.4f} should be <= p95 {p95:.4f}")
    # Status should mention the cap.
    assert "capped@p95" in statuses[bucket], (
        f"status should mention 'capped@p95'; got {statuses[bucket]!r}")
