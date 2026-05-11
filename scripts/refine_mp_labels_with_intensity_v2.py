"""V2 of MP-label refinement — 7-stage biology-first pipeline.

Supersedes ``scripts/refine_mp_labels_with_intensity.py`` (PR #35). The key
shift in v2 is that **the unit of evaluation is the per-cell-marker positivity
prediction**, derived from compartment-aware intensity quantification and
gated by lineage/co-expression rules. The "rulebook" only enters as a CT-prior
for stages 1 and 2 (doublet / CT-mislabel detection).

Pipeline (per the four biology-first proposals on
``proposal/bio-quant``, ``proposal/bio-thresholds``, ``proposal/bio-context``,
``proposal/bio-validation``):

  1. Doublet pre-filter             (cheap; area + lineage-co-positivity)
  2. CT-mislabel pre-filter         (cheap; required-marker check)
  3. Compartment-aware intensity    (nuclear / membrane / cytoplasmic /
                                     pericellular hard sub-masks; 75th-pct in
                                     the compartment; per-FOV background
                                     subtraction = median(mask==0))
  4. Marker-class threshold fit     (Class A: GMM on log1p(I-bg) per marker;
                                     Class B: MAD-based on hard-negative pool;
                                     Class C/D: opt-out — emit "?")
  5. Lineage-exclusion rules        (~6 rules from bio-context proposal)
  6. Apply refinement → emit        (per-cell-marker label: 0 / 1 / "?")
  7. Gold-validation gate           (separate script:
                                     ``analysis/validate_mp_refinement.py``)

For the MVP, stages 1, 2, 5, 6 are fully implemented; stage 3 uses the
simple per-FOV median background (no REDSEA spillover); stage 4 covers
classes A and B only (C/D are opt-out). Stage 7 is the validator script.

Post-MVP fixes (2026-04-27, this commit):
  * **Fix 1 — Dual-summary GMM for class-A markers.** Stage 3 now emits both
    a compartment-restricted 75th-pct summary (legacy) AND a whole-cell mean
    (new). Stage 4 fits a 2-component GMM on each and keeps whichever
    summary statistic yields the lower BIC ratio (BIC2 / BIC1 — lower means
    more bimodal). The chosen summary is persisted per-bucket and used at
    scoring time. This unlocks markers like CD3 / CD45 whose membrane
    sub-mask is too noisy for clean GMM bimodality.
  * **Fix 2 — Class-B fallback when no negatives are available.** Some
    markers (e.g. Calprotectin) have no cells in their hard-negative pool
    when restricted by (dataset, marker). When the hard-neg pool has
    fewer than ``min_neg_for_mad`` cells (default 50), Stage 4 falls back
    to ``3 * 1.4826 * MAD-of-marker-across-all-cells`` and emits a
    warning. (Iter-3: this fallback is now floor-gated by the marker's
    5th-percentile global intensity to stop FAP-style threshold→0
    failures.)
  * **Fix 3 — Per-(dataset, marker) stratification with (marker) fallback.**
    Stage 4 now fits class-A GMMs and class-B MAD thresholds per
    (dataset, marker) bucket. If a bucket has fewer than ``min_bucket_n``
    cells (default 200), the pipeline falls back to the pooled
    (None, marker) bucket at scoring time.

Iter-3 fix (2026-04-27):
  * **Fix 4 — Lineage-aware hard-negative pool for class-B threshold
    fitting.** Replaces the "lower 70 percent of intensities" proxy
    negative pool, which is contaminated for graded-state markers (cells
    that biologically might be expressing the marker but at low absolute
    intensity get pulled into the negative pool, dragging the threshold
    down). Instead, after class-A thresholds are fit we derive a coarse
    per-cell lineage tag (lymphoid / Tcell / Bcell / myeloid / epithelial
    / endothelial / mesenchymal) from the class-A signals (CD3, CD20,
    CD45, CD68, panCK, EpCAM, ECadherin, CD31, SMA), then build the
    class-B negative pool from cells whose lineage is in
    ``LINEAGE_HARD_NEGATIVES[marker]`` — biologically-defended cell types
    that should never (or almost never) express the given graded marker.
    Falls back to the prior "lower 70 percent" pool when the lineage pool
    has fewer than ``min_neg_for_lineage`` cells (default 50), and
    further to the floor-gated Fix 2 global MAD when even that is too
    small. Floor gate: threshold cannot drop below the 5th percentile of
    the global per-marker intensity distribution. Also moves FAP out of
    ``CLASS_OPTOUT`` into class-B (now a fibroblast-specific scored
    marker rather than a structural opt-out).

Future work (deferred to v3):
  * Spatial-neighborhood prior (Agent C; lymphocyte-clustering check).
  * Iterative trimming of high-tail "negatives" before refit (Agent B).
  * REDSEA spillover correction at archive-prep (Agent A).
  * Calibration via gold (rather than only validation against gold).

Usage::

    DATA_DIR=/data/xwang3/tissuenet-caitlin-labels.zarr \
    uv run python -m scripts.refine_mp_labels_with_intensity_v2 \
        --gold_dir data/gold_standard/gold_standard_labelled \
        --rulebook output/mp_matrix_assignments.json \
        --output output/mp_refined_v2.json \
        --max_cells 5000 --dry_run

This script is gold-aware: it walks the per-FOV TIFF + mask layout under
``--gold_dir`` and emits per-(dataset, fov, cell_id, channel) predictions.
The companion script ``analysis/validate_mp_refinement.py`` joins those
predictions to ``gold_standard_groundtruth.csv`` and reports macro F1.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import click
import numpy as np


# ---------------------------------------------------------------------------
# Iter-5 (v3) — Class-B threshold fitting statistic.
#
# CLASS_B_IQR_K is the multiplier on the inter-quartile range (IQR) used to
# build a robust upper-fence on the class-B hard-negative pool:
#
#     threshold (log-space) = Q3(neg) + CLASS_B_IQR_K * IQR(neg)
#
# This replaces iter-1..4's ``median + 3 * 1.4826 * MAD``. The MAD-based
# formula was inflated by autofluorescence in graded-state markers (SOX9,
# FAP, Vimentin): the lineage hard-negative pool includes mid-expressers
# whose long upper tail blew the MAD past the panel's dynamic range, so
# the threshold landed above every cell's intensity and zero positives
# were called.
#
# We use Tukey's "outer fence" multiplier of 1.5 (the convention for the
# upper IQR-fence outlier rule). For a pure normal distribution,
# ``Q3 + 1.5 * IQR`` is roughly equivalent to ``median + 2 * sigma``;
# ``median + 3 * MAD`` is closer to ``median + 2 * sigma`` as well, so
# the sensitivity is comparable on clean data — but IQR ignores the
# distribution's tails, making it MUCH less sensitive than MAD on the
# heavy-tail hard-negative pools we actually see for graded markers.
#
# The choice of 1.5 (rather than the stricter 3.0 of the "far-outlier
# fence") is conservative biologically: graded-state markers like Ki67
# / FoxP3 are NOT cleanly bimodal and the supposed "negative" pool
# already contains some mid-expressers, so over-tightening the threshold
# would call those mid-cells positive. 1.5 keeps the call somewhat
# permissive — appropriate when the negative pool is itself slightly
# contaminated.
# ---------------------------------------------------------------------------
CLASS_B_IQR_K: float = 1.5

# Iter-5: safety cap percentile. After IQR-based fitting, the threshold is
# capped at the marker's p95 across the (None, marker) global pool. This
# stops iter-4 SOX9 / FAP / Vimentin failure modes where the negative pool's
# upper fence saturated above the panel's dynamic range and 0 cells were
# called positive.
CLASS_B_CAP_PCT: float = 95.0

# Iter-3: floor percentile (kept). After fitting, the threshold cannot fall
# below the marker's p05 across the global pool — stops FAP-style "threshold
# collapses to ~0" when the pool's IQR is tiny.
CLASS_B_FLOOR_PCT: float = 5.0


# ---------------------------------------------------------------------------
# Marker-class registries (Agents A and B; canonical short list — extend
# via config/marker_threshold_classes.yaml in v3).
# ---------------------------------------------------------------------------
CLASS_A_BIMODAL: set = {
    "CD3", "CD3e", "CD3d", "CD4", "CD8", "CD8a",
    "CD20", "CD19", "CD45", "CD68", "CD56",
    "CD11c", "CD11b", "EpCAM", "CK", "PanCK", "panCK",
    "ECadherin", "E-cadherin", "CD31", "SMA", "aSMA",
    "CD138", "CD79a",
}
CLASS_B_GRADED: set = {
    "Ki67", "FoxP3", "Foxp3", "PD1", "PD-1", "PDL1", "PD-L1",
    "CD25", "CD45RO", "CD45RA", "GranzymeB", "GZMB",
    "HLA-DR", "HLA-Class-2", "HLADR", "CD163", "CD163-DT",
    "CD206", "Vimentin", "Tox", "TCF1", "ICOS", "CD40",
    "CD40-L", "TIM3", "TIGIT", "Galectin9", "iNOS",
    "CD69",
    # Iter-3 (Fix 4): SOX9 is epithelial/progenitor-restricted; FAP is
    # fibroblast-specific. Both were either implicit class-B (SOX9; default
    # fallback) or wrongly OPTOUT (FAP). Explicit class-B gives them the
    # lineage-aware-pool path.
    "SOX9", "FAP",
    # Iter-4 (Fix 6): CD15 (SSEA-1 / Lewis-X) is a granulocyte / late-myeloid
    # marker whose expression is graded across maturation states (not a
    # crisp bimodal-lineage call). It was previously implicit class-B-by-
    # default, but with no LINEAGE_HARD_NEGATIVES entry it fell through to
    # the "lower 70 percent" proxy and lost 0.300 F1 in iter-3 (the largest
    # remaining max-marker-loss in the gate). Explicit class-B + lineage
    # pool entry below recovers it.
    "CD15",
}
# Class C / D: opt-out — never call positive
CLASS_OPTOUT: set = {
    "DAPI", "Hoechst", "DRAQ5",
    "Collagen", "Collagen-I", "Fibronectin", "LYVE1",
    "Na-K-ATPase", "H3", "H3K27me3", "H3K9ac",
    "aDefensin5",  # secreted enteric defensin; structural
    "Lumican",
}

# Cytoplasmic / membrane localization (for compartment selection)
# Iter-4 Fix 5: SOX9 is a transcription factor and lives in the nucleus.
# It was previously falling through to the default "cytoplasmic" compartment,
# so the per-cell summary statistic averaged over the wrong sub-mask and the
# fitted threshold landed in noise (SOX9 lost ~0.27 macro F1 in iter-3).
NUCLEAR = {"Ki67", "FoxP3", "Foxp3", "DAPI", "Hoechst", "PCNA", "p53",
           "Sox10", "TBX21", "GATA3", "H3", "DRAQ5", "SOX9"}
MEMBRANE = {
    "CD3", "CD3e", "CD3d", "CD4", "CD8", "CD8a", "CD20", "CD19",
    "CD45", "CD31", "CD56", "CD11c", "CD11b", "ECadherin", "E-cadherin",
    "EpCAM", "PD1", "PD-1", "PDL1", "PD-L1", "HLA-DR", "HLA-Class-2",
    "HLADR", "CD16", "CD138", "CD25", "CD68",
}
SECRETED = {"Collagen", "Collagen-I", "FAP", "Fibronectin", "aDefensin5"}


# ---------------------------------------------------------------------------
# Iter-3 Fix 4: lineage-aware hard-negative pools for class-B threshold fitting
#
# Each entry maps a class-B marker to a set of coarse lineage tags whose
# cells should be hard-negative (i.e. biologically should never or almost
# never express this marker). Lineage tags are derived from class-A markers
# (CD3, CD20, CD45, CD68, panCK, EpCAM, ECadherin, CD31, SMA) AFTER class-A
# thresholds are fit; see ``_derive_cell_lineage_tags``.
#
# Rules cited briefly:
#   * Ki67 — proliferation marker; structural / terminally-differentiated
#     CTs (epithelial, endothelial, mesenchymal) have <2% Ki67+ baseline.
#   * FoxP3 — Treg-defining transcription factor; everything except Treg
#     (which is a sub-set of CD3+ Tcell) is a hard-negative.
#   * CD163 / CD206 — myeloid / M2-macrophage markers; lymphocytes and
#     structural cells are hard-negatives.
#   * HLA-DR — restricted to professional APCs (B/dendritic/myeloid) plus
#     a small fraction of activated T; structural lineages are hard-neg.
#   * PD1 / PD-L1 — PD1 is T/B-restricted on lymphocytes; PD-L1 is
#     inducible across many CTs but quiescent lymphocytes are
#     conservative hard-negatives.
#   * Vimentin — mesenchymal IF marker; epithelial cells are vimentin-low.
#   * SOX9 — epithelial progenitor / chondrogenic; lymphocytes /
#     myeloid / structural are hard-neg.
#   * FAP — fibroblast activation protein; lymphocytes / endothelial /
#     epithelial are hard-neg.
#   * GranzymeB — cytotoxic T / NK only.
#   * CD45RO / CD45RA — lymphoid memory / naive isoforms.
#   * CD25 / CD40 / CD40-L / CD69 / ICOS / TIM3 / TIGIT / Tox / TCF1 —
#     activation / exhaustion markers, lineage-restricted to lymphoid.
#
# Lineage tags used (derived from class-A markers in
# ``_derive_cell_lineage_tags``):
#   * lymphoid    — CD3+ OR CD20+ OR CD45+ (any lymphocyte signal)
#   * Tcell       — CD3+ (subset of lymphoid)
#   * Bcell       — CD20+ (subset of lymphoid)
#   * myeloid     — CD68+ OR CD11c+ (subset of CD45+ but distinct in pool)
#   * epithelial  — panCK+ OR EpCAM+ OR ECadherin+
#   * endothelial — CD31+
#   * mesenchymal — SMA+ (and not lymphoid / epithelial)
#
# Fallback: any class-B marker not in this dict drops to the prior
# "lower 70 percent of intensities" proxy negative pool (legacy behaviour).
LINEAGE_HARD_NEGATIVES: Dict[str, set] = {
    # Proliferation — structural CTs
    "Ki67":      {"epithelial", "endothelial", "mesenchymal"},
    # Treg-specific TF — everything except Tcell
    "FoxP3":     {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    "Foxp3":     {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    # Myeloid / M2 markers
    "CD163":     {"Tcell", "Bcell", "epithelial", "endothelial", "mesenchymal"},
    "CD163-DT":  {"Tcell", "Bcell", "epithelial", "endothelial", "mesenchymal"},
    "CD206":     {"Tcell", "Bcell", "epithelial", "endothelial", "mesenchymal"},
    # Professional-APC marker
    "HLA-DR":    {"epithelial", "endothelial", "mesenchymal"},
    "HLADR":     {"epithelial", "endothelial", "mesenchymal"},
    "HLA-Class-2": {"epithelial", "endothelial", "mesenchymal"},
    # T-cell exhaustion / checkpoint (T- and partly B-restricted)
    "PD1":       {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    "PD-1":      {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    # PDL1 inducible on many CTs; conservative neg = quiescent lymphocytes
    "PDL1":      {"Tcell", "Bcell"},
    "PD-L1":     {"Tcell", "Bcell"},
    # Memory T-cell isoform; non-lymphoid is hard-neg
    "CD45RO":    {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    "CD45RA":    {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    # Cytotoxic granule
    "GranzymeB": {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    "GZMB":      {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    # Mesenchymal IF marker — epithelial-low
    "Vimentin":  {"epithelial"},
    # Epithelial / progenitor TF
    "SOX9":      {"Tcell", "Bcell", "myeloid", "endothelial", "mesenchymal"},
    # Fibroblast activation protein — fibroblasts only
    "FAP":       {"Tcell", "Bcell", "myeloid", "endothelial", "epithelial"},
    # IL-2Ralpha — Treg / activated T; non-lymphoid hard-neg
    "CD25":      {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    # Exhaustion / costimulatory — lymphoid-restricted
    "TIM3":      {"epithelial", "endothelial", "mesenchymal", "Bcell"},
    "TIGIT":     {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    "ICOS":      {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    "Tox":       {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    "TCF1":      {"epithelial", "endothelial", "mesenchymal", "myeloid"},
    # Costimulatory — APC / T
    "CD40":      {"epithelial", "endothelial", "mesenchymal"},
    "CD40-L":    {"epithelial", "endothelial", "mesenchymal", "myeloid", "Bcell"},
    # Activation — lymphoid
    "CD69":      {"epithelial", "endothelial", "mesenchymal"},
    # Galectin-9 — APC / tumor; conservative
    "Galectin9": {"Tcell", "Bcell"},
    # iNOS — myeloid effector
    "iNOS":      {"Tcell", "Bcell", "epithelial", "endothelial"},
    # Iter-4 Fix 6: CD15 (SSEA-1 / Lewis-X) — granulocyte / late-myeloid
    # marker. Restricted to the myeloid (esp. neutrophil) lineage, so
    # hard-negatives = lymphocytes + structural / parenchymal cells.
    # Lineage tags available: Tcell / Bcell / epithelial / endothelial /
    # mesenchymal (the lymphoid-via-CD45 set is implicitly covered by
    # Tcell + Bcell at the lineage-tag granularity).
    "CD15":      {"Tcell", "Bcell", "epithelial", "endothelial", "mesenchymal"},
}


# Iter-3 Fix 4: per-lineage marker tests (which class-A markers determine
# membership in each lineage tag).
LINEAGE_DEFINITIONS: Dict[str, List[str]] = {
    "lymphoid":    ["CD3", "CD3e", "CD3d", "CD20", "CD19", "CD45", "CD138", "CD79a"],
    "Tcell":       ["CD3", "CD3e", "CD3d"],
    "Bcell":       ["CD20", "CD19", "CD138", "CD79a"],
    "myeloid":     ["CD68", "CD11c", "CD11b"],
    "epithelial":  ["panCK", "PanCK", "CK", "EpCAM", "ECadherin", "E-cadherin"],
    "endothelial": ["CD31"],
    "mesenchymal": ["SMA", "aSMA"],
}


def marker_class(name: str) -> str:
    """Return one of {'A','B','optout'} for a marker name."""
    if name in CLASS_OPTOUT:
        return "optout"
    if name in CLASS_A_BIMODAL:
        return "A"
    if name in CLASS_B_GRADED:
        return "B"
    # Default fallback: B (graded) is the safest because a true bimodal-lineage
    # marker mistakenly treated as graded yields a slightly conservative
    # threshold; the reverse flips quiescent cells.
    return "B"


def localization(name: str) -> str:
    """Return one of {'nuclear','membrane','cytoplasmic','pericellular'}."""
    if name in NUCLEAR:
        return "nuclear"
    if name in MEMBRANE:
        return "membrane"
    if name in SECRETED:
        return "pericellular"
    return "cytoplasmic"


# ---------------------------------------------------------------------------
# Lineage-exclusion rules (Stage 5 — Agent C, top 6 highest-confidence)
#
# Each rule: (markerA, markerB) — biologically impossible to be positive on
# both simultaneously in steady-state tissue.
# ---------------------------------------------------------------------------
LINEAGE_EXCLUSIONS: List[Tuple[str, str]] = [
    ("CD20", "CD3"),     # B vs T lineage
    ("CD68", "CD3"),     # myeloid vs lymphoid
    ("PanCK", "CD45"),   # epithelial vs hematopoietic (also CK and panCK)
    ("CK", "CD45"),
    ("panCK", "CD45"),
    ("CD4", "CD8"),      # mature αβ T (single-positive)
    ("CD4", "CD8a"),
    ("CD56", "CD3"),     # NK vs conventional T (NKT exception ignored at MVP)
    ("Tryptase", "CD3"), # mast vs T
]


# Lineage co-positivity (Stage 2 — CT-mislabel detection): cell labelled with
# a CT requires at least one of these markers to be positive. Maps standard CT
# names to a list of "any-of" defining markers.
LINEAGE_COPOS: Dict[str, List[str]] = {
    "Bcell": ["CD20", "CD19", "CD79a"],
    "Tcell": ["CD3", "CD3e", "CD3d"],
    "CD4T": ["CD3", "CD3e", "CD4"],
    "CD8T": ["CD3", "CD3e", "CD8", "CD8a"],
    "Treg": ["CD3", "CD3e", "FoxP3", "Foxp3"],
    "NKT": ["CD3", "CD3e", "CD56"],
    "NK": ["CD56", "NKp46"],
    "Macrophage": ["CD68", "CD163", "CD14"],
    "Monocyte": ["CD14", "CD16"],
    "Endothelial": ["CD31", "CD34"],
    "BloodVesselEndothelial": ["CD31", "CD34"],
    "LymphaticEndothelial": ["Podoplanin", "LYVE1", "CD31"],
    "Fibroblast": ["SMA", "aSMA", "Vimentin"],
    "SmoothMuscle": ["SMA", "aSMA", "Desmin"],
    "Tumor": ["CK", "PanCK", "panCK", "EpCAM"],
    "Epithelial": ["CK", "PanCK", "panCK", "EpCAM", "ECadherin", "E-cadherin"],
    "Plasma": ["CD138"],
    "Dendritic": ["CD11c", "HLA-DR", "HLADR"],
    "Mast": ["Tryptase", "c-kit", "CD117"],
    "Neutrophil": ["MPO", "CD15", "CD66b"],
}


# ---------------------------------------------------------------------------
# Stage 3: per-cell compartment-aware intensity
# ---------------------------------------------------------------------------

def _compartment_masks(self_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Return hard sub-masks for nuclear / membrane / cytoplasmic / pericell.

    Approximations (documented; sufficient for v2 MVP):
      * **nuclear** = cell-mask pixels at high distance-from-boundary
        (top 30 percent by distance_transform_edt). This is a proxy when an
        explicit nucleus channel/mask is unavailable.
      * **membrane** = cell-mask minus 1-px erosion (the boundary ring).
      * **cytoplasmic** = cell-mask minus nuclear sub-mask.
      * **pericellular** = 2-px dilation minus cell-mask (ECM neighbourhood).
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    m = self_mask.astype(bool)
    if not m.any():
        return {"nuclear": m, "membrane": m, "cytoplasmic": m, "pericellular": m}

    interior = distance_transform_edt(m)
    if interior.max() > 0:
        nuc_thr = np.percentile(interior[m], 70.0)
        nuclear = m & (interior >= nuc_thr)
    else:
        nuclear = m
    eroded = binary_erosion(m, iterations=1)
    membrane = m & ~eroded
    cytoplasmic = m & ~nuclear
    if cytoplasmic.sum() < 4:
        cytoplasmic = m
    pericellular = binary_dilation(m, iterations=2) & ~m
    return {
        "nuclear": nuclear, "membrane": membrane,
        "cytoplasmic": cytoplasmic, "pericellular": pericellular,
    }


def quantify_cell_marker(
    image: np.ndarray, cell_mask: np.ndarray,
    marker: str, fov_bg: float, fov_mad: float,
    compartments: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Per-(cell, marker) quantification.

    Returns dict with keys:
      * summary               : legacy 75th-pct of bg-subtracted pixels in the
                                marker's preferred compartment (membrane for
                                membrane markers, etc.)
      * summary_compartment   : the compartment-restricted 75th-pct (same as
                                ``summary`` — kept under explicit name)
      * summary_wholecell     : bg-subtracted mean across the whole cell
                                (no compartment restriction). Used by Fix 1 as
                                the alternative class-A summary statistic.
      * coverage              : fraction of compartment pixels > 3*MAD over bg
      * coherence             : 1 - CV
      * compartment           : the compartment name actually used
    """
    nan_out = {
        "summary": float("nan"),
        "summary_compartment": float("nan"),
        "summary_wholecell": float("nan"),
        "coverage": float("nan"),
        "coherence": float("nan"),
        "compartment": "skip",
    }
    if cell_mask.sum() < 9:
        return nan_out
    img = np.clip(image.astype(np.float64) - fov_bg, 0.0, None)
    if compartments is None:
        compartments = _compartment_masks(cell_mask)
    comp_name = localization(marker)
    if comp_name == "nuclear":
        comp = compartments["nuclear"]
    elif comp_name == "membrane":
        comp = compartments["membrane"]
    elif comp_name == "pericellular":
        comp = compartments["pericellular"]
    else:
        comp = compartments["cytoplasmic"]
    if comp.sum() < 4:
        comp = cell_mask.astype(bool)
        comp_name = "cell"
    px = img[comp]
    cell_px = img[cell_mask.astype(bool)]
    if px.size == 0:
        out = dict(nan_out)
        out["compartment"] = comp_name
        return out
    summary_compartment = float(np.percentile(px, 75.0))
    summary_wholecell = float(cell_px.mean()) if cell_px.size else float("nan")
    on = px > max(3.0 * fov_mad, 1.0)
    coverage = float(on.mean())
    mu = float(px.mean())
    sd = float(px.std())
    coherence = max(0.0, min(1.0, 1.0 - (sd / (mu + 1e-6))))
    return {
        "summary": summary_compartment,
        "summary_compartment": summary_compartment,
        "summary_wholecell": summary_wholecell,
        "coverage": coverage,
        "coherence": coherence,
        "compartment": comp_name,
    }


def per_fov_background(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Per-FOV per-channel background floor: median + MAD over non-mask pixels."""
    bg_pixels = image[mask == 0]
    if bg_pixels.size == 0:
        return 0.0, 1.0
    bg = float(np.median(bg_pixels))
    mad = float(np.median(np.abs(bg_pixels - bg))) + 1e-6
    return bg, mad


# ---------------------------------------------------------------------------
# Stage 4: marker-class threshold fit
# ---------------------------------------------------------------------------

def fit_threshold_class_a(intensities: np.ndarray, min_n: int = 200,
                           min_bimodality: float = 0.55,
                           ) -> Tuple[Optional[float], str, Optional[float]]:
    """Class A bimodal-lineage: fit 2-component GMM on log1p(I).

    Returns ``(threshold_in_intensity_space, status, bic_ratio)`` where
    ``bic_ratio = BIC(2-component) / BIC(1-component)`` — lower means more
    bimodal (a 2-component fit explains the data better). When the
    distribution is unimodal or n is too small, returns
    ``(None, reason, bic_ratio_or_None)``."""
    if intensities.size < min_n:
        return None, f"insufficient_n={intensities.size}", None
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None, "no_sklearn", None
    x = np.log1p(np.asarray(intensities, dtype=np.float64)).reshape(-1, 1)
    if not np.isfinite(x).all():
        x = x[np.isfinite(x).flatten()].reshape(-1, 1)
        if x.size < min_n:
            return None, f"insufficient_finite_n={x.size}", None
    try:
        gmm2 = GaussianMixture(n_components=2, random_state=0,
                               max_iter=200).fit(x)
        gmm1 = GaussianMixture(n_components=1, random_state=0,
                               max_iter=200).fit(x)
    except Exception as exc:  # pragma: no cover
        return None, f"gmm_failed:{type(exc).__name__}", None
    bic1 = float(gmm1.bic(x))
    bic2 = float(gmm2.bic(x))
    # Guard against zero / negative BICs (can happen for tiny n with
    # high log-likelihood); treat as ratio = 1.0 (no preference).
    if bic1 <= 0 or bic2 <= 0:
        bic_ratio = 1.0
    else:
        bic_ratio = bic2 / bic1
    if bic1 < bic2:
        return None, "unimodal_no_positive", bic_ratio
    means = sorted(gmm2.means_.flatten().tolist())
    sds = [float(np.sqrt(c)) for c in gmm2.covariances_.flatten()]
    s_lo, s_hi = sorted(sds)
    thr_log = (means[0] * s_hi + means[1] * s_lo) / (s_lo + s_hi)
    return float(np.expm1(thr_log)), "bimodal_gmm_valley", bic_ratio


def fit_threshold_class_b(neg_intensities: np.ndarray,
                           min_n: int = 200) -> Tuple[Optional[float], str]:
    """Class B graded-state: ``Q3 + CLASS_B_IQR_K * IQR`` over negatives.

    Iter-5 (v3) change: replaces the iter-1..4 ``median + 3 * 1.4826 * MAD``
    fitting statistic with Tukey's IQR-based upper fence. MAD on the lineage
    hard-negative pool was getting inflated by autofluorescence in graded
    markers (SOX9, FAP, Vimentin), pushing the threshold above the panel's
    dynamic range and producing 0 positive calls. IQR is much less sensitive
    to long upper tails on the negative pool.

    See module-level docstring for ``CLASS_B_IQR_K`` for the rationale on
    the choice of multiplier.

    Negatives must be hard-negatives (e.g. cells from CTs that don't express
    this marker). Caller is responsible for that selection.
    """
    if neg_intensities.size < min_n:
        return None, f"insufficient_neg_n={neg_intensities.size}"
    x = np.log1p(np.asarray(neg_intensities, dtype=np.float64))
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return None, f"insufficient_finite_neg_n={x.size}"
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    thr_log = q3 + CLASS_B_IQR_K * iqr
    return float(np.expm1(thr_log)), f"graded_iqr_q3+{CLASS_B_IQR_K}iqr"


# ---------------------------------------------------------------------------
# Stages 1, 2, 5: rule-based pre-filters
# ---------------------------------------------------------------------------

def detect_doublets(areas: np.ndarray, lineage_violations: np.ndarray,
                     area_factor: float = 2.5) -> np.ndarray:
    """Doublet rule: area > ``area_factor`` × median(area) AND >= 1 lineage
    violation (both members above threshold).
    """
    if areas.size == 0:
        return np.zeros(0, dtype=bool)
    median_area = float(np.median(areas))
    is_giant = areas > (area_factor * median_area)
    return is_giant & (lineage_violations >= 1)


def detect_ct_mislabel(
    cell_intensities: Dict[str, float],
    cell_type: Optional[str],
    thresholds: Dict[str, float],
    bg: Dict[str, Tuple[float, float]],
) -> bool:
    """If the cell's CT requires any-of {markers} and ALL of those in panel
    are below noise floor, flag CT as suspect.
    """
    if cell_type is None or cell_type not in LINEAGE_COPOS:
        return False
    required = LINEAGE_COPOS[cell_type]
    in_panel = [m for m in required if m in cell_intensities]
    if not in_panel:
        return False
    for m in in_panel:
        thr = thresholds.get(m)
        if thr is None:
            # No threshold => can't assess => assume OK
            return False
        if cell_intensities[m] >= thr:
            return False
    return True


def apply_lineage_exclusion(labels: Dict[str, Any], intensities: Dict[str, float],
                             thresholds: Dict[str, float]) -> Dict[str, Any]:
    """If both members of an exclusion pair are predicted POS, mask both to "?"."""
    out = dict(labels)
    for a, b in LINEAGE_EXCLUSIONS:
        if a not in intensities or b not in intensities:
            continue
        ta = thresholds.get(a)
        tb = thresholds.get(b)
        if ta is None or tb is None:
            continue
        ia = intensities[a]; ib = intensities[b]
        if (np.isfinite(ia) and np.isfinite(ib)
                and ia >= ta and ib >= tb):
            out[a] = "?"
            out[b] = "?"
    return out


# ---------------------------------------------------------------------------
# Gold-data walker (Stage 7 input).
# ---------------------------------------------------------------------------

@dataclass
class FOVPredictions:
    dataset: str
    fov: str
    # cell_id -> {channel -> {label, score, compartment, ct_suspect, doublet}}
    cells: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class V2Stats:
    n_fovs: int = 0
    n_cells: int = 0
    n_doublets: int = 0
    n_ct_suspect: int = 0
    n_pred_pos: int = 0
    n_pred_neg: int = 0
    n_pred_unknown: int = 0
    n_lineage_violations_masked: int = 0
    thresholds_fit: Dict[str, str] = field(default_factory=dict)
    per_marker_n_pos: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_marker_n_neg: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _list_gold_fovs(gold_dir: Path, exclude: Iterable[str] = ()) -> List[Tuple[str, str]]:
    """Return (dataset, fov) tuples found under ``gold_dir``."""
    out: List[Tuple[str, str]] = []
    excl = set(exclude)
    if not gold_dir.exists():
        return out
    for ds_dir in sorted(gold_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in excl:
            continue
        fovs_dir = ds_dir / "fovs"
        if not fovs_dir.exists():
            continue
        for fov_dir in sorted(fovs_dir.iterdir()):
            if fov_dir.is_dir():
                out.append((ds_dir.name, fov_dir.name))
    return out


def _read_fov_channels(gold_dir: Path, dataset: str, fov: str
                        ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Return ({channel_name: image}, mask) for one FOV.

    Robust to the four gold-standard datasets' divergent naming conventions:
      * codex_colon: ``*.ome.tif`` channels, ``{fov}.ome.tif`` mask
      * mibi_breast: ``*.tiff`` channels, ``{fov}.tif`` mask
      * vectra_colon: ``*.ome.tif`` channels, mask name has ``feature_0`` suffix
      * vectra_pancreas: ``*.ome.tif`` channels, mask uses
        ``component_data.ome.tif`` (no ``_image`` suffix vs. fov dir name)
    """
    import tifffile
    fov_dir = gold_dir / dataset / "fovs" / fov
    if not fov_dir.exists():
        return {}, None

    # Channel files: try .ome.tif first, fall back to .tiff
    images: Dict[str, np.ndarray] = {}
    channel_paths = sorted(fov_dir.glob("*.ome.tif")) or sorted(fov_dir.glob("*.tiff"))
    for path in channel_paths:
        name = path.name
        for ext in (".ome.tif", ".tiff", ".tif"):
            if name.endswith(ext):
                ch = name[: -len(ext)]
                break
        else:
            continue
        # Skip "_nuc_exclude" overlay and similar derived layers
        if "_nuc_exclude" in ch:
            continue
        try:
            img = tifffile.imread(str(path))
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            if img.ndim != 2:
                continue
            images[ch] = img
        except Exception:
            continue

    # Mask: try {fov}.ome.tif, {fov}.tif, {fov}feature_0.ome.tif first, then
    # any file in masks/ whose stem starts with the FOV name (or is a prefix
    # of it; vectra_pancreas drops "_image" from the mask filename).
    masks_dir = gold_dir / dataset / "masks"
    mask_path: Optional[Path] = None
    for cand_name in (f"{fov}.ome.tif", f"{fov}.tif",
                       f"{fov}feature_0.ome.tif"):
        cand = masks_dir / cand_name
        if cand.exists():
            mask_path = cand
            break
    if mask_path is None and masks_dir.exists():
        # Fuzzy: prefix match either way (mask file may have extra suffix
        # beyond the fov name, OR fov name may have extra suffix beyond mask)
        for cand in masks_dir.iterdir():
            if not cand.is_file():
                continue
            stem = cand.name
            for ext in (".ome.tif", ".tiff", ".tif"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            if stem.startswith(fov) or fov.startswith(stem):
                mask_path = cand
                break
    mask: Optional[np.ndarray] = None
    if mask_path is not None and mask_path.exists():
        try:
            m = tifffile.imread(str(mask_path))
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            elif m.ndim == 3 and m.shape[-1] == 1:
                m = m[..., 0]
            if m.ndim == 2:
                mask = m
        except Exception:
            mask = None
    return images, mask


def _cell_bbox(mask: np.ndarray, cell_id: int, pad: int = 4
               ) -> Optional[Tuple[int, int, int, int]]:
    """Return (rmin, rmax, cmin, cmax) bounding box for one cell, with padding."""
    ys, xs = np.where(mask == cell_id)
    if ys.size == 0:
        return None
    rmin = max(int(ys.min()) - pad, 0)
    rmax = min(int(ys.max()) + pad + 1, mask.shape[0])
    cmin = max(int(xs.min()) - pad, 0)
    cmax = min(int(xs.max()) + pad + 1, mask.shape[1])
    return rmin, rmax, cmin, cmax


# ---------------------------------------------------------------------------
# Threshold fitting from gold-FOV data (cross-FOV, per-marker)
# ---------------------------------------------------------------------------

def _precompute_bboxes(mask: np.ndarray, pad: int = 4) -> Dict[int, Tuple[int, int, int, int]]:
    """Single-pass per-cell bbox computation using scipy.ndimage.find_objects."""
    from scipy.ndimage import find_objects
    mask_int = mask.astype(np.int32, copy=False)
    n_labels = int(mask_int.max())
    if n_labels < 1:
        return {}
    slices = find_objects(mask_int)
    out: Dict[int, Tuple[int, int, int, int]] = {}
    H, W = mask.shape
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        cid = i + 1
        ys, xs = sl
        rmin = max(ys.start - pad, 0)
        rmax = min(ys.stop + pad, H)
        cmin = max(xs.start - pad, 0)
        cmax = min(xs.stop + pad, W)
        out[cid] = (rmin, rmax, cmin, cmax)
    return out


def collect_intensities_per_marker(
    gold_dir: Path, fov_pairs: Sequence[Tuple[str, str]],
    max_cells_per_fov: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
    Dict[Tuple[str, str], np.ndarray],
    Dict[str, np.ndarray],
    Dict[Tuple[str, str], np.ndarray],
]:
    """Walk a set of FOVs, compute per-(cell, channel) intensity, and return
    five aggregations:

    1. ``inten_per_marker`` (legacy compartment-summary pool, ``{marker: arr}``)
    2. ``per_fov_cells`` — per-cell records carrying both summary stats
       (``intensities`` = compartment summary; ``intensities_wholecell`` =
       whole-cell mean) plus area/compartment/coverage
    3. ``inten_per_ds_marker`` — Fix 3 stratification:
       ``{(dataset, marker): compartment-summary-array}``
    4. ``inten_wholecell_per_marker`` — Fix 1 alt summary pool, ``{marker: arr}``
    5. ``inten_wholecell_per_ds_marker`` — Fix 1 + Fix 3 combined,
       ``{(dataset, marker): wholecell-summary-array}``
    """
    inten_per_marker: Dict[str, List[float]] = defaultdict(list)
    inten_per_ds_marker: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    inten_wc_per_marker: Dict[str, List[float]] = defaultdict(list)
    inten_wc_per_ds_marker: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_fov_cells: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for ds, fov in fov_pairs:
        images, mask = _read_fov_channels(gold_dir, ds, fov)
        if mask is None or not images:
            continue
        click.echo(f"  {ds}/{fov}: mask={mask.shape}, n_channels={len(images)}",
                   err=True)
        # per-FOV per-channel background
        bg_per_ch = {ch: per_fov_background(img, mask) for ch, img in images.items()}
        bboxes = _precompute_bboxes(mask)
        cell_ids = sorted(bboxes.keys())
        if max_cells_per_fov > 0:
            cell_ids = cell_ids[:max_cells_per_fov]
        cell_records: Dict[int, Dict[str, Any]] = {}
        for cid in cell_ids:
            bbox = bboxes.get(cid)
            if bbox is None:
                continue
            rmin, rmax, cmin, cmax = bbox
            sub_mask = (mask[rmin:rmax, cmin:cmax] == cid)
            if sub_mask.sum() < 9:
                continue
            comps = _compartment_masks(sub_mask)
            cell_inten: Dict[str, float] = {}
            cell_inten_wc: Dict[str, float] = {}
            cell_comp: Dict[str, str] = {}
            cell_coverage: Dict[str, float] = {}
            for ch, img in images.items():
                bg, mad = bg_per_ch[ch]
                sub_img = img[rmin:rmax, cmin:cmax]
                q = quantify_cell_marker(sub_img, sub_mask, ch, bg, mad,
                                         compartments=comps)
                if np.isfinite(q["summary"]):
                    inten_per_marker[ch].append(q["summary"])
                    inten_per_ds_marker[(ds, ch)].append(q["summary"])
                    cell_inten[ch] = q["summary"]
                    cell_comp[ch] = q["compartment"]
                    cell_coverage[ch] = q["coverage"]
                if np.isfinite(q.get("summary_wholecell", float("nan"))):
                    inten_wc_per_marker[ch].append(q["summary_wholecell"])
                    inten_wc_per_ds_marker[(ds, ch)].append(q["summary_wholecell"])
                    cell_inten_wc[ch] = q["summary_wholecell"]
            cell_records[cid] = {
                "area": int(sub_mask.sum()),
                "intensities": cell_inten,
                "intensities_wholecell": cell_inten_wc,
                "compartments": cell_comp,
                "coverage": cell_coverage,
            }
        per_fov_cells[(ds, fov)] = cell_records
    return (
        {k: np.asarray(v, dtype=np.float64) for k, v in inten_per_marker.items()},
        per_fov_cells,
        {k: np.asarray(v, dtype=np.float64) for k, v in inten_per_ds_marker.items()},
        {k: np.asarray(v, dtype=np.float64) for k, v in inten_wc_per_marker.items()},
        {k: np.asarray(v, dtype=np.float64) for k, v in inten_wc_per_ds_marker.items()},
    )


def _class_b_global_mad_fallback(all_vals: np.ndarray) -> Tuple[Optional[float], str]:
    """Fix 2 fallback: when a class-B marker has no hard-negative pool, use
    ``3 * 1.4826 * MAD`` over ALL cells in this bucket as the threshold (in
    log1p space). Threshold is *whole-cell-mean > 3*MAD*. Caller should pass
    the same summary statistic the pipeline will use at scoring time.
    """
    x = np.log1p(np.asarray(all_vals, dtype=np.float64))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, "fallback_no_data"
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    thr_log = 3.0 * 1.4826 * mad
    return float(np.expm1(thr_log)), "graded_fallback_3mad_global"


def _floor_gated_global_mad(
    all_vals: np.ndarray, global_pool: Optional[np.ndarray] = None,
    floor_pct: float = 5.0,
) -> Tuple[Optional[float], str]:
    """Iter-3: floor-gated Fix 2 fallback. Computes the global-MAD threshold
    via :func:`_class_b_global_mad_fallback`, then enforces
    ``threshold = max(threshold, percentile(global_pool, floor_pct))``.
    The floor stops FAP-style failure modes where the global MAD is so small
    that the threshold collapses to ~0 and every nonzero pixel becomes
    positive.

    Parameters
    ----------
    all_vals
        Per-bucket pool (the same pool fed to ``_class_b_global_mad_fallback``).
    global_pool
        Global per-marker intensity pool used to compute the floor; if
        ``None``, ``all_vals`` is used (less defensive but still better than
        no floor).
    floor_pct
        Percentile of the global pool to use as the floor (default 5).
    """
    thr, status = _class_b_global_mad_fallback(all_vals)
    if thr is None:
        return None, status
    floor_pool = global_pool if global_pool is not None and global_pool.size > 0 else all_vals
    floor_pool = np.asarray(floor_pool, dtype=np.float64)
    floor_pool = floor_pool[np.isfinite(floor_pool)]
    if floor_pool.size == 0:
        return float(thr), status + "_no_floor"
    floor = float(np.percentile(floor_pool, floor_pct))
    if floor > thr:
        return float(floor), status + f"_floored@p{int(floor_pct)}"
    return float(thr), status + f"_above_floor@p{int(floor_pct)}"


# ---------------------------------------------------------------------------
# Iter-3 Fix 4: derive coarse per-cell lineage tags from class-A signals.
# ---------------------------------------------------------------------------

def _derive_cell_lineage_tags(
    cells: Dict[int, Dict[str, Any]],
    ds: str,
    class_a_thresholds: Dict[Any, float],
    summary_choice: Dict[Any, str],
) -> Dict[int, set]:
    """For each cell in ``cells``, return the set of lineage tags that it
    matches based on the class-A markers with fitted thresholds.

    A cell is tagged ``lineage`` if at least one of the class-A markers in
    ``LINEAGE_DEFINITIONS[lineage]`` is present AND above its (ds, marker)
    threshold (with (None, marker) fallback). The chosen summary statistic
    is honoured per-bucket (Fix 1 + Fix 3).

    Side-effect-free; returns ``{cid -> set[str]}``.
    """
    out: Dict[int, set] = {}
    for cid, cell in cells.items():
        legacy_inten = cell.get("intensities", {})
        wc_inten = cell.get("intensities_wholecell", {})
        tags: set = set()
        for lineage, markers in LINEAGE_DEFINITIONS.items():
            for m in markers:
                if m not in legacy_inten and m not in wc_inten:
                    continue
                thr = _resolve_threshold(class_a_thresholds, ds, m)
                if thr is None:
                    continue
                choice = _resolve_summary_choice(summary_choice, ds, m)
                val = _pick_intensity(cell, m, choice)
                if np.isfinite(val) and val >= thr:
                    tags.add(lineage)
                    break  # one positive marker is enough for this lineage
        # mesenchymal exclusivity: SMA+ cells that are also lymphoid /
        # epithelial / endothelial are NOT mesenchymal — SMA cross-reactivity
        # on activated lymphocytes / pericytes confounds the pool. The intent
        # of "mesenchymal" here is "stromal SMA+ fibroblast / smooth muscle
        # without immune or epithelial signal".
        if "mesenchymal" in tags and (tags & {"lymphoid", "Tcell", "Bcell",
                                                "epithelial", "endothelial"}):
            tags.discard("mesenchymal")
        out[cid] = tags
    return out


def _build_lineage_negative_pool(
    per_fov_cells: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
    cell_lineage_tags: Dict[Tuple[str, str], Dict[int, set]],
    marker: str,
    summary: str,
    restrict_to_dataset: Optional[str] = None,
) -> np.ndarray:
    """For a given class-B ``marker``, collect the marker-intensity values
    of all cells whose lineage is in ``LINEAGE_HARD_NEGATIVES[marker]``.

    ``summary`` is "compartment" or "wholecell" — picks which intensity
    field to read (matches Fix-1 dispatch). ``restrict_to_dataset`` limits
    the pool to a single dataset (Fix-3 (ds, marker) bucket); ``None`` =
    pool across all datasets ((None, marker) fallback bucket).
    """
    hard_neg_lineages = LINEAGE_HARD_NEGATIVES.get(marker)
    if not hard_neg_lineages:
        return np.array([], dtype=np.float64)
    out: List[float] = []
    inten_key = "intensities_wholecell" if summary == "wholecell" else "intensities"
    for (ds, fov), cells in per_fov_cells.items():
        if restrict_to_dataset is not None and ds != restrict_to_dataset:
            continue
        tags_for_fov = cell_lineage_tags.get((ds, fov), {})
        for cid, cell in cells.items():
            tags = tags_for_fov.get(cid, set())
            if not (tags & hard_neg_lineages):
                continue
            inten = cell.get(inten_key, {})
            v = inten.get(marker)
            if v is None:
                # fallback to the other summary if requested one is missing
                alt_key = "intensities" if inten_key == "intensities_wholecell" else "intensities_wholecell"
                v = cell.get(alt_key, {}).get(marker)
            if v is None or not np.isfinite(v):
                continue
            out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def derive_thresholds(
    intensities: Dict[str, np.ndarray],
    hard_neg_intensities: Optional[Dict[str, np.ndarray]] = None,
    min_n: int = 200,
    *,
    intensities_wholecell: Optional[Dict[str, np.ndarray]] = None,
    intensities_per_ds_marker: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    intensities_wc_per_ds_marker: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    min_neg_for_mad: int = 50,
    min_bucket_n: int = 200,
    per_fov_cells: Optional[Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]] = None,
    min_neg_for_lineage: int = 50,
    fix4_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[Any, float], Dict[Any, str], Dict[Any, str]]:
    """For each marker, dispatch to its class fitter.

    Returns ``(thresholds, statuses, summary_choice)``.

    Backward-compatible shape: when called WITHOUT the new keyword args (as
    the pre-fix code path does), keys are plain marker strings — the legacy
    ``Dict[str, float]`` form. With the new keyword args, keys are
    ``(dataset_or_None, marker)`` tuples; ``(None, marker)`` is the
    fallback bucket. The pipeline will look up
    ``(ds, marker)`` first and fall back to ``(None, marker)``.

    ``summary_choice[bucket]`` is one of ``"compartment"`` / ``"wholecell"``
    indicating which summary statistic the threshold was fit on (Fix 1, only
    set for class-A markers; class-B is always ``"compartment"``).

    Iter-3 (Fix 4): when ``per_fov_cells`` is provided, after class-A
    thresholds are fit, per-cell lineage tags are derived from class-A
    signals and used to build a lineage-aware hard-negative pool for each
    class-B marker (per the ``LINEAGE_HARD_NEGATIVES`` table). When the
    lineage pool has fewer than ``min_neg_for_lineage`` cells, the code
    falls back to the prior "lower 70 percent of intensities" proxy. Below
    ``min_neg_for_mad``, falls further to the floor-gated Fix 2 global MAD
    (threshold cannot drop below the marker's 5th-percentile global
    intensity). ``fix4_diagnostics``, when supplied, is populated in-place
    with per-bucket "lineage" / "lower70" / "floored" tallies for telemetry.
    """
    new_api = (intensities_per_ds_marker is not None) or (
        intensities_wholecell is not None) or (
        intensities_wc_per_ds_marker is not None)

    thresholds: Dict[Any, float] = {}
    statuses: Dict[Any, str] = {}
    summary_choice: Dict[Any, str] = {}

    if not new_api:
        # Legacy code path — keep behaviour identical to pre-fix v2.
        for marker, vals in intensities.items():
            cls = marker_class(marker)
            if cls == "optout":
                statuses[marker] = "optout"
                continue
            if cls == "A":
                res = fit_threshold_class_a(vals, min_n=min_n)
                thr, status = res[0], res[1]
            else:  # B
                if hard_neg_intensities and marker in hard_neg_intensities:
                    neg = hard_neg_intensities[marker]
                else:
                    if vals.size == 0:
                        statuses[marker] = "no_data"
                        continue
                    neg = vals[vals < np.percentile(vals, 70.0)]
                thr, status = fit_threshold_class_b(neg, min_n=min_n)
            statuses[marker] = status
            if thr is not None:
                thresholds[marker] = thr
                summary_choice[marker] = "compartment"
        return thresholds, statuses, summary_choice

    # New API: per-(dataset, marker) stratification with (marker) fallback.
    intensities_per_ds_marker = intensities_per_ds_marker or {}
    intensities_wholecell = intensities_wholecell or {}
    intensities_wc_per_ds_marker = intensities_wc_per_ds_marker or {}

    # Build a helper that picks the better summary for a class-A marker by
    # comparing BIC ratios; persists choice and threshold.
    def _fit_class_a_dual(comp_vals: np.ndarray, wc_vals: np.ndarray
                          ) -> Tuple[Optional[float], str, str, Dict[str, Any]]:
        """Returns (threshold, status, summary_choice, diag)."""
        thr_c, st_c, br_c = fit_threshold_class_a(comp_vals, min_n=min_bucket_n)
        thr_w, st_w, br_w = fit_threshold_class_a(wc_vals, min_n=min_bucket_n)
        diag = {"bic_ratio_compartment": br_c, "bic_ratio_wholecell": br_w,
                "status_compartment": st_c, "status_wholecell": st_w}
        # Prefer the one that returned a valid bimodal threshold AND has the
        # lower BIC ratio (lower = more bimodal).
        if thr_c is not None and thr_w is not None:
            if (br_w if br_w is not None else 1.0) < (br_c if br_c is not None else 1.0):
                return thr_w, "bimodal_gmm_valley[wc]", "wholecell", diag
            return thr_c, "bimodal_gmm_valley[comp]", "compartment", diag
        if thr_c is not None:
            return thr_c, st_c + "[comp_only]", "compartment", diag
        if thr_w is not None:
            return thr_w, st_w + "[wc_only]", "wholecell", diag
        return None, st_c, "compartment", diag

    # Iter-3 telemetry: count which class-B path each bucket used.
    # Iter-5 (v3): also count IQR-cap/floor activations.
    diag_counters = {"lineage": 0, "lower70": 0, "global_mad_fallback": 0,
                     "global_mad_floored": 0, "no_data": 0,
                     "iqr_capped_at_p95": 0, "iqr_floored_at_p05": 0}

    def _apply_floor_and_cap(
        thr: float, marker: str, fallback_pool: np.ndarray,
    ) -> Tuple[float, str]:
        """Iter-5 (v3): apply the floor (>= p05_global) and cap (<= p95_global)
        to a class-B threshold fitted via the IQR rule.

        Uses the (None, marker) global pool as the reference for both
        gates — keeps the floor / cap stable across (ds, marker) buckets
        even when a per-ds pool is small.
        """
        ref = intensities.get(marker, fallback_pool)
        ref = np.asarray(ref, dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        if ref.size == 0:
            return float(thr), ""
        floor = float(np.percentile(ref, CLASS_B_FLOOR_PCT))
        cap = float(np.percentile(ref, CLASS_B_CAP_PCT))
        suffix = ""
        # Cap takes precedence over floor when both would fire (cap < floor
        # is theoretically impossible because p95 >= p05; just compare).
        if thr > cap:
            diag_counters["iqr_capped_at_p95"] += 1
            return float(cap), f"[capped@p{int(CLASS_B_CAP_PCT)}]"
        if thr < floor:
            diag_counters["iqr_floored_at_p05"] += 1
            return float(floor), f"[floored@p{int(CLASS_B_FLOOR_PCT)}]"
        return float(thr), suffix

    def _fit_class_b_with_lineage(
        marker: str, comp_vals: np.ndarray, wc_vals: np.ndarray,
        ds: Optional[str], summary_pref: str,
    ) -> Tuple[Optional[float], str]:
        """Iter-3 Fix 4 + Iter-5 IQR fit: lineage-aware -> lower70 ->
        floor-gated global MAD fallback.

        ``summary_pref`` is which summary statistic the lineage pool was
        collected with (matches the per-cell summary the pipeline will use
        at scoring time). Class-B always uses "compartment" today, but kept
        explicit so a future class-B-on-wholecell switch is one-liner.
        """
        global_pool = comp_vals if comp_vals.size > 0 else wc_vals
        # 1) Lineage-aware pool (only if per_fov_cells provided AND marker
        #    has a defined LINEAGE_HARD_NEGATIVES set).
        if (per_fov_cells is not None and marker in LINEAGE_HARD_NEGATIVES
                and cell_lineage_tags is not None):
            lineage_pool = _build_lineage_negative_pool(
                per_fov_cells, cell_lineage_tags, marker, summary_pref,
                restrict_to_dataset=ds,
            )
            if lineage_pool.size >= min_neg_for_lineage:
                thr, status = fit_threshold_class_b(
                    lineage_pool, min_n=min_neg_for_lineage)
                if thr is not None:
                    diag_counters["lineage"] += 1
                    thr, gate_suffix = _apply_floor_and_cap(
                        thr, marker, global_pool)
                    return thr, status + f"[lineage_n={lineage_pool.size}]" + gate_suffix
        # 2) Legacy lower-70 % proxy
        if global_pool.size > 0:
            neg = global_pool[global_pool < np.percentile(global_pool, 70.0)]
            if neg.size >= min_neg_for_mad:
                thr, status = fit_threshold_class_b(
                    neg, min_n=min_neg_for_mad)
                if thr is not None:
                    diag_counters["lower70"] += 1
                    thr, gate_suffix = _apply_floor_and_cap(
                        thr, marker, global_pool)
                    return thr, status + f"[lower70_n={neg.size}]" + gate_suffix
        # 3) Floor-gated Fix 2 global MAD
        if global_pool.size == 0:
            diag_counters["no_data"] += 1
            return None, "no_data"
        # Use the (None, marker) global pool as the floor reference even
        # when this is a (ds, marker) bucket; that way per-ds floors aren't
        # set by a possibly-tiny per-ds pool.
        floor_ref = intensities.get(marker, global_pool)
        thr, status = _floor_gated_global_mad(global_pool, global_pool=floor_ref)
        if thr is not None:
            if "floored" in status:
                diag_counters["global_mad_floored"] += 1
            else:
                diag_counters["global_mad_fallback"] += 1
        return thr, status

    # Determine bucket universe: union of all (ds, marker) keys + fallback (None, marker).
    all_markers = set(intensities.keys()) | set(intensities_wholecell.keys())
    ds_markers = set(intensities_per_ds_marker.keys()) | set(intensities_wc_per_ds_marker.keys())

    # -----------------------------------------------------------------
    # PASS 1: class-A (lineage-defining) thresholds — must fit before
    # lineage tags can be derived in pass 2. Class-B fitting is deferred
    # to pass 3 so it can use the lineage tags.
    # -----------------------------------------------------------------
    for marker in all_markers:
        cls = marker_class(marker)
        bucket = (None, marker)
        if cls == "optout":
            statuses[bucket] = "optout"
            continue
        if cls != "A":
            continue  # defer class-B
        comp_vals = intensities.get(marker, np.array([]))
        wc_vals = intensities_wholecell.get(marker, np.array([]))
        thr, status, choice, diag = _fit_class_a_dual(comp_vals, wc_vals)
        statuses[bucket] = (
            status + f" bic_c={diag['bic_ratio_compartment']} "
            f"bic_w={diag['bic_ratio_wholecell']}"
        )
        if thr is not None:
            thresholds[bucket] = thr
            summary_choice[bucket] = choice

    for (ds, marker) in ds_markers:
        cls = marker_class(marker)
        bucket = (ds, marker)
        if cls == "optout":
            statuses[bucket] = "optout"
            continue
        if cls != "A":
            continue  # defer class-B
        comp_vals = intensities_per_ds_marker.get((ds, marker), np.array([]))
        wc_vals = intensities_wc_per_ds_marker.get((ds, marker), np.array([]))
        if comp_vals.size < min_bucket_n and wc_vals.size < min_bucket_n:
            statuses[bucket] = f"bucket_too_small_n={comp_vals.size}_falls_back"
            continue
        thr, status, choice, diag = _fit_class_a_dual(comp_vals, wc_vals)
        statuses[bucket] = (
            status + f" bic_c={diag['bic_ratio_compartment']} "
            f"bic_w={diag['bic_ratio_wholecell']}"
        )
        if thr is not None:
            thresholds[bucket] = thr
            summary_choice[bucket] = choice

    # -----------------------------------------------------------------
    # PASS 2: derive per-cell lineage tags from class-A thresholds.
    # Skipped when ``per_fov_cells`` is None (legacy / unit-test path).
    # -----------------------------------------------------------------
    cell_lineage_tags: Optional[Dict[Tuple[str, str], Dict[int, set]]] = None
    if per_fov_cells is not None:
        cell_lineage_tags = {}
        for (ds, fov), cells in per_fov_cells.items():
            cell_lineage_tags[(ds, fov)] = _derive_cell_lineage_tags(
                cells, ds, thresholds, summary_choice)

    # -----------------------------------------------------------------
    # PASS 3: class-B thresholds with lineage-aware hard-neg pool.
    # -----------------------------------------------------------------
    for marker in all_markers:
        cls = marker_class(marker)
        bucket = (None, marker)
        if cls == "optout":
            continue  # already set in pass 1
        if cls != "B":
            continue
        comp_vals = intensities.get(marker, np.array([]))
        wc_vals = intensities_wholecell.get(marker, np.array([]))
        if comp_vals.size == 0 and wc_vals.size == 0:
            statuses[bucket] = "no_data"
            continue
        thr, status = _fit_class_b_with_lineage(
            marker, comp_vals, wc_vals, ds=None, summary_pref="compartment")
        statuses[bucket] = status
        if thr is not None:
            thresholds[bucket] = thr
            summary_choice[bucket] = "compartment"

    for (ds, marker) in ds_markers:
        cls = marker_class(marker)
        bucket = (ds, marker)
        if cls == "optout":
            continue
        if cls != "B":
            continue
        comp_vals = intensities_per_ds_marker.get((ds, marker), np.array([]))
        wc_vals = intensities_wc_per_ds_marker.get((ds, marker), np.array([]))
        if comp_vals.size < min_bucket_n and wc_vals.size < min_bucket_n:
            statuses[bucket] = f"bucket_too_small_n={comp_vals.size}_falls_back"
            continue
        thr, status = _fit_class_b_with_lineage(
            marker, comp_vals, wc_vals, ds=ds, summary_pref="compartment")
        statuses[bucket] = status
        if thr is not None:
            thresholds[bucket] = thr
            summary_choice[bucket] = "compartment"

    if fix4_diagnostics is not None:
        fix4_diagnostics.update(diag_counters)

    return thresholds, statuses, summary_choice


# ---------------------------------------------------------------------------
# Apply pipeline stages 1-6
# ---------------------------------------------------------------------------

def _resolve_threshold(
    thresholds: Dict[Any, float], ds: str, marker: str,
) -> Optional[float]:
    """Resolve threshold for (ds, marker) with fallback. Supports both
    legacy ``Dict[str, float]`` and new ``Dict[(ds_or_None, marker), float]``."""
    # Tuple-keyed form: try (ds, marker) then (None, marker)
    if (ds, marker) in thresholds:
        return thresholds[(ds, marker)]
    if (None, marker) in thresholds:
        return thresholds[(None, marker)]
    # Legacy str-keyed form
    if marker in thresholds:
        return thresholds[marker]
    return None


def _resolve_summary_choice(
    summary_choice: Dict[Any, str], ds: str, marker: str,
) -> str:
    """Resolve which summary statistic to use for (ds, marker)."""
    if (ds, marker) in summary_choice:
        return summary_choice[(ds, marker)]
    if (None, marker) in summary_choice:
        return summary_choice[(None, marker)]
    if marker in summary_choice:
        return summary_choice[marker]
    return "compartment"


def _pick_intensity(cell: Dict[str, Any], marker: str, choice: str) -> float:
    """Return the marker intensity using the chosen summary statistic.
    Falls back to the compartment summary if wholecell is requested but missing.
    """
    if choice == "wholecell":
        wc = cell.get("intensities_wholecell", {})
        if marker in wc and np.isfinite(wc[marker]):
            return wc[marker]
    inten = cell.get("intensities", {})
    return inten.get(marker, float("nan"))


def apply_pipeline(
    per_fov_cells: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
    thresholds: Dict[Any, float],
    cell_type_lookup: Optional[Dict[Tuple[str, str, int], str]] = None,
    stats: Optional[V2Stats] = None,
    summary_choice: Optional[Dict[Any, str]] = None,
) -> Dict[Tuple[str, str], FOVPredictions]:
    """Stage 1 (doublet) -> Stage 2 (CT-mislabel) -> Stage 5 (lineage excl) ->
    Stage 6 (emit per-cell-marker labels).

    ``thresholds`` may be either a legacy ``Dict[str, float]`` (pre-fix
    behaviour, treats the cell's ``intensities[marker]`` as the value) or
    a ``Dict[(ds_or_None, marker), float]`` (new) — when tuple-keyed and
    ``summary_choice`` is provided, the chosen summary statistic per bucket
    is used at scoring time (Fix 1 + Fix 3).
    """
    if stats is None:
        stats = V2Stats()
    if summary_choice is None:
        summary_choice = {}
    out: Dict[Tuple[str, str], FOVPredictions] = {}

    for (ds, fov), cells in per_fov_cells.items():
        stats.n_fovs += 1
        # --- Stage 1: precompute lineage-violation count per cell ---
        cell_ids = sorted(cells.keys())
        areas = np.array([cells[c]["area"] for c in cell_ids], dtype=np.float64)
        violations = np.zeros(len(cell_ids), dtype=int)
        for i, cid in enumerate(cell_ids):
            cell_i = cells[cid]
            inten_legacy = cell_i["intensities"]
            for a, b in LINEAGE_EXCLUSIONS:
                if a not in inten_legacy or b not in inten_legacy:
                    continue
                ta = _resolve_threshold(thresholds, ds, a)
                tb = _resolve_threshold(thresholds, ds, b)
                if ta is None or tb is None:
                    continue
                ca = _resolve_summary_choice(summary_choice, ds, a)
                cb = _resolve_summary_choice(summary_choice, ds, b)
                ia = _pick_intensity(cell_i, a, ca)
                ib = _pick_intensity(cell_i, b, cb)
                if (np.isfinite(ia) and np.isfinite(ib)
                        and ia >= ta and ib >= tb):
                    violations[i] += 1
        is_doublet = detect_doublets(areas, violations)
        stats.n_doublets += int(is_doublet.sum())

        fov_pred = FOVPredictions(dataset=ds, fov=fov)
        for i, cid in enumerate(cell_ids):
            stats.n_cells += 1
            cell = cells[cid]
            inten_legacy = cell["intensities"]

            # Build a per-cell intensities dict using the chosen summary
            # per marker (used for thresholding only — legacy ``inten``
            # is preserved for stage-2/-5 helpers that operate on the
            # compartment summary as a stable identity).
            inten_for_threshold: Dict[str, float] = {}
            thr_for_cell: Dict[str, float] = {}
            for marker in inten_legacy.keys():
                thr = _resolve_threshold(thresholds, ds, marker)
                if thr is None:
                    continue
                choice = _resolve_summary_choice(summary_choice, ds, marker)
                val = _pick_intensity(cell, marker, choice)
                if not np.isfinite(val):
                    continue
                inten_for_threshold[marker] = val
                thr_for_cell[marker] = thr

            # --- Stage 2: CT-mislabel (only if CT label provided) ---
            ct = None
            if cell_type_lookup is not None:
                ct = cell_type_lookup.get((ds, fov, cid))
            ct_suspect = detect_ct_mislabel(inten_for_threshold, ct,
                                             thr_for_cell, {})
            if ct_suspect:
                stats.n_ct_suspect += 1

            # --- Stage 6 baseline: per-marker call from threshold ---
            labels: Dict[str, Any] = {}
            for marker, val in inten_legacy.items():
                if marker_class(marker) == "optout":
                    labels[marker] = "?"
                    continue
                thr = _resolve_threshold(thresholds, ds, marker)
                if thr is None:
                    labels[marker] = "?"
                    continue
                choice = _resolve_summary_choice(summary_choice, ds, marker)
                used_val = _pick_intensity(cell, marker, choice)
                if not np.isfinite(used_val):
                    labels[marker] = "?"
                    continue
                labels[marker] = 1 if used_val >= thr else 0

            # --- Stage 5: lineage-exclusion (mask both members of any
            #            simultaneously-positive exclusion pair to "?") ---
            before = sum(1 for v in labels.values() if v != "?")
            labels = apply_lineage_exclusion(labels, inten_for_threshold,
                                              thr_for_cell)
            after = sum(1 for v in labels.values() if v != "?")
            stats.n_lineage_violations_masked += max(0, before - after)

            # --- Stage 1 / 2 fallout: mask ALL labels for doublets and
            #     CT-suspect cells ---
            if is_doublet[i] or ct_suspect:
                labels = {m: "?" for m in labels}

            for m, v in labels.items():
                if v == 1:
                    stats.n_pred_pos += 1
                    stats.per_marker_n_pos[m] += 1
                elif v == 0:
                    stats.n_pred_neg += 1
                    stats.per_marker_n_neg[m] += 1
                else:
                    stats.n_pred_unknown += 1

            fov_pred.cells[cid] = {
                "labels": labels,
                "intensities": inten_legacy,
                "area": cell["area"],
                "ct_suspect": bool(ct_suspect),
                "doublet": bool(is_doublet[i]),
                "ct": ct,
            }
        out[(ds, fov)] = fov_pred
    return out


# ---------------------------------------------------------------------------
# Unrefined-baseline predictions (for delta vs gold)
# ---------------------------------------------------------------------------

def baseline_predictions(
    per_fov_cells: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
    k_mad: float = 3.0,
) -> Dict[Tuple[str, str], FOVPredictions]:
    """Simple baseline: per-FOV background floor (median + k_mad*MAD) on the
    whole-cell mean. No compartment, no rules. This is what PR #35 effectively
    does in its degenerate case (whole_cell mean + global percentile).

    Concretely we just emit ``label = 1 if intensity > 0`` (the per-FOV bg
    has already been subtracted during quantification). This is intentionally
    weak so that the v2 pipeline has something to beat.
    """
    out: Dict[Tuple[str, str], FOVPredictions] = {}
    for (ds, fov), cells in per_fov_cells.items():
        fov_pred = FOVPredictions(dataset=ds, fov=fov)
        for cid, cell in cells.items():
            labels: Dict[str, Any] = {}
            for marker, val in cell["intensities"].items():
                if not np.isfinite(val):
                    labels[marker] = "?"
                    continue
                # naive: positive if any in-cell signal at all above bg
                labels[marker] = 1 if val > 0 else 0
            fov_pred.cells[cid] = {
                "labels": labels, "intensities": cell["intensities"],
                "area": cell["area"], "ct_suspect": False, "doublet": False,
                "ct": None,
            }
        out[(ds, fov)] = fov_pred
    return out


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def predictions_to_dict(pred: Dict[Tuple[str, str], FOVPredictions]) -> Dict[str, Any]:
    """Flatten predictions to JSON: {dataset: {fov: {cell_id: {channel: label}}}}."""
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for (ds, fov), fp in pred.items():
        for cid, info in fp.cells.items():
            out[ds][fov][str(cid)] = {
                "labels": info["labels"],
                "ct_suspect": info["ct_suspect"],
                "doublet": info["doublet"],
            }
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--gold_dir", type=click.Path(path_type=Path),
              default=Path("data/gold_standard/gold_standard_labelled"))
@click.option("--rulebook", type=click.Path(path_type=Path),
              default=Path("output/mp_matrix_assignments.json"),
              help="Optional MP rulebook (only used for reporting / deferred CT-mislabel "
                   "stage when a CT label is available).")
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("output/mp_refined_v2.json"))
@click.option("--baseline_output", type=click.Path(path_type=Path),
              default=Path("output/mp_unrefined_v2.json"))
@click.option("--max_cells_per_fov", type=int, default=2000, show_default=True)
@click.option("--exclude", multiple=True, default=("mibi_decidua",),
              help="Datasets to skip (default drops mibi_decidua per "
                   "docs/archive/mp_investigation_report.md).")
@click.option("--min_n_threshold", type=int, default=200, show_default=True,
              help="Min cells for class-A GMM / class-B MAD fit.")
@click.option("--dry_run", is_flag=True)
def main(gold_dir, rulebook, output, baseline_output, max_cells_per_fov,
         exclude, min_n_threshold, dry_run):
    fov_pairs = _list_gold_fovs(gold_dir, exclude=exclude)
    click.echo(f"Found {len(fov_pairs)} FOVs across "
               f"{len(set(d for d, _ in fov_pairs))} datasets")
    if not fov_pairs:
        click.echo(f"No FOVs found under {gold_dir}", err=True)
        return

    click.echo("Walking FOVs and quantifying intensities...")
    (intensities, per_fov_cells, intensities_per_ds_marker,
     intensities_wholecell, intensities_wc_per_ds_marker
     ) = collect_intensities_per_marker(
        gold_dir, fov_pairs, max_cells_per_fov)
    click.echo(f"Collected intensities for {len(intensities)} markers; "
               f"{sum(len(c) for c in per_fov_cells.values())} cells; "
               f"{len(intensities_per_ds_marker)} (dataset, marker) buckets")

    click.echo("Fitting per-(dataset, marker) thresholds with fallback...")
    fix4_diag: Dict[str, Any] = {}
    thresholds, statuses, summary_choice = derive_thresholds(
        intensities,
        min_n=min_n_threshold,
        intensities_wholecell=intensities_wholecell,
        intensities_per_ds_marker=intensities_per_ds_marker,
        intensities_wc_per_ds_marker=intensities_wc_per_ds_marker,
        min_bucket_n=min_n_threshold,
        per_fov_cells=per_fov_cells,
        fix4_diagnostics=fix4_diag,
    )
    click.echo(f"Fit thresholds for {len(thresholds)} buckets "
               f"(out of {len(statuses)} attempted)")
    # Summary of summary_choice for class-A markers
    n_wc = sum(1 for v in summary_choice.values() if v == "wholecell")
    n_comp = sum(1 for v in summary_choice.values() if v == "compartment")
    click.echo(f"  Fix 1 summary picks: compartment={n_comp}, wholecell={n_wc}")
    if fix4_diag:
        click.echo(
            f"  Fix 4 class-B path counts: "
            f"lineage={fix4_diag.get('lineage', 0)} "
            f"lower70={fix4_diag.get('lower70', 0)} "
            f"global_mad={fix4_diag.get('global_mad_fallback', 0)} "
            f"global_mad_floored={fix4_diag.get('global_mad_floored', 0)} "
            f"no_data={fix4_diag.get('no_data', 0)}"
        )
        click.echo(
            f"  Iter-5 IQR gate counts: "
            f"capped@p95={fix4_diag.get('iqr_capped_at_p95', 0)} "
            f"floored@p05={fix4_diag.get('iqr_floored_at_p05', 0)}"
        )

    click.echo("Applying pipeline stages 1, 2, 5, 6...")
    stats_statuses = {f"{k}": v for k, v in statuses.items()}
    stats = V2Stats(thresholds_fit=stats_statuses)
    refined = apply_pipeline(per_fov_cells, thresholds, stats=stats,
                              summary_choice=summary_choice)

    click.echo("Generating baseline (unrefined) predictions for delta...")
    baseline = baseline_predictions(per_fov_cells)

    click.echo(
        f"V2 stats: cells={stats.n_cells} "
        f"pos={stats.n_pred_pos} neg={stats.n_pred_neg} unk={stats.n_pred_unknown} "
        f"doublets={stats.n_doublets} ct_suspect={stats.n_ct_suspect} "
        f"lineage_masked={stats.n_lineage_violations_masked}"
    )

    if dry_run:
        click.echo("--dry_run: not writing outputs.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions_to_dict(refined), indent=2))
    baseline_output.write_text(json.dumps(predictions_to_dict(baseline), indent=2))
    stats_path = output.with_suffix(output.suffix + ".stats.json")
    # Serialize tuple-keyed thresholds/statuses/summary_choice as
    # "ds__marker" (with ds=__fallback__ for the (None, marker) bucket).
    def _key_to_str(k: Any) -> str:
        if isinstance(k, tuple):
            ds, m = k
            return f"{ds if ds is not None else '__fallback__'}__{m}"
        return str(k)
    stats_path.write_text(json.dumps({
        "n_fovs": stats.n_fovs, "n_cells": stats.n_cells,
        "n_pred_pos": stats.n_pred_pos, "n_pred_neg": stats.n_pred_neg,
        "n_pred_unknown": stats.n_pred_unknown,
        "n_doublets": stats.n_doublets, "n_ct_suspect": stats.n_ct_suspect,
        "n_lineage_violations_masked": stats.n_lineage_violations_masked,
        "thresholds": {_key_to_str(k): v for k, v in thresholds.items()},
        "threshold_status": {_key_to_str(k): v for k, v in statuses.items()},
        "summary_choice": {_key_to_str(k): v for k, v in summary_choice.items()},
        "fix4_diagnostics": fix4_diag,
    }, indent=2, default=float))
    click.echo(f"Wrote refined predictions  -> {output}")
    click.echo(f"Wrote unrefined baseline   -> {baseline_output}")
    click.echo(f"Wrote stats                 -> {stats_path}")


if __name__ == "__main__":
    main()
