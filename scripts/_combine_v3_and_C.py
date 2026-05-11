"""Hybrid combiner: v3 IQR refined labels + C's canonical lineage rules.

Background
----------
v3 (``scripts/refine_mp_labels_with_intensity_v2.py``) produces per-cell
binary MP calls via per-(dataset, marker) IQR-based thresholds (gold macro
F1 = 0.446). C (``scripts/refine_mp_labels_lineage_rules.py`` on
``origin/experiment/mp-bio-no-ct``) produces calls via per-FOV Otsu plus
12 mutual-exclusion + 4 implication lineage rules sourced from canonical
immunology references (gold macro F1 = 0.407).

A previous B+C combo failed (-0.016) because B emits binary calls only,
collapsing C's "strong-NEG" implication check to "binary call == 0".
v3 retains continuous intensity context, so implications can use real
intensity-vs-threshold comparison.

This script applies C's lineage rules (verbatim) to v3's calls:

  Mutual exclusion: both POS -> both "?"
  Implication:      A POS but B "strong-NEG" (intensity < 0.5 * threshold) -> A "?"

Per-cell intensities are re-extracted from the gold raw FOV TIFFs
(whole-cell mean inside each cell mask) -- mirrors the way C does it,
keeps v3 untouched, and matches C's per-FOV Otsu pipeline shape.

Inputs
------
- v3 labels JSON      : output/mp_refined_v3.json
- v3 stats sidecar    : output/mp_refined_v3.json.stats.json (thresholds)
- gold dir            : data/gold_standard/gold_standard_labelled

Output
------
- output/mp_v3_plus_c.json (force-add for git, since output/ is gitignored)

Usage::

    DATA_DIR=/data/xwang3/tissuenet-caitlin-labels.zarr \
    uv run python -m scripts._combine_v3_and_C \
        --v3_labels output/mp_refined_v3.json \
        --v3_stats output/mp_refined_v3.json.stats.json \
        --output output/mp_v3_plus_c.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np


# ---------------------------------------------------------------------------
# C's lineage rules -- VERBATIM copies from
# origin/experiment/mp-bio-no-ct:scripts/refine_mp_labels_lineage_rules.py
# ---------------------------------------------------------------------------

MUTUAL_EXCLUSIONS: List[Tuple[str, str, str]] = [
    ("CD3", "CD19", "T/B mutual exclusion (LeBien&Tedder Blood 2008)"),
    ("CD3", "CD20", "T/B mutual exclusion"),
    ("CD3", "CD68", "T-cell vs macrophage (Akashi Nature 2000)"),
    ("Cytokeratin", "CD45", "Epithelial vs hematopoietic (Moll JCB 1982)"),
    ("panCK", "CD45", "Epithelial vs hematopoietic"),
    ("CD4", "CD8", "Mature T-cell SP exclusion (Sprent&Kishimoto 2002)"),
    ("CD20", "CD68", "B-cell vs macrophage"),
    ("CD31", "Cytokeratin", "Endothelial vs epithelial (Pusztaszeri 2006)"),
    ("CD31", "panCK", "Endothelial vs epithelial"),
    ("CD56", "CD3", "NK vs T (Godfrey NRI 2010)"),
    ("FoxP3", "CD8", "Treg requires CD4 (Sakaguchi 2004)"),
    ("SMA", "CD45", "Smooth muscle vs hematopoietic (Skalli JCB 1986)"),
]

IMPLICATIONS: List[Tuple[str, str, str]] = [
    ("CD3", "CD45", "T-cell implies CD45+ (Trowbridge 1994)"),
    ("CD20", "CD45", "B-cell implies CD45+"),
    ("CD19", "CD45", "B-cell implies CD45+"),
    ("CD68", "CD45", "Macrophage implies CD45+"),
]

# Strong-NEG factor (C's verbatim convention: ``intensity < 0.5 * threshold``)
STRONG_NEG_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Marker-name canonicalization (matches C and v3 conventions)
# ---------------------------------------------------------------------------

MARKER_CANON: Dict[str, str] = {
    "panCK+CK7+CAM5.2": "panCK",
    "PD-L1": "PDL1",
    "PD-1": "PD1",
    "HLADR": "HLA-DR",
    "Foxp3": "FoxP3",
    "aSMA": "SMA",
}


def _canon_marker(name: str) -> str:
    return MARKER_CANON.get(name, name)


# ---------------------------------------------------------------------------
# Threshold resolution -- mirrors v3's _resolve_threshold(thresholds, ds, marker)
# ---------------------------------------------------------------------------

def _parse_threshold_key(key: str) -> Tuple[Optional[str], str]:
    """``ds__marker`` or ``__fallback____marker`` -> (ds_or_None, marker)."""
    if key.startswith("__fallback____"):
        return None, key[len("__fallback____"):]
    if "__" in key:
        ds, _, marker = key.partition("__")
        return ds, marker
    return None, key


def _build_threshold_lookup(stats_blob: Dict[str, Any]
                             ) -> Tuple[Dict[Tuple[Optional[str], str], float],
                                        Dict[Tuple[Optional[str], str], str]]:
    thresholds_raw = stats_blob.get("thresholds", {})
    statuses_raw = stats_blob.get("threshold_status", {})
    thr: Dict[Tuple[Optional[str], str], float] = {}
    status: Dict[Tuple[Optional[str], str], str] = {}
    for k, v in thresholds_raw.items():
        ds, marker = _parse_threshold_key(k)
        thr[(ds, marker)] = float(v)
    for k, v in statuses_raw.items():
        ds, marker = _parse_threshold_key(k)
        status[(ds, marker)] = str(v)
    return thr, status


def _resolve_threshold(thr: Dict[Tuple[Optional[str], str], float],
                        ds: str, marker: str) -> Optional[float]:
    if (ds, marker) in thr:
        return thr[(ds, marker)]
    if (None, marker) in thr:
        return thr[(None, marker)]
    return None


# ---------------------------------------------------------------------------
# Per-cell mean-intensity extraction (mirrors C's _per_cell_means / _read_fov)
# ---------------------------------------------------------------------------

def _read_fov_images_and_mask(gold_dir: Path, dataset: str, fov: str
                               ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Mirror of C's _read_fov: load *.ome.tif channels + mask for one FOV."""
    import tifffile
    fov_dir = gold_dir / dataset / "fovs" / fov
    if not fov_dir.exists():
        return {}, None
    images: Dict[str, np.ndarray] = {}
    paths = sorted(fov_dir.glob("*.ome.tif")) or sorted(fov_dir.glob("*.tiff"))
    for path in paths:
        name = path.name
        ch = name
        for ext in (".ome.tif", ".tiff", ".tif"):
            if name.endswith(ext):
                ch = name[: -len(ext)]
                break
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
            images[ch] = img.astype(np.float32)
        except Exception:
            continue

    masks_dir = gold_dir / dataset / "masks"
    mask_path: Optional[Path] = None
    for cand_name in (f"{fov}.ome.tif", f"{fov}.tif", f"{fov}feature_0.ome.tif"):
        cand = masks_dir / cand_name
        if cand.exists():
            mask_path = cand
            break
    if mask_path is None and masks_dir.exists():
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
    mask = None
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


def _per_cell_means(image: np.ndarray, mask: np.ndarray,
                    cell_ids: np.ndarray) -> Dict[int, float]:
    """Whole-cell mean intensity for each cell in cell_ids (C's helper)."""
    from scipy.ndimage import mean as ndi_mean
    if cell_ids.size == 0:
        return {}
    means = ndi_mean(image, labels=mask, index=cell_ids)
    out: Dict[int, float] = {}
    for cid, m in zip(cell_ids.tolist(), means.tolist()):
        if not np.isnan(m):
            out[int(cid)] = float(m)
    return out


def extract_intensities_for_fov(gold_dir: Path, dataset: str, fov: str
                                 ) -> Dict[int, Dict[str, float]]:
    """Per-cell whole-cell mean intensity per canonical marker for one FOV."""
    images, mask = _read_fov_images_and_mask(gold_dir, dataset, fov)
    if mask is None or not images:
        return {}
    cell_ids = np.unique(mask[mask > 0])
    if cell_ids.size == 0:
        return {}
    # Aggregate across raw->canonical aliases by max-mean (avoids zeroing
    # when a panel uses one of several alias channels).
    out: Dict[int, Dict[str, float]] = {int(c): {} for c in cell_ids}
    canon_to_raw: Dict[str, List[str]] = defaultdict(list)
    for raw_ch in images:
        canon_to_raw[_canon_marker(raw_ch)].append(raw_ch)
    for canon, raw_list in canon_to_raw.items():
        best_means: Optional[Dict[int, float]] = None
        for raw_ch in raw_list:
            means = _per_cell_means(images[raw_ch], mask, cell_ids)
            if not means:
                continue
            if best_means is None:
                best_means = means
            else:
                # Take max per cell (keeps signal when multiple alias
                # channels exist; mirrors C's "best bimodal" intent).
                for cid, v in means.items():
                    prev = best_means.get(cid, -np.inf)
                    if v > prev:
                        best_means[cid] = v
        if best_means is None:
            continue
        for cid, v in best_means.items():
            out[int(cid)][canon] = v
    return out


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def apply_rules_to_cell(
    labels: Dict[str, Any],
    intensities: Dict[str, float],
    thresholds: Dict[str, float],
    rule_counts: Optional[Dict[str, int]] = None,
    per_marker_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Apply C's mutual exclusions + implications to one cell's labels.

    ``labels`` is a {marker: 0|1|"?"} dict (from v3).
    ``intensities`` is a {canonical_marker: float} dict (whole-cell mean,
    re-extracted from raw images).
    ``thresholds`` is a {marker: float} dict (resolved per-(ds, marker)
    from v3's stats sidecar).

    Returns a NEW dict (never mutates the input).
    """
    out = dict(labels)
    if rule_counts is None:
        rule_counts = {}
    if per_marker_counts is None:
        per_marker_counts = {}

    # Build canonical->key map so panel names like "aSMA" resolve to rule
    # names like "SMA" without forcing rule-side rewriting.
    canon_to_key: Dict[str, str] = {}
    for k in list(out.keys()):
        canon_to_key.setdefault(_canon_marker(k), k)

    def _get_label(m: str) -> Any:
        canon = _canon_marker(m)
        if m in out:
            return out[m]
        key = canon_to_key.get(canon) or canon_to_key.get(m)
        if key is not None:
            return out[key]
        return None

    def _set_label(m: str, val: Any) -> None:
        canon = _canon_marker(m)
        if m in out:
            out[m] = val
            return
        key = canon_to_key.get(canon) or canon_to_key.get(m)
        if key is not None:
            out[key] = val
            return
        out[m] = val
        canon_to_key.setdefault(canon, m)

    # Mutual exclusion: both POS -> both "?"
    for a, b, _ in MUTUAL_EXCLUSIONS:
        la = _get_label(a)
        lb = _get_label(b)
        if la == 1 and lb == 1:
            _set_label(a, "?")
            _set_label(b, "?")
            rule_key = f"excl[{a}|{b}]"
            rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1
            for m in (a, b):
                per_marker_counts.setdefault(m, {})
                per_marker_counts[m]["excl"] = per_marker_counts[m].get("excl", 0) + 1

    # Implication: A POS, B "strong-NEG" (intensity < STRONG_NEG_FACTOR * thr) -> A "?"
    for a, b, _ in IMPLICATIONS:
        la = _get_label(a)
        if la != 1:
            continue
        ib = intensities.get(b)
        # also try canonical form -> raw if missing (shouldn't happen normally)
        if ib is None:
            ib = intensities.get(_canon_marker(b))
        tb = thresholds.get(b)
        if tb is None:
            tb = thresholds.get(_canon_marker(b))
        if ib is None or tb is None:
            continue
        if ib < STRONG_NEG_FACTOR * tb:
            _set_label(a, "?")
            rule_key = f"impl[{a}=>{b}]"
            rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1
            per_marker_counts.setdefault(a, {})
            per_marker_counts[a]["impl"] = per_marker_counts[a].get("impl", 0) + 1

    return out


# ---------------------------------------------------------------------------
# Coverage / change diagnostics
# ---------------------------------------------------------------------------

def _count_definite(labels: Dict[str, Any]) -> Tuple[int, int]:
    n_def = sum(1 for v in labels.values() if v in (0, 1))
    n_total = len(labels)
    return n_def, n_total


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------

@click.command()
@click.option("--v3_labels", type=click.Path(path_type=Path),
              default=Path("output/mp_refined_v3.json"))
@click.option("--v3_stats", type=click.Path(path_type=Path),
              default=Path("output/mp_refined_v3.json.stats.json"))
@click.option("--gold_dir", type=click.Path(path_type=Path),
              default=Path("data/gold_standard/gold_standard_labelled"))
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("output/mp_v3_plus_c.json"))
@click.option("--exclude", multiple=True, default=("mibi_decidua",))
def main(v3_labels: Path, v3_stats: Path, gold_dir: Path, output: Path,
         exclude: Tuple[str, ...]):
    click.echo(f"Loading v3 labels from {v3_labels}...")
    v3 = json.loads(v3_labels.read_text())
    click.echo(f"Loading v3 thresholds from {v3_stats}...")
    stats_blob = json.loads(v3_stats.read_text())
    thr_lookup, _ = _build_threshold_lookup(stats_blob)

    rule_counts: Dict[str, int] = {}
    per_marker_counts: Dict[str, Dict[str, int]] = {}
    n_cells = n_def_before = n_def_after = 0

    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict))
    for ds, ds_block in v3.items():
        if ds in exclude:
            continue
        # Resolve per-marker threshold map for this dataset
        # (v3's resolve = ds-specific then __fallback__ fallback)
        marker_pool = set()
        for fov_block in ds_block.values():
            for cell_block in fov_block.values():
                marker_pool.update(cell_block.get("labels", {}).keys())
        ds_thr: Dict[str, float] = {}
        for m in marker_pool:
            t = _resolve_threshold(thr_lookup, ds, m)
            if t is not None:
                ds_thr[m] = t

        for fov, fov_block in ds_block.items():
            click.echo(f"  Processing {ds}/{fov} ({len(fov_block)} cells)...")
            intens_by_cell = extract_intensities_for_fov(gold_dir, ds, fov)
            for cid_str, cell_block in fov_block.items():
                labels = dict(cell_block.get("labels", {}))
                ct_suspect = bool(cell_block.get("ct_suspect", False))
                doublet = bool(cell_block.get("doublet", False))
                # If v3 already masked everything (doublet/ct_suspect),
                # nothing for our rules to do. Pass labels through.
                if ct_suspect or doublet or all(v == "?" for v in labels.values()):
                    out_labels = labels
                else:
                    try:
                        cid_int = int(cid_str)
                    except (TypeError, ValueError):
                        cid_int = -1
                    intensities = intens_by_cell.get(cid_int, {})
                    nb, _ = _count_definite(labels)
                    n_def_before += nb
                    out_labels = apply_rules_to_cell(
                        labels, intensities, ds_thr,
                        rule_counts=rule_counts,
                        per_marker_counts=per_marker_counts,
                    )
                    na, _ = _count_definite(out_labels)
                    n_def_after += na
                out[ds][fov][cid_str] = {
                    "labels": out_labels,
                    "ct_suspect": ct_suspect,
                    "doublet": doublet,
                }
                n_cells += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({k: dict(v) for k, v in out.items()}, indent=2))
    stats_path = output.with_suffix(output.suffix + ".stats.json")
    stats_path.write_text(json.dumps({
        "n_cells": n_cells,
        "n_definite_before_rules": n_def_before,
        "n_definite_after_rules": n_def_after,
        "n_masked_by_rules": n_def_before - n_def_after,
        "rule_counts": rule_counts,
        "per_marker_counts": per_marker_counts,
        "n_mutual_exclusions": len(MUTUAL_EXCLUSIONS),
        "n_implications": len(IMPLICATIONS),
        "strong_neg_factor": STRONG_NEG_FACTOR,
    }, indent=2))
    click.echo(f"Wrote combined predictions -> {output}")
    click.echo(f"Wrote stats                 -> {stats_path}")
    click.echo(f"Total cells: {n_cells}; "
               f"definite before rules: {n_def_before}; "
               f"after: {n_def_after}; "
               f"masked: {n_def_before - n_def_after}")
    click.echo(f"Rule fire counts: {dict(sorted(rule_counts.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
