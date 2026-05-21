#!/usr/bin/env python3
"""
Benchmark marker positivity prediction on the Pan-Multiplex Gold Standard dataset.

Compares:
  1. Nimbus (pretrained UNet) — pixel-level marker prediction → per-cell mean
  2. Our model's marker positivity head — patch-level prediction

The gold standard contains ~1.1M expert-annotated marker positivity labels
(0=negative, 1=positive, 2/3=ambiguous) across 5 tissue/modality subsets.

Reference: Rumberger et al., Nature Methods 2025.
Dataset: https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard

Usage:
    # Download first:
    bash scripts/download_gold_standard.sh

    # Run benchmark (Nimbus only — needs nimbus venv):
    source .venv_nimbus/bin/activate
    python scripts/benchmark_gold_standard.py --method nimbus --device cuda:0

    # Run benchmark (our model — needs main venv):
    source .venv/bin/activate
    python scripts/benchmark_gold_standard.py --method ours \\
        --checkpoint ~/.deepcell/models/deepcell-types_2026-05-17.pt --device cuda:0

    # Run both (two separate runs, then compare):
    python scripts/benchmark_gold_standard.py --method compare \\
        --nimbus_results output/gold_standard_nimbus.json \\
        --ours_results output/gold_standard_ours.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def discover_gold_standard_subsets(gold_dir: Path) -> dict:
    """Discover available subsets in the gold standard directory.

    Supports two layouts:

    (A) Per-subset labels (legacy):
        <gold_dir>/<subset>/{raw|images}/, <gold_dir>/<subset>/{labels|label}/,
        <gold_dir>/<subset>/{masks|mask}/

    (B) Central CSV + per-FOV folders (Pan-Multiplex Gold Standard as
        distributed by Rumberger et al.):
        <gold_dir>/<subset>/fovs/<fov_name>/<marker>.ome.tif
        <gold_dir>/<subset>/masks/<fov_name>.ome.tif
        <gold_dir>/gold_standard_groundtruth.csv  (one row per (fov, cell_id, channel, activity))

    Returns:
        Dict mapping subset_name -> {
            "images_dir": Path to per-FOV image directories root,
            "labels_dir": Path to per-subset label CSVs (or None if central CSV),
            "masks_dir": Path to segmentation masks,
            "central_csv": Path to central CSV (or None),
        }
    """
    subsets = {}
    central_csv = gold_dir / "gold_standard_groundtruth.csv"
    if not central_csv.exists():
        central_csv = None

    # The extracted zip may have a top-level wrapper dir; search both
    # `gold_dir` and one level down. Subsets are subdirectories that contain
    # either a labels/ dir (layout A) or a fovs/ dir (layout B).
    for candidate in [gold_dir, *gold_dir.iterdir()]:
        if not candidate.is_dir():
            continue
        for subset_dir in candidate.iterdir():
            if not subset_dir.is_dir():
                continue
            name = subset_dir.name
            if name in subsets:
                continue
            images_dir = (
                subset_dir / "raw" if (subset_dir / "raw").exists()
                else subset_dir / "images" if (subset_dir / "images").exists()
                else subset_dir / "fovs" if (subset_dir / "fovs").exists()
                else None
            )
            labels_dir = (
                subset_dir / "labels" if (subset_dir / "labels").exists()
                else subset_dir / "label" if (subset_dir / "label").exists()
                else None
            )
            masks_dir = (
                subset_dir / "masks" if (subset_dir / "masks").exists()
                else subset_dir / "mask" if (subset_dir / "mask").exists()
                else None
            )
            # Accept if either per-subset labels exist OR a central CSV
            # is present at the gold_dir root and an images_dir exists.
            if labels_dir is not None or (central_csv is not None and images_dir is not None):
                subsets[name] = {
                    "root": subset_dir,
                    "images_dir": images_dir,
                    "labels_dir": labels_dir,
                    "masks_dir": masks_dir,
                    "central_csv": central_csv,
                }
    return subsets


def _load_labels_from_central_csv(central_csv: Path, subset_name: str) -> pd.DataFrame:
    """Slice the central Pan-Multiplex CSV for one subset and reshape to the
    (fov, cell_index, marker, label) schema used downstream.

    The central CSV has columns: dataset, fov, cell_id, channel, activity.
    """
    df = pd.read_csv(central_csv)
    sub = df[df["dataset"] == subset_name]
    if sub.empty:
        return pd.DataFrame(columns=["fov", "cell_index", "marker", "label"])
    out = pd.DataFrame({
        "fov": sub["fov"].astype(str).values,
        "cell_index": sub["cell_id"].astype(int).values,
        "marker": sub["channel"].astype(str).values,
        "label": sub["activity"].astype(int).values,
    })
    return out


def load_gold_standard_labels(labels_dir: Path) -> pd.DataFrame:
    """Load gold standard marker positivity labels from CSV files.

    Returns:
        DataFrame with columns: fov, cell_index, marker, label
        where label is 0 (negative), 1 (positive), 2/3 (ambiguous)
    """
    records = []
    for csv_file in sorted(labels_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        # Extract FOV name from filename (convention: {fov_name}_{suffix}.csv)
        fov_name = "_".join(csv_file.stem.split("_")[:-1])
        if not fov_name:
            fov_name = csv_file.stem

        # The CSV may have columns: cell_index_orig, marker1, marker2, ...
        # or cell_index, marker1, marker2, ...
        id_col = "cell_index_orig" if "cell_index_orig" in df.columns else "cell_index"
        if id_col not in df.columns:
            # Try first column as ID
            id_col = df.columns[0]

        marker_cols = [c for c in df.columns if c != id_col and c != "cell_type"]

        for _, row in df.iterrows():
            for marker in marker_cols:
                val = row[marker]
                if pd.isna(val):
                    continue
                records.append({
                    "fov": fov_name,
                    "cell_index": int(row[id_col]),
                    "marker": marker,
                    "label": int(val),
                })

    return pd.DataFrame(records)


def evaluate_predictions(
    labels_df: pd.DataFrame,
    predictions: dict,
    threshold: float = 0.5,
) -> dict:
    """Evaluate marker positivity predictions against the Pan-Multiplex Gold Standard.

    Reduction routes through ``deepcell_types.training.utils.summarize_mp_per_marker`` —
    the same helper used by the main model's ``MPMetricsTracker`` and the
    Nimbus baseline's ``compute_marker_positivity_metrics``. So a gold-set
    table comparing methods is bit-exact apples-to-apples on macro/micro F1.

    Args:
        labels_df: Gold standard labels (fov, cell_index, marker, label).
            Ambiguous codes (2, 3) are dropped automatically.
        predictions: Dict mapping (fov, cell_index, marker) -> predicted score (0-1).
        threshold: Threshold for binarizing predictions.

    Returns:
        Dict with overall (shared-helper keys) and per-marker metrics.
    """
    from deepcell_types.training.utils import summarize_mp_per_marker

    # Strict gold-only: keep 0 and 1, drop 2/3 ambiguous (Rumberger et al. convention)
    valid = labels_df[labels_df["label"].isin([0, 1])].copy()

    y_true = []
    y_score = []
    y_pred = []
    markers = []

    for _, row in valid.iterrows():
        key = (row["fov"], row["cell_index"], row["marker"])
        if key in predictions:
            y_true.append(row["label"])
            score = predictions[key]
            y_score.append(score)
            y_pred.append(1 if score >= threshold else 0)
            markers.append(row["marker"])

    if not y_true:
        return {"error": "No matching predictions found"}

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)
    markers = np.array(markers)

    # Per-marker tp/fp/fn/tn — canonical intermediate for the shared reduction
    per_marker_counts = {}
    per_marker = {}
    for marker in np.unique(markers):
        mask = markers == marker
        yt = y_true[mask]
        yp = y_pred[mask]
        ys = y_score[mask]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        per_marker_counts[marker] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        m = {
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "n_samples": int(len(yt)),
            "n_positive": int(yt.sum()),
        }
        try:
            m["auroc"] = float(roc_auc_score(yt, ys))
        except ValueError:
            m["auroc"] = None
        per_marker[marker] = m

    # Shared reduction (bit-exact identical to MPMetricsTracker + Nimbus baseline)
    summary = summarize_mp_per_marker(per_marker_counts)

    overall = {
        # Shared-helper keys (use these for apples-to-apples comparison):
        "mp_macro_f1": summary["mp_macro_f1"],
        "mp_micro_f1": summary["mp_micro_f1"],
        "mp_macro_precision": summary["mp_macro_precision"],
        "mp_macro_recall": summary["mp_macro_recall"],
        "mp_macro_accuracy": summary["mp_macro_accuracy"],
        "mp_micro_precision": summary["mp_micro_precision"],
        "mp_micro_recall": summary["mp_micro_recall"],
        "mp_num_markers": summary["mp_num_markers"],
        "mp_num_markers_excluded_from_macro_f1": summary["mp_num_markers_excluded_from_macro_f1"],
        # Legacy / informational:
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }
    try:
        overall["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        overall["auroc"] = None

    return {"overall": overall, "per_marker": per_marker}


def run_nimbus_benchmark(gold_dir: Path, device: str = "cuda:0") -> dict:
    """Run Nimbus inference on gold standard data and evaluate.

    Requires: nimbus_inference package (install in .venv_nimbus)
    """
    from nimbus_inference.nimbus import Nimbus, prep_naming_convention
    from nimbus_inference.utils import MultiplexDataset

    subsets = discover_gold_standard_subsets(gold_dir)
    if not subsets:
        print(f"ERROR: No subsets found in {gold_dir}")
        print("Run: bash scripts/download_gold_standard.sh first")
        return {}

    all_predictions = {}
    all_labels = []

    for subset_name, paths in subsets.items():
        print(f"\n--- Subset: {subset_name} ---")

        if paths["images_dir"] is None or paths["masks_dir"] is None:
            print(f"  Skipping (missing images or masks)")
            continue

        # Load gold standard labels — prefer per-subset CSVs, fall back to the
        # central Pan-Multiplex CSV at the gold_dir root.
        if paths["labels_dir"] is not None:
            labels_df = load_gold_standard_labels(paths["labels_dir"])
        elif paths["central_csv"] is not None:
            labels_df = _load_labels_from_central_csv(paths["central_csv"], subset_name)
        else:
            labels_df = pd.DataFrame()
        if labels_df.empty:
            print(f"  Skipping (no labels)")
            continue
        labels_df["subset"] = subset_name
        all_labels.append(labels_df)
        print(f"  Labels: {len(labels_df)} annotations")

        # Prepare Nimbus input
        tiff_dir = paths["images_dir"]
        mask_dir = paths["masks_dir"]

        fov_paths = [str(p) for p in tiff_dir.iterdir() if p.is_dir()]
        if not fov_paths:
            print(f"  Skipping (no FOV directories)")
            continue

        # Check for shape.txt
        shape_file = tiff_dir / "shape.txt"
        if shape_file.exists():
            with open(shape_file) as f:
                input_shape = [int(x) for x in f.read().strip().split()]
        else:
            input_shape = [1024, 1024]

        # Build a naming convention that tolerates the Pan-Multiplex layout
        # (masks named ``<fov>.ome.tif``) as well as the legacy DeepCell layout
        # (``<fov>_whole_cell.tiff``). We probe both per FOV.
        def segmentation_naming_convention(fov_path, _mask_dir=str(mask_dir)):
            import os as _os
            fov_name = _os.path.basename(fov_path)
            # Strip trailing image-suffixes that some Pan-Multiplex subsets append
            # to FOV folder names but not to mask filenames.
            for _strip in (
                ".ome.tiff", ".ome.tif", ".tiff", ".tif",
                ".tif_image", "_image",
            ):
                if fov_name.endswith(_strip):
                    fov_name = fov_name[: -len(_strip)]
                    break
            for cand in (
                f"{fov_name}.ome.tif",
                f"{fov_name}.ome.tiff",
                f"{fov_name}feature_0.ome.tif",
                f"{fov_name}feature_0.ome.tiff",
                f"{fov_name}_whole_cell.tiff",
                f"{fov_name}_whole_cell.tif",
                f"{fov_name}.tiff",
                f"{fov_name}.tif",
            ):
                p = _os.path.join(_mask_dir, cand)
                if _os.path.exists(p):
                    return p
            # Prefix-match fallback: some Pan-Multiplex subsets append arbitrary
            # suffixes (e.g. ``feature_0``, ``_image``) between the FOV stem and
            # the file extension. Pick the unique prefix match if there is one.
            try:
                cands = [
                    f for f in _os.listdir(_mask_dir)
                    if f.startswith(fov_name)
                ]
                if len(cands) == 1:
                    return _os.path.join(_mask_dir, cands[0])
            except OSError:
                pass
            # Fall back to DeepCell-style name; Nimbus will raise a clean error.
            return _os.path.join(_mask_dir, f"{fov_name}_whole_cell.tiff")

        # Per-FOV folders in the Pan-Multiplex gold layout hold ``.ome.tif``
        # files (e.g. ``CD3.ome.tif``); accept either suffix to stay backward-
        # compatible with the legacy ``.tiff`` layout.
        sample_fov_files = list(Path(fov_paths[0]).iterdir())
        suffix = ".ome.tif" if any(
            p.name.endswith(".ome.tif") for p in sample_fov_files
        ) else ".tiff"
        output_dir = str(gold_dir / "nimbus_output" / subset_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        dataset = MultiplexDataset(
            fov_paths=fov_paths,
            suffix=suffix,
            include_channels=[],
            segmentation_naming_convention=segmentation_naming_convention,
            output_dir=output_dir,
        )

        # Nimbus v0.0.5: ``prepare_normalization_dict`` lives on the dataset,
        # not on the ``Nimbus`` instance. ``Nimbus.predict_fovs`` reaches into
        # the dataset for the normalization dict at inference time and will
        # auto-prepare lazily, but we trigger it explicitly so progress is
        # visible and any quantile-related errors surface up front.
        dataset.prepare_normalization_dict(
            n_subset=50, multiprocessing=False, overwrite=True
        )

        nimbus = Nimbus(
            dataset=dataset,
            output_dir=output_dir,
            save_predictions=False,
            batch_size=1,
            test_time_aug=True,
            input_shape=input_shape,
            device=device,
        )

        nimbus.check_inputs()

        # Run inference
        cell_table = nimbus.predict_fovs()
        print(f"  Nimbus predictions: {len(cell_table)} cells")

        # Convert to (fov, cell_index, marker) -> score
        cell_table.rename(columns={"label": "cell_index"}, inplace=True)
        meta_cols = {"cell_index", "fov"}
        marker_cols = [c for c in cell_table.columns if c not in meta_cols]

        for _, row in cell_table.iterrows():
            fov = row["fov"]
            cell_idx = int(row["cell_index"])
            for marker in marker_cols:
                all_predictions[(fov, cell_idx, marker)] = float(row[marker])

    if not all_labels:
        return {"error": "No labels found"}

    labels_combined = pd.concat(all_labels, ignore_index=True)
    print(f"\nTotal labels: {len(labels_combined)}")
    print(f"Total predictions: {len(all_predictions)}")

    results = evaluate_predictions(labels_combined, all_predictions)
    results["method"] = "nimbus"
    return results


def _build_gold_channel_name_map(marker2idx: dict) -> dict:
    """Map gold standard channel names to our marker vocabulary names.

    Returns dict: gold_name -> our_name (only for channels that have a match).
    """
    mapping = {}
    # Direct matches (case-sensitive)
    for ch in marker2idx:
        mapping[ch] = ch

    # Gold standard uses different naming conventions for some markers
    aliases = {
        "Foxp3": "FoxP3",
        "FOXP3": "FoxP3",
        "PD-L1": "PDL1",
        "PD-1": "PD1",
        "aSMA": "SMA",
        "aDefensin5": "DEFA5",
        "panCK": "PanCK",
        "panCK+CK7+CAM5.2": "PanCK",
        "Cytokeratin": "PanCK",
        "CD40-L": "CD154",
        "BCL2": "Bcl-2",
        "CD45RB": "CD45R",
        "Collagen1": "COL1",
        "ECAD": "E-cadherin",
        "DCSIGN": "CD209",
        "HLADR": "HLA-Class-2",
        "HLAG": "HLA-G",
        "VIM": "Vimentin",
        "Podoplanin": "PDPN",
        "ChyTr": "Chymase",
        "CD117": "c-kit",
        "CD123": "CD123",
        "CD127": "CD127",
    }
    for gold_name, our_name in aliases.items():
        if our_name in marker2idx:
            mapping[gold_name] = our_name

    return mapping


def run_ours_benchmark(
    gold_dir: Path,
    checkpoint: str,
    zarr_dir: str,
    device: str = "cuda:0",
    svd_path: str = "embeddings/svd_512_v6.npz",
    batch_size: int = 64,
) -> dict:
    """Run our model's MP head on gold standard data and evaluate.

    Loads gold standard TIFFs and segmentation masks directly, extracts
    patches in our model's input format, and runs inference.

    Requires: main .venv with deepcelltypes installed
    """
    import torch
    import tifffile
    from scipy.ndimage import center_of_mass
    from deepcell_types.training.config import (
        TissueNetConfig, extract_patch, compute_distance_transform,
    )
    from deepcell_types.model import create_model

    config = TissueNetConfig(zarr_dir)

    # Build channel name mapping
    ch_name_map = _build_gold_channel_name_map(config.marker2idx)

    # Load marker embeddings
    marker_embeddings = config.load_marker_embeddings_array(svd_path=svd_path)

    # Load model
    print(f"Loading model from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Auto-detect resnet_base_channels from checkpoint
    resnet_channels = 32
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        for k, v in state_dict.items():
            if k == "channel_encoder.stem.0.bias":
                resnet_channels = v.shape[0]
                break

    # Auto-detect tumor head
    has_tumor_head = False
    if isinstance(ckpt, dict) and "model" in ckpt:
        has_tumor_head = any(k.startswith("tumor_head.") for k in ckpt["model"])

    # Auto-detect mean_intensity_mode from ckpt keys
    mean_intensity_mode = "none"
    if isinstance(ckpt, dict) and "model" in ckpt:
        has_cls = any(k.startswith("intensity_cls_branch.") for k in ckpt["model"])
        has_pch = any(k.startswith("intensity_per_channel_proj.") for k in ckpt["model"])
        if has_cls and has_pch:
            mean_intensity_mode = "both"
        elif has_cls:
            mean_intensity_mode = "cls_residual"
        elif has_pch:
            mean_intensity_mode = "per_channel"
    print(f"  mean_intensity_mode={mean_intensity_mode}")

    model = create_model(
        config, marker_embeddings, d_model=256,
        resnet_base_channels=resnet_channels,
        tumor_head=has_tumor_head,
        mean_intensity_mode=mean_intensity_mode,
    )
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"  resnet_base_channels={resnet_channels}, tumor_head={has_tumor_head}")

    # Load gold standard labels from the master CSV
    gs_csv = gold_dir / "gold_standard_groundtruth.csv"
    if not gs_csv.exists():
        # Search one level down
        for child in gold_dir.iterdir():
            if child.name == "gold_standard_groundtruth.csv":
                gs_csv = child
                break
    gs_df = pd.read_csv(gs_csv)
    print(f"Gold standard: {len(gs_df)} annotations across "
          f"{gs_df['dataset'].nunique()} datasets, {gs_df['fov'].nunique()} FOVs")

    all_predictions = {}  # (fov, cell_id, channel) -> score
    total_cells_processed = 0
    total_channels_mapped = 0
    total_channels_unmapped = 0

    for (dataset_name, fov_name), fov_group in gs_df.groupby(["dataset", "fov"]):
        print(f"\n--- {dataset_name}/{fov_name} ---")
        cell_ids = sorted(fov_group["cell_id"].unique())
        gold_channels = sorted(fov_group["channel"].unique())
        print(f"  {len(cell_ids)} cells, {len(gold_channels)} channels")

        # Find FOV directory and mask file
        fov_dir = gold_dir / dataset_name / "fovs" / fov_name
        mask_dir = gold_dir / dataset_name / "masks"

        if not fov_dir.exists():
            print(f"  WARNING: FOV directory not found: {fov_dir}")
            continue

        # Find mask file (try various extensions and fuzzy matching)
        mask_path = None
        for ext in [".ome.tif", ".tif", ".tiff", ".ome.tiff"]:
            candidate = mask_dir / (fov_name + ext)
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            # Fuzzy match: mask filename may differ from FOV name
            # Strategy: extract a unique ID prefix from the FOV name and match
            # against mask filenames. Handles cases like:
            # - FOV "_image" vs mask "_imagefeature_0.ome.tif"
            # - FOV "..._data.tif_image" vs mask "..._data.ome.tif"
            for mask_file in mask_dir.iterdir():
                if not mask_file.is_file():
                    continue
                mf = mask_file.name
                # Check if FOV name is a prefix of mask filename
                if mf.startswith(fov_name):
                    mask_path = mask_file
                    break
                # Strip common suffixes to find shared stem
                # FOV: "..._component_data.tif_image" -> "..._component_data"
                # Mask: "..._component_data.ome.tif" -> "..._component_data"
                fov_stem = fov_name
                for suffix in [".tif_image", "_image", ".tif", ".tiff"]:
                    if fov_stem.endswith(suffix):
                        fov_stem = fov_stem[:-len(suffix)]
                        break
                mask_stem = mask_file.name
                for suffix in [".ome.tif", ".ome.tiff", ".tif", ".tiff"]:
                    if mask_stem.endswith(suffix):
                        mask_stem = mask_stem[:-len(suffix)]
                        break
                if fov_stem and mask_stem.startswith(fov_stem):
                    mask_path = mask_file
                    break
                if fov_stem and fov_stem.startswith(mask_stem):
                    mask_path = mask_file
                    break
        if mask_path is None:
            print(f"  WARNING: Mask not found for {fov_name}")
            continue

        # Load mask
        mask = tifffile.imread(str(mask_path))
        # Some masks have extra leading dimensions, e.g. (1, H, W) — squeeze
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        print(f"  Mask shape: {mask.shape}")

        # Load channel TIFFs and build raw image stack
        # Only load channels that map to our vocabulary
        channel_names = []  # our vocabulary names
        gold_to_our = {}  # gold_channel -> (index_in_stack, our_name)
        channel_images = []
        for gold_ch in sorted(set(ch for ch in gold_channels)):
            our_name = ch_name_map.get(gold_ch)
            if our_name is None or our_name not in config.marker2idx:
                total_channels_unmapped += 1
                continue

            # Find the TIFF file for this channel
            tiff_path = None
            for ext in [".tiff", ".ome.tif", ".tif", ".ome.tiff"]:
                candidate = fov_dir / (gold_ch + ext)
                if candidate.exists():
                    tiff_path = candidate
                    break
            if tiff_path is None:
                # Try our_name too
                for ext in [".tiff", ".ome.tif", ".tif", ".ome.tiff"]:
                    candidate = fov_dir / (our_name + ext)
                    if candidate.exists():
                        tiff_path = candidate
                        break
            if tiff_path is None:
                print(f"    WARNING: TIFF not found for channel {gold_ch}")
                total_channels_unmapped += 1
                continue

            img = tifffile.imread(str(tiff_path)).astype(np.float32)
            # Normalize to [0, 1] range
            img_max = img.max()
            if img_max > 0:
                img = img / img_max

            gold_to_our[gold_ch] = (len(channel_images), our_name)
            channel_names.append(our_name)
            channel_images.append(img)
            total_channels_mapped += 1

        if not channel_images:
            print(f"  No mapped channels, skipping")
            continue

        # Stack into (C, H, W) raw array
        raw = np.stack(channel_images, axis=0)  # (C, H, W)
        n_channels = raw.shape[0]
        print(f"  Loaded {n_channels} channels: {channel_names}")

        # Build channel index tensor
        ch_indices = [config.marker2idx[name] for name in channel_names]

        # Pad to MAX_NUM_CHANNELS
        max_ch = config.MAX_NUM_CHANNELS
        padded_ch_indices = ch_indices + [-1] * (max_ch - n_channels)
        ch_idx_tensor = torch.tensor(padded_ch_indices, dtype=torch.long)

        # Compute centroids for cells in this FOV
        cell_centroids = {}
        for cid in cell_ids:
            coords = np.argwhere(mask == cid)
            if len(coords) == 0:
                continue
            centroid = coords.mean(axis=0)  # (row, col)
            cell_centroids[cid] = (float(centroid[0]), float(centroid[1]))

        print(f"  Found centroids for {len(cell_centroids)}/{len(cell_ids)} cells")

        # Process cells in batches
        cell_list = list(cell_centroids.items())
        crop_size = config.CROP_SIZE
        output_size = config.OUTPUT_SIZE

        for batch_start in range(0, len(cell_list), batch_size):
            batch_cells = cell_list[batch_start:batch_start + batch_size]
            samples = []
            spatials = []

            for cell_id, centroid in batch_cells:
                # Extract patch using our standard function
                raw_masked, spatial_context = extract_patch(
                    raw, mask, centroid, cell_id,
                    crop_size=crop_size, output_size=output_size,
                )
                # raw_masked: (C, H, W), spatial_context: (3, H, W)

                # Pad channels to MAX_NUM_CHANNELS
                if n_channels < max_ch:
                    pad_shape = (max_ch - n_channels, output_size, output_size)
                    raw_masked = np.concatenate(
                        [raw_masked, np.zeros(pad_shape, dtype=np.float32)], axis=0
                    )

                samples.append(raw_masked)
                spatials.append(spatial_context)

            # Stack into batch tensors
            B = len(samples)
            sample_t = torch.from_numpy(np.stack(samples)).unsqueeze(2).to(device)  # (B, C_max, 1, H, W)
            spatial_t = torch.from_numpy(np.stack(spatials)).to(device)  # (B, 3, H, W)
            ch_idx_batch = ch_idx_tensor.unsqueeze(0).expand(B, -1).to(device)  # (B, C_max)
            mask_batch = (ch_idx_batch == -1)  # (B, C_max)

            with torch.no_grad():
                outputs = model(sample_t, spatial_t, ch_idx_batch, mask_batch)
                mp_logits = outputs[2]  # (B, C_max)
                mp_probs = torch.sigmoid(mp_logits).cpu().numpy()  # (B, C_max)

            # Store predictions keyed by (fov, cell_id, gold_channel_name)
            for i, (cell_id, _) in enumerate(batch_cells):
                for gold_ch, (ch_stack_idx, our_name) in gold_to_our.items():
                    score = float(mp_probs[i, ch_stack_idx])
                    all_predictions[(fov_name, cell_id, gold_ch)] = score

            total_cells_processed += B

        if total_cells_processed % 5000 < batch_size:
            print(f"  Processed {total_cells_processed} cells so far...")

    print(f"\nTotal cells processed: {total_cells_processed}")
    print(f"Total channels mapped: {total_channels_mapped}, unmapped: {total_channels_unmapped}")
    print(f"Total predictions: {len(all_predictions)}")

    # Optionally persist raw predictions so downstream callers can post-hoc
    # tune per-marker thresholds (single-float threshold is too coarse on the
    # gold standard, where most markers have heavily imbalanced positivity).
    preds_csv = os.environ.get("DCT_GOLD_PREDS_CSV")
    if preds_csv:
        # Persist raw (fov, cell_id, channel, pred_score) so downstream
        # callers can post-hoc tune per-marker thresholds (single-float
        # threshold is too coarse on the gold standard, where most markers
        # have heavily imbalanced positivity).
        rows = [
            {"fov": fov, "cell_id": cell_id,
             "channel": marker, "pred_score": float(score)}
            for (fov, cell_id, marker), score in all_predictions.items()
        ]
        pd.DataFrame(rows).to_csv(preds_csv, index=False)
        print(f"Raw predictions → {preds_csv}")

    # Build labels DataFrame for evaluate_predictions (expects fov, cell_index, marker, label)
    labels_for_eval = gs_df.rename(columns={
        "cell_id": "cell_index",
        "channel": "marker",
        "activity": "label",
    })[["fov", "cell_index", "marker", "label"]]

    results = evaluate_predictions(labels_for_eval, all_predictions)
    results["method"] = "ours"
    results["n_cells_processed"] = total_cells_processed
    results["n_channels_mapped"] = total_channels_mapped
    results["n_channels_unmapped"] = total_channels_unmapped
    return results


def compare_results(nimbus_path: str, ours_path: str):
    """Compare saved results from both methods."""
    with open(nimbus_path) as f:
        nimbus = json.load(f)
    with open(ours_path) as f:
        ours = json.load(f)

    print("\n" + "=" * 70)
    print("MARKER POSITIVITY BENCHMARK — Pan-Multiplex Gold Standard")
    print("Reduction: deepcell_types.training.utils.summarize_mp_per_marker (shared)")
    print("=" * 70)
    print(f"\n{'Metric':<32} {'Nimbus':>12} {'Ours':>12}")
    print("-" * 56)

    # Headline keys come from the shared helper (apples-to-apples)
    headline_metrics = [
        "mp_macro_f1",
        "mp_micro_f1",
        "mp_macro_precision",
        "mp_macro_recall",
        "mp_macro_accuracy",
        "mp_num_markers",
    ]
    for metric in headline_metrics:
        n_val = nimbus.get("overall", {}).get(metric, "N/A")
        o_val = ours.get("overall", {}).get(metric, "N/A")
        n_str = f"{n_val:.4f}" if isinstance(n_val, (int, float)) else str(n_val)
        o_str = f"{o_val:.4f}" if isinstance(o_val, (int, float)) else str(o_val)
        print(f"{metric:<32} {n_str:>12} {o_str:>12}")

    print(f"\n{'Legacy (sklearn flat pool)':<32}")
    print("-" * 56)
    for metric in ["accuracy", "f1", "precision", "recall", "auroc"]:
        n_val = nimbus.get("overall", {}).get(metric, "N/A")
        o_val = ours.get("overall", {}).get(metric, "N/A")
        n_str = f"{n_val:.4f}" if isinstance(n_val, (int, float)) else str(n_val)
        o_str = f"{o_val:.4f}" if isinstance(o_val, (int, float)) else str(o_val)
        print(f"{metric:<32} {n_str:>12} {o_str:>12}")

    n_samples = nimbus.get("overall", {}).get("n_samples", "?")
    o_samples = ours.get("overall", {}).get("n_samples", "?")
    print(f"{'n_samples':<20} {n_samples:>12} {o_samples:>12}")

    # Per-marker comparison (top markers by sample count)
    print(f"\n{'Marker':<20} {'Nimbus F1':>12} {'Ours F1':>12} {'N':>8}")
    print("-" * 52)

    nimbus_markers = nimbus.get("per_marker", {})
    ours_markers = ours.get("per_marker", {})
    all_markers = sorted(
        set(nimbus_markers.keys()) | set(ours_markers.keys()),
        key=lambda m: nimbus_markers.get(m, {}).get("n_samples", 0),
        reverse=True,
    )
    for marker in all_markers[:20]:
        n_f1 = nimbus_markers.get(marker, {}).get("f1", "-")
        o_f1 = ours_markers.get(marker, {}).get("f1", "-")
        n_n = nimbus_markers.get(marker, {}).get("n_samples", 0)
        n_str = f"{n_f1:.3f}" if isinstance(n_f1, float) else str(n_f1)
        o_str = f"{o_f1:.3f}" if isinstance(o_f1, float) else str(o_f1)
        print(f"{marker:<20} {n_str:>12} {o_str:>12} {n_n:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark marker positivity on Pan-Multiplex Gold Standard"
    )
    parser.add_argument(
        "--method",
        choices=["nimbus", "ours", "compare"],
        required=True,
        help="Which method to benchmark (or 'compare' to compare saved results)",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default="data/gold_standard",
        help="Path to extracted gold standard data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to our model checkpoint (required for --method ours)",
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        default=str(Path(os.environ.get("DATA_DIR", "")) / "tissuenet-caitlin-labels.zarr"),
        help="Path to zarr archive (for --method ours)",
    )
    parser.add_argument(
        "--svd_path",
        type=str,
        default="embeddings/svd_512_v6.npz",
        help="Path to SVD embeddings (for --method ours)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )
    # Compare mode args
    parser.add_argument("--nimbus_results", type=str, default=None)
    parser.add_argument("--ours_results", type=str, default=None)

    args = parser.parse_args()
    gold_dir = Path(args.gold_dir)

    if args.method == "compare":
        if not args.nimbus_results or not args.ours_results:
            parser.error("--nimbus_results and --ours_results required for compare mode")
        compare_results(args.nimbus_results, args.ours_results)
        return

    if args.method == "nimbus":
        results = run_nimbus_benchmark(gold_dir, device=args.device)
    elif args.method == "ours":
        if not args.checkpoint:
            parser.error("--checkpoint required for --method ours")
        results = run_ours_benchmark(
            gold_dir,
            checkpoint=args.checkpoint,
            zarr_dir=args.zarr_dir,
            device=args.device,
            svd_path=args.svd_path,
        )

    # Print results
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    overall = results["overall"]
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.method}")
    print(f"{'='*60}")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    output_path = args.output or f"output/gold_standard_{args.method}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
