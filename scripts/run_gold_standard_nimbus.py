#!/usr/bin/env python3
"""Run Nimbus on the Pan-Multiplex Gold Standard and evaluate marker positivity.

The gold standard contains ~1.1M expert annotations (0=neg, 1=pos, 2/3=ambiguous)
across 5 subsets: codex_colon, mibi_breast, mibi_decidua, vectra_colon, vectra_pancreas.

Usage:
    # Download first:
    bash scripts/download_gold_standard.sh

    # Run:
    python scripts/run_gold_standard_nimbus.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", default="data/gold_standard/gold_standard_labelled")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="output/gold_standard_nimbus.json")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Subsets to run (default: all available)")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)

    # Load ground truth
    gt_path = gold_dir / "gold_standard_groundtruth.csv"
    print(f"Loading ground truth from {gt_path}...")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df)} annotations across {gt_df['dataset'].nunique()} subsets")
    print(f"  Activity distribution: {dict(gt_df['activity'].value_counts())}")

    # Discover subsets
    all_subsets = [d.name for d in gold_dir.iterdir()
                   if d.is_dir() and (d / "fovs").exists()]
    if args.subsets:
        subsets = [s for s in args.subsets if s in all_subsets]
    else:
        subsets = all_subsets
    print(f"Subsets to process: {subsets}")

    # Import Nimbus
    from nimbus_inference.nimbus import Nimbus, prep_naming_convention
    from nimbus_inference.utils import MultiplexDataset

    all_predictions = {}  # (dataset, fov, cell_id, channel) -> score

    for subset in subsets:
        subset_dir = gold_dir / subset
        fovs_dir = subset_dir / "fovs"
        masks_dir = subset_dir / "masks"

        fov_paths = sorted([str(p) for p in fovs_dir.iterdir() if p.is_dir()])
        if not fov_paths:
            print(f"  {subset}: no FOVs found, skipping")
            continue

        # Detect file suffix
        sample_fov = Path(fov_paths[0])
        sample_files = list(sample_fov.iterdir())
        if sample_files:
            suffix = "".join(sample_files[0].suffixes)  # e.g. ".ome.tif"
        else:
            suffix = ".tiff"

        # Detect mask suffix
        mask_files = list(masks_dir.iterdir()) if masks_dir.exists() else []
        if mask_files:
            mask_suffix = "".join(mask_files[0].suffixes)
        else:
            print(f"  {subset}: no masks found, skipping")
            continue

        print(f"\n--- {subset} ({len(fov_paths)} FOVs, suffix={suffix}) ---")

        # Build segmentation naming convention
        def make_seg_fn(masks_dir, mask_suffix):
            def seg_fn(fov_path):
                fov_name = Path(fov_path).name
                return str(masks_dir / f"{fov_name}{mask_suffix}")
            return seg_fn

        seg_fn = make_seg_fn(masks_dir, mask_suffix)

        # Verify mask exists for first FOV
        test_mask = seg_fn(fov_paths[0])
        if not Path(test_mask).exists():
            print(f"  WARNING: mask not found at {test_mask}, skipping")
            continue

        dataset = MultiplexDataset(
            fov_paths=fov_paths,
            suffix=suffix,
            include_channels=None,
            segmentation_naming_convention=seg_fn,
        )

        # Get image shape from first FOV
        import tifffile
        first_img = tifffile.imread(sample_files[0])
        input_shape = list(first_img.shape[:2])
        print(f"  Input shape: {input_shape}")

        # Nimbus only accepts "cuda"/"cpu"/"mps", not "cuda:N"
        nimbus_device = "cuda" if "cuda" in args.device else args.device
        nimbus = Nimbus(
            dataset=dataset,
            output_dir=str(gold_dir / "nimbus_output" / subset),
            save_predictions=False,
            batch_size=1,
            test_time_aug=True,
            input_shape=input_shape,
            device=nimbus_device,
        )

        nimbus.check_inputs()

        # Run inference (V1 handles normalization automatically)
        cell_table = nimbus.predict_fovs()
        print(f"  Nimbus predictions: {len(cell_table)} cells")

        # Store predictions keyed by (dataset, fov, cell_id, channel)
        cell_table.rename(columns={"label": "cell_id"}, inplace=True)
        meta_cols = {"cell_id", "fov"}
        marker_cols = [c for c in cell_table.columns if c not in meta_cols]

        for _, row in cell_table.iterrows():
            fov = row["fov"]
            cell_id = int(row["cell_id"])
            for marker in marker_cols:
                all_predictions[(subset, fov, cell_id, marker)] = float(row[marker])

        print(f"  Total predictions so far: {len(all_predictions)}")

    # Evaluate against ground truth
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    # Filter GT to non-ambiguous (activity 0 or 1)
    gt_valid = gt_df[gt_df["activity"].isin([0, 1])].copy()
    print(f"Valid GT annotations (0 or 1): {len(gt_valid)}")

    y_true = []
    y_pred = []
    y_score = []
    markers = []
    datasets = []

    for _, row in tqdm(gt_valid.iterrows(), total=len(gt_valid), desc="Matching"):
        key = (row["dataset"], row["fov"], int(row["cell_id"]), row["channel"])
        if key in all_predictions:
            y_true.append(row["activity"])
            score = all_predictions[key]
            y_score.append(score)
            y_pred.append(1 if score >= 0.5 else 0)
            markers.append(row["channel"])
            datasets.append(row["dataset"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    markers = np.array(markers)
    datasets = np.array(datasets)

    print(f"Matched predictions: {len(y_true)} / {len(gt_valid)}")

    if len(y_true) == 0:
        print("ERROR: No matching predictions found")
        return

    # Overall metrics
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_matched": int(len(y_true)),
        "n_total_gt": int(len(gt_valid)),
    }

    print(f"\nOverall:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-subset metrics
    per_subset = {}
    for ds in np.unique(datasets):
        mask = datasets == ds
        yt, yp = y_true[mask], y_pred[mask]
        per_subset[ds] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "n_samples": int(len(yt)),
        }
        print(f"  {ds}: acc={per_subset[ds]['accuracy']:.4f}, f1={per_subset[ds]['f1']:.4f}, n={per_subset[ds]['n_samples']}")

    # Per-marker metrics (top 20)
    per_marker = {}
    for m in np.unique(markers):
        mask = markers == m
        yt, yp = y_true[mask], y_pred[mask]
        per_marker[m] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "n_samples": int(len(yt)),
        }

    print(f"\nTop 20 markers by sample count:")
    for m, v in sorted(per_marker.items(), key=lambda x: x[1]["n_samples"], reverse=True)[:20]:
        print(f"  {m:<20} acc={v['accuracy']:.3f}  f1={v['f1']:.3f}  n={v['n_samples']}")

    # Save results
    results = {
        "method": "nimbus",
        "overall": overall,
        "per_subset": per_subset,
        "per_marker": per_marker,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
