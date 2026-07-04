"""
XGBoost baseline for cell type classification.

Uses mean intensity per channel as features to classify cell types.
This provides a simple baseline to compare against the transformer-based model.
"""

import json
import os
import click
import numpy as np
from pathlib import Path
from typing import Tuple
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

# Default data directory from environment
DATA_DIR = Path(
    os.environ.get("DEEPCELL_TYPES_ZARR_PATH") or os.environ.get("DATA_DIR", "")
)

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.baseline_features import (
    compute_baseline_metrics,
    save_baseline_predictions,
    extract_features_from_zarr,
)
from deepcell_types.training.samplers import compute_sample_weights_dct


@click.command()
@click.option("--model_name", type=str, default="xgb_baseline_0")
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR),
)
@click.option(
    "--skip_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to skip",
)
@click.option(
    "--keep_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to keep (exclusive with skip_datasets)",
)
@click.option(
    "--n_estimators",
    type=int,
    default=100,
    help="Number of boosting rounds",
)
@click.option(
    "--max_depth",
    type=int,
    default=6,
    help="Maximum tree depth",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.1,
    help="Learning rate (eta)",
)
@click.option(
    "--split_file",
    type=str,
    default=None,
    help="Path to pre-computed FOV split JSON (required)",
)
@click.option(
    "--features_cache",
    type=str,
    default=None,
    help="Path to cache extracted features (.npz). Reuses cache if it exists.",
)
@click.option(
    "--val_split_file",
    type=str,
    default=None,
    help="If set, train on the FULL --split_file train and early-stop on the "
    "'val' FOVs of THIS file (capped to 200k cells at seed 42, mirroring "
    "deepcell_types/training/dataloader.py:269-273 max_val_samples) used as the "
    "XGBoost eval_set. The reported set stays --split_file 'val'. When unset, "
    "keeps the legacy 10% FOV-grouped inner-val carve for early stopping.",
)
@click.option(
    "--class_balance",
    type=click.Choice(["dct", "none"]),
    default="dct",
    help="Training class-balancing. 'dct' (default): pass per-row "
    "sample_weight to fit() using the DCT sampler weights (sqrt-inverse-"
    "frequency with a 1000-count floor), identical to the main DeepCell-Types "
    "model and the other baselines (shared comparison footing). 'none' "
    "(faithful XGBoost, ablation): no class weighting.",
)
def main(
    model_name: str,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    split_file: str,
    features_cache: str,
    val_split_file: str,
    class_balance: str,
):
    """Train XGBoost baseline for cell type classification."""
    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError(
            "--split_file is required. Generate one with: python -m scripts.generate_splits"
        )

    # Relax the strict split-coverage check ONLY for the canonical-val smoke
    # path (--val_split_file with --keep_datasets restricting extraction to a
    # FOV subset). A real run (keep_datasets is None) keeps strict=True.
    _relax_strict = (val_split_file is not None) and (keep_datasets is not None)

    # Extract features directly from zarr (fast path, no DataLoader overhead).
    # ``missing_value=np.nan`` so absent markers route through XGBoost's
    # ``missing=NaN`` default-direction logic at every split — without it
    # absent markers would be conflated with real channels whose mean
    # intensity happens to be 0.0.
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        cache_path=features_cache,
        missing_value=np.nan,
        strict_split=not _relax_strict,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    train_fov_names = data["train_fov_names"]
    X_test, y_test = data["X_val"], data["y_val"]
    test_dataset_names = data["val_dataset_names"]
    test_fov_names = data["val_fov_names"]
    test_cell_indices = data["val_cell_indices"]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    if val_split_file is not None:
        # === Canonical external-val protocol (mirrors the main DCT model) ===
        # Train on the FULL --split_file train; early-stop on the 'val' FOVs of
        # --val_split_file (capped to 200k cells at seed 42, mirroring
        # dataloader.py:269-273 max_val_samples) as the XGBoost eval_set; report
        # on --split_file 'val'. Contiguous label space over the TRAIN classes
        # (XGBClassifier requires y in exactly [0..K-1]); selection-val / test
        # rows whose label is absent from train are dropped (at full 1722-FOV
        # scale every class is in train, so this is a smoke-scale no-op).
        train_unique = np.sort(np.unique(y_train))
        label_to_compact = {int(orig): i for i, orig in enumerate(train_unique)}
        compact_to_label = {i: int(orig) for orig, i in label_to_compact.items()}
        n_classes_compact = len(train_unique)
        compact_ct2idx = {
            name: label_to_compact[idx]
            for name, idx in dct_config.ct2idx.items()
            if idx in label_to_compact
        }
        X_inner_train = X_train
        y_inner_train = np.array([label_to_compact[y] for y in y_train])

        # Extract the selection-val (the 'val' FOVs of --val_split_file).
        val_features_cache = (
            str(Path(features_cache).with_suffix(".valsel.npz"))
            if features_cache
            else None
        )
        print(f"\nExtracting selection-val features from {val_split_file}...")
        sel_data = extract_features_from_zarr(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            split_file=val_split_file,
            skip_datasets=skip_datasets,
            keep_datasets=keep_datasets,
            cache_path=val_features_cache,
            missing_value=np.nan,
            strict_split=not _relax_strict,
        )
        X_sel = sel_data["X_val"]
        y_sel_orig = sel_data["y_val"]
        sel_mask = np.isin(y_sel_orig, train_unique)
        X_sel = X_sel[sel_mask]
        y_sel = np.array([label_to_compact[y] for y in y_sel_orig[sel_mask]])
        # Cap to 200k cells EXACTLY as the main model caps its val set
        # (dataloader.py:269-273: np.random.default_rng(42).choice, no
        # replacement). Seed 42 is the DCT training seed.
        n_val_cells = X_sel.shape[0]
        cap = min(200000, n_val_cells)
        rng = np.random.default_rng(42)
        sel_idx = rng.choice(n_val_cells, size=cap, replace=False)
        X_inner_val = X_sel[sel_idx]
        y_inner_val = y_sel[sel_idx]

        # Filter the test (report) rows to train classes + compact-map.
        test_mask = np.isin(y_test, train_unique)
        n_dropped_test = int((~test_mask).sum())
        if n_dropped_test:
            print(
                f"  Dropped {n_dropped_test} test rows with labels absent from "
                f"train (smoke-scale artifact)."
            )
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        y_test_compact = np.array([label_to_compact[y] for y in y_test])
        test_dataset_names = [n for n, k in zip(test_dataset_names, test_mask) if k]
        test_fov_names = [n for n, k in zip(test_fov_names, test_mask) if k]
        test_cell_indices = [c for c, k in zip(test_cell_indices, test_mask) if k]

        n_train_fovs = len(np.unique(np.asarray(train_fov_names)))
        n_test_fovs = len(set(test_fov_names))
        print(
            f"Selection protocol: train on FULL --split_file train "
            f"({n_train_fovs} FOVs, {len(X_inner_train)} cells, no inner carve); "
            f"selection-val (eval_set) from {val_split_file} capped to {cap} of "
            f"{n_val_cells} cells; test (report) = {n_test_fovs} FOVs, "
            f"{len(X_test)} cells."
        )
    else:
        # === BEGIN legacy 10% FOV-grouped inner-val carve (back-compat) ===
        # Remap labels to contiguous 0-indexed (XGBoost requires this).
        # Train labels must be contiguous [0..n_train-1]; test-only labels appended after.
        train_unique = np.sort(np.unique(y_train))
        label_to_compact = {orig: i for i, orig in enumerate(train_unique)}
        next_idx = len(train_unique)
        for label in np.sort(np.unique(y_test)):
            if label not in label_to_compact:
                label_to_compact[label] = next_idx
                next_idx += 1
        compact_to_label = {i: orig for orig, i in label_to_compact.items()}
        n_classes_compact = next_idx
        compact_ct2idx = {
            name: label_to_compact[idx]
            for name, idx in dct_config.ct2idx.items()
            if idx in label_to_compact
        }
        y_train_compact = np.array([label_to_compact[y] for y in y_train])
        y_test_compact = np.array([label_to_compact[y] for y in y_test])
        print(f"Unique cell types in data: {n_classes_compact} (of {num_classes} total)")

        # Carve a FOV-grouped inner-validation set out of training data for early stopping.
        # Test set MUST NOT be used as eval_set — that leaks test signal into tree-count selection.
        train_fov_array = np.asarray(train_fov_names)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        inner_train_idx, inner_val_idx = next(
            gss.split(X_train, y_train_compact, groups=train_fov_array)
        )
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train_compact[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train_compact[inner_val_idx]
        print(
            f"Inner-val (FOV-grouped, disjoint from test): {len(inner_val_idx)} samples from "
            f"{len(np.unique(train_fov_array[inner_val_idx]))} FOVs"
        )

        # Re-tighten compact labels to be strictly contiguous 0..K-1 over the inner-train
        # label set. Modern xgboost.sklearn.XGBClassifier rejects targets whose unique values
        # don't equal [0..K-1] exactly. The initial compact mapping above is over
        # union(train_unique, test_unique), but GroupShuffleSplit can leave compact labels
        # with zero examples in y_inner_train, producing holes. We re-encode here and drop
        # any inner-val / test rows with labels absent from inner_train (this only fires on
        # tiny smoke-sized subsets — at full scale every train label is in inner_train).
        train_present = np.unique(y_inner_train)
        if (
            len(train_present) != n_classes_compact
            or train_present[0] != 0
            or train_present[-1] != len(train_present) - 1
        ):
            inner_remap = {int(orig): i for i, orig in enumerate(train_present)}
            val_mask = np.isin(y_inner_val, train_present)
            test_mask = np.isin(y_test_compact, train_present)
            n_dropped_val = int((~val_mask).sum())
            n_dropped_test = int((~test_mask).sum())
            if n_dropped_val or n_dropped_test:
                print(
                    f"  Dropped {n_dropped_val} inner-val + {n_dropped_test} test rows "
                    f"with labels absent from inner-train (smoke-scale artifact)."
                )
            X_inner_val = X_inner_val[val_mask]
            y_inner_val = y_inner_val[val_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            y_test_compact = y_test_compact[test_mask]
            test_dataset_names = [
                n for n, keep in zip(test_dataset_names, test_mask) if keep
            ]
            test_fov_names = [n for n, keep in zip(test_fov_names, test_mask) if keep]
            test_cell_indices = [c for c, keep in zip(test_cell_indices, test_mask) if keep]
            remap_fn = np.vectorize(inner_remap.get, otypes=[np.int64])
            y_inner_train = remap_fn(y_inner_train)
            y_inner_val = remap_fn(y_inner_val)
            y_test_compact = remap_fn(y_test_compact)
            compact_to_label = {
                i: compact_to_label[int(orig)] for i, orig in enumerate(train_present)
            }
            compact_ct2idx = {
                name: inner_remap[orig]
                for name, orig in compact_ct2idx.items()
                if orig in inner_remap
            }
            n_classes_compact = len(train_present)

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    early_stopping_rounds = max(10, n_estimators // 10)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes_compact,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=early_stopping_rounds,
    )

    print(f"  Early stopping: {early_stopping_rounds} rounds (inner-val FOV-grouped)")
    # 'dct' balancing: weight each training row by the DCT sampler scheme
    # (sqrt-inverse-frequency, 1000-count floor) — the tree analog (per-row
    # gradient weighting) of the WeightedRandomSampler the neural baselines use,
    # equivalent in expectation but not identical. normalize=True rescales to
    # mean 1 so the balancing does not also alter effective regularization (see
    # compute_sample_weights_dct). 'none' = faithful unweighted XGBoost.
    sample_weight = (
        compute_sample_weights_dct(y_inner_train, normalize=True)
        if class_balance == "dct"
        else None
    )
    print(f"  Class balancing: {class_balance}")
    model.fit(
        X_inner_train,
        y_inner_train,
        sample_weight=sample_weight,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=True,
    )
    print(
        f"  Best iteration: {model.best_iteration}, Best score (mlogloss): {model.best_score:.6f}"
    )

    if val_split_file is not None:
        # Report the canonical selection signal: the hierarchical ct_macro_f1 on
        # the capped external selection-val at the early-stopped best_iteration
        # (same reduction the main model reports via LossesAndMetrics.compute,
        # metrics.py:399-419). Early stopping itself uses XGBoost's default
        # mlogloss on this same eval_set.
        sel_metrics = compute_baseline_metrics(
            y_inner_val,
            model.predict(X_inner_val),
            model.predict_proba(X_inner_val),
            n_classes_compact,
            hierarchy=CELL_TYPE_HIERARCHY,
            ct2idx=compact_ct2idx,
        )
        print(
            f"  selection ct_macro_f1={sel_metrics['macro_f1']:.4f} "
            f"(capped external val, at best_iteration)"
        )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_compact = model.predict(X_test)
    y_prob_compact = model.predict_proba(X_test)  # (N, n_model_classes)

    # Metrics on compact labels (contiguous 0-indexed, required by confusion_matrix).
    # Use shared hierarchy collapse (Tcell + Stromal) so XGBoost numbers are
    # directly comparable to the main model's LossesAndMetrics.compute() output.
    metrics = compute_baseline_metrics(
        y_test_compact,
        y_pred_compact,
        y_prob_compact,
        n_classes_compact,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    print("\nTest Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    # Save model
    model_path = Path(f"models/xgb_model_{model_name}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Save the label-space remap alongside the ckpt. Out-of-band evaluators
    # need this to align ``model.predict()`` outputs back to original
    # ``ct2idx`` indices — the post-remap → original_ct mapping is otherwise
    # only recoverable by replaying ``GroupShuffleSplit(test_size=0.1,
    # random_state=42)`` with the same train labels + FOV grouping, which
    # depends on data state that isn't shipped with the ckpt.
    idx2ct = {v: k for k, v in dct_config.ct2idx.items()}
    remap_meta = {
        "n_classes": int(n_classes_compact),
        "compact_to_orig_ct_idx": {int(k): int(v) for k, v in compact_to_label.items()},
        "compact_to_ct_name": {
            int(k): idx2ct.get(int(v), "?") for k, v in compact_to_label.items()
        },
        "gss_recipe": (
            "GroupShuffleSplit(test_size=0.1, random_state=42) on "
            "(X_train, y_train_compact, groups=train_fov_names)"
        ),
    }
    remap_path = model_path.with_suffix(".remap.json")
    with open(remap_path, "w") as f:
        json.dump(remap_meta, f, indent=2, sort_keys=True)
    print(f"Remap metadata saved to {remap_path}")

    # Map probabilities to ct2idx-sorted columns for saving
    # save_baseline_predictions expects y_prob with len(ct2idx) columns,
    # one per cell type sorted by ct2idx value
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob = np.zeros((len(y_test), len(dct_config.ct2idx)), dtype=np.float32)
    for compact_idx, orig_idx in compact_to_label.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]

    # Save predictions
    output_path = Path(f"output/{model_name}_xgb_prediction.csv")
    save_baseline_predictions(
        y_test,
        y_prob,
        test_cell_indices,
        test_dataset_names,
        test_fov_names,
        dct_config.ct2idx,
        output_path,
        run_metadata={"method": "xgboost", "class_balance": class_balance},
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
