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
DATA_DIR = Path(os.environ.get("DATA_DIR", ""))

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.baseline_features import (
    compute_baseline_metrics,
    save_baseline_predictions,
    extract_features_from_zarr,
)


@click.command()
@click.option("--model_name", type=str, default="xgb_baseline_0")
@click.option("--enable_wandb", type=bool, default=False)
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
def main(
    model_name: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    split_file: str,
    features_cache: str,
):
    """Train XGBoost baseline for cell type classification."""
    # Initialize wandb if enabled
    if enable_wandb:
        import wandb

        wandb.login()
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "deepcell-types"),
            dir="wandb_tmp",
            job_type="train",
            name=f"{model_name}_xgboost",
            config={
                "model_type": "xgboost",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            },
        )

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
    )

    X_train, y_train = data["X_train"], data["y_train"]
    train_fov_names = data["train_fov_names"]
    X_test, y_test = data["X_val"], data["y_val"]
    test_dataset_names = data["val_dataset_names"]
    test_fov_names = data["val_fov_names"]
    test_cell_indices = data["val_cell_indices"]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

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
    model.fit(
        X_inner_train,
        y_inner_train,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=True,
    )
    print(
        f"  Best iteration: {model.best_iteration}, Best score (mlogloss): {model.best_score:.6f}"
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

    # Log to wandb if enabled
    if enable_wandb:
        wandb.log(
            {
                "test/macro_accuracy": metrics["macro_accuracy"],
                "test/weighted_accuracy": metrics["weighted_accuracy"],
                "test/macro_f1": metrics["macro_f1"],
                "test/weighted_f1": metrics["weighted_f1"],
            }
        )

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
    )

    if enable_wandb:
        run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
