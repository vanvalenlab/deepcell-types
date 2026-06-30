"""
MAPS baseline training and evaluation.

Implements the MAPS (Machine learning for Analysis of Proteomics in Spatial biology)
MLP classifier from the Mahmood Lab for cell type classification in multiplexed imaging data.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-44188-w
- Code: https://github.com/mahmoodlab/MAPS
"""

import os
import random
import click
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit

from .model import MAPSModel

# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", ""))

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.baseline_features import (
    compute_baseline_metrics,
    save_baseline_predictions,
    extract_features_from_zarr,
)
from deepcell_types.training.samplers import compute_sample_weights_dct


def set_seed(seed: int) -> None:
    """Match canonical mahmoodlab/MAPS trainer.set_seed (seed=7325111 default)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Per-sample weights for canonical mahmoodlab/MAPS class balancing:
    `weight_per_class = n / count(c)` (full inverse frequency).
    """
    class_counts = np.bincount(y)
    n = float(len(y))
    weight_per_class = np.where(class_counts > 0, n / class_counts, 0.0)
    sample_weights = weight_per_class[y]
    return sample_weights


def normalize_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    znorm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize per-cell features for the MAPS MLP.

    DeepCell Types stores preprocessed marker intensities in ``[0, 1]`` and
    appends raw pixel-count ``cellSize`` as the final MAPS feature. The default
    DCT adapter therefore applies train-set z-score statistics before the
    ``/255`` scaling so marker means and cell size remain on a controlled
    relative scale. ``znorm=False`` keeps the `/255`-only path available as an
    upstream-provenance ablation, but it is not the DCT-safe default.

    Args:
        X: (N, D) feature matrix
        mean: Optional pre-computed mean (for test set normalization)
        std: Optional pre-computed std (for test set normalization)
        znorm: If True, z-score with train stats before dividing by 255.

    Returns:
        X_norm: (N, D) normalized features
        mean: (D,) feature means
        std: (D,) feature stds
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    if znorm:
        X_norm = ((X - mean) / std) / 255.0
    else:
        X_norm = X / 255.0
    return X_norm, mean, std


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Args:
        model: MAPS model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)  # Use logits for loss computation
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: MAPS model
        X: (N, D) features (normalized)
        y: (N,) true labels
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    """
    model.eval()
    all_prob = []

    # Process in batches
    for i in range(0, len(X), batch_size):
        X_batch = X[i : i + batch_size].to(device)
        _, probs = model(X_batch)  # Use probs for evaluation
        all_prob.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_prob, axis=0)
    y_pred = y_prob.argmax(axis=1)

    return y, y_pred, y_prob


@click.command()
@click.option("--model_name", type=str, default="maps_0")
@click.option("--device_num", type=str, default="cuda:0")
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
    "--hidden_dim",
    type=int,
    default=512,
    help="Hidden layer dimension",
)
@click.option(
    "--dropout",
    type=float,
    default=0.25,
    help="Dropout rate (0.25 matches original experiment scripts)",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate",
)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training",
)
@click.option(
    "--max_epochs",
    type=int,
    default=500,
    help="Max training epochs (canonical mahmoodlab/MAPS trainer.py max_epochs=500). "
    "Best epoch selected on inner-validation loss; early-stops per --min_epochs/--patience.",
)
@click.option(
    "--min_epochs",
    type=int,
    default=250,
    help="Minimum epochs before early stopping may trigger "
    "(canonical mahmoodlab/MAPS trainer.py min_epochs=250).",
)
@click.option(
    "--patience",
    type=int,
    default=100,
    help="Early-stopping patience on inner-validation loss "
    "(canonical mahmoodlab/MAPS trainer.py patience=100).",
)
@click.option(
    "--znorm/--no_znorm",
    default=True,
    help="Apply a train-set z-score ((x-mu)/sigma) before the /255 (default on). "
    "On is the DCT-safe default for [0,1] marker means plus raw cellSize; "
    "--no_znorm keeps a /255-only provenance ablation available.",
)
@click.option(
    "--seed",
    type=int,
    default=7325111,
    help="Random seed (canonical mahmoodlab/MAPS default)",
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
    help="If set, train on the FULL --split_file train and select the best epoch "
    "on the 'val' FOVs of THIS file (capped to 200k cells at seed 42, mirroring "
    "deepcell_types/training/dataloader.py:269-273 max_val_samples), scored with "
    "the canonical hierarchical ct_macro_f1 (LossesAndMetrics.compute, "
    "metrics.py:399-419). The reported set stays --split_file 'val'. When unset, "
    "keeps the legacy 10% FOV-grouped inner-val carve + inner-val-loss selection.",
)
@click.option(
    "--class_balance",
    type=click.Choice(["dct", "full_inv_freq", "none"]),
    default="dct",
    help="Training class-balancing scheme for the WeightedRandomSampler. "
    "'dct' (default): DCT sampler — sqrt-inverse-frequency with a 1000-count "
    "floor, identical to the main DeepCell-Types model and the other baselines "
    "(shared comparison footing). 'full_inv_freq' (FAITHFUL, ablation): "
    "canonical mahmoodlab/MAPS full-inverse-frequency weights (weight=n/count). "
    "'none': uniform shuffle (ablation).",
)
def main(
    model_name: str,
    device_num: str,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    znorm: bool,
    seed: int,
    split_file: str,
    features_cache: str,
    val_split_file: str,
    class_balance: str,
):
    """Train MAPS baseline for cell type classification."""
    # Seed everything (canonical mahmoodlab/MAPS trainer.py:81-101)
    set_seed(seed)

    # Set device
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES
    # input_dim = NUM_MARKERS + 1 (cellSize column appended, paper-faithful per
    # canonical mahmoodlab/MAPS data_preprocessing/*.py and README §Datasets)
    input_dim = dct_config.NUM_MARKERS + 1

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(f"Input features: {input_dim} (mean intensity per channel + cellSize)")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError(
            "--split_file is required. Generate one with: python -m scripts.generate_splits"
        )

    # Relax the strict split-coverage check ONLY for the canonical-val smoke
    # path (--val_split_file with --keep_datasets restricting extraction to a
    # FOV subset). A real run (keep_datasets is None) keeps strict=True. Gated
    # on val_split_file so the legacy path's behaviour is unchanged.
    _relax_strict = (val_split_file is not None) and (keep_datasets is not None)

    # Extract features directly from zarr (fast path, no DataLoader overhead)
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        cache_path=features_cache,
        strict_split=not _relax_strict,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    train_fov_names = data["train_fov_names"]
    X_test, y_test = data["X_val"], data["y_val"]
    test_dataset_names = data["val_dataset_names"]
    test_fov_names = data["val_fov_names"]
    test_cell_indices = data["val_cell_indices"]

    # Append cellSize as the last feature column (canonical mahmoodlab/MAPS
    # data_preprocessing/*.py emits N markers + cellSize in dataset CSVs).
    train_cell_sizes = data["train_cell_sizes"].astype(np.float32).reshape(-1, 1)
    val_cell_sizes = data["val_cell_sizes"].astype(np.float32).reshape(-1, 1)
    X_train = np.concatenate([X_train, train_cell_sizes], axis=1)
    X_test = np.concatenate([X_test, val_cell_sizes], axis=1)

    print(
        f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features (markers + cellSize)"
    )
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Build a contiguous 0..N-1 output label space covering ALL archive cell
    # types (sorted by ct2idx), not just those present in train. Previously
    # this used ``np.sort(np.unique(y_train))`` (the canonical mahmoodlab/MAPS
    # recipe), which produces an output head with fewer dims than
    # ``len(ct2idx)`` whenever some archive classes have zero train support.
    # That structurally prevents the model from ever predicting those classes
    # at inference (their probability is identically zero), dragging macro-F1
    # down by 5–10 pp without a corresponding macro_accuracy gain.
    # Switching to the full sorted-ct2idx space adds a few unused output dims
    # whose loss gradient is always zero (no positive examples) — no harm,
    # and the saved ckpt is now a drop-in 51-class head that matches
    # CellSighter / "ours" predictions schema.
    sorted_ct = sorted(dct_config.ct2idx.values())
    label_to_compact = {orig: i for i, orig in enumerate(sorted_ct)}
    compact_to_label = {i: orig for orig, i in label_to_compact.items()}
    n_classes_compact = len(sorted_ct)
    compact_ct2idx = {
        name: label_to_compact[idx] for name, idx in dct_config.ct2idx.items()
    }
    y_train = np.array([label_to_compact[y] for y in y_train])
    y_test = np.array([label_to_compact[y] for y in y_test])
    n_train_unique = int(len(np.unique(y_train)))
    print(
        f"Output head: {n_classes_compact} classes (of {num_classes} total in ct2idx); "
        f"{n_train_unique} have train cells, {n_classes_compact - n_train_unique} have zero-train-support "
        f"and will receive no loss gradient."
    )

    # Build the model-selection validation set. Two protocols:
    #   (1) --val_split_file (canonical, matches the main DCT model): train on
    #       the FULL --split_file train (no inner carve) and select the best
    #       epoch on the 'val' FOVs of --val_split_file, capped to 200k cells at
    #       seed 42 (mirrors dataloader.py:269-273 max_val_samples) and scored by
    #       the hierarchical ct_macro_f1 (LossesAndMetrics.compute,
    #       metrics.py:399-419). The reported (test) set is --split_file 'val'.
    #   (2) no --val_split_file (legacy back-compat): carve a FOV-grouped 10%
    #       inner-validation set out of train and select on its inner-val loss.
    if val_split_file is not None:
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
            strict_split=not _relax_strict,
        )
        X_sel = sel_data["X_val"]
        y_sel = sel_data["y_val"]
        sel_cell_sizes = sel_data["val_cell_sizes"].astype(np.float32).reshape(-1, 1)
        X_sel = np.concatenate([X_sel, sel_cell_sizes], axis=1)
        # label_to_compact covers all ct2idx values, so every known label maps.
        y_sel = np.array([label_to_compact[y] for y in y_sel])
        # Cap to 200k cells EXACTLY as the main model caps its val set
        # (dataloader.py:269-273: np.random.default_rng(42).choice, no
        # replacement). Seed 42 is the DCT training seed, independent of --seed.
        n_val_cells = X_sel.shape[0]
        cap = min(200000, n_val_cells)
        rng = np.random.default_rng(42)
        sel_idx = rng.choice(n_val_cells, size=cap, replace=False)
        X_inner_val = X_sel[sel_idx]
        y_inner_val = y_sel[sel_idx]
        X_inner_train, y_inner_train = X_train, y_train
        select_on_macro_f1 = True
        n_train_fovs = len(np.unique(np.asarray(train_fov_names)))
        n_test_fovs = len(set(test_fov_names))
        print(
            f"Selection protocol: train on FULL --split_file train "
            f"({n_train_fovs} FOVs, {len(X_inner_train)} cells, no inner carve); "
            f"selection-val from {val_split_file} capped to {cap} of {n_val_cells} cells; "
            f"test (report) = {n_test_fovs} FOVs, {len(X_test)} cells."
        )
    else:
        # Legacy FOV-grouped 10% inner-val carve (back-compat, unchanged). The
        # reported (test) set MUST NOT drive checkpoint selection. Deviates from
        # canonical mahmoodlab/MAPS trainer.py (which selects on the eval set).
        train_fov_array = np.asarray(train_fov_names)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        inner_train_idx, inner_val_idx = next(
            gss.split(X_train, y_train, groups=train_fov_array)
        )
        X_inner_train, y_inner_train = X_train[inner_train_idx], y_train[inner_train_idx]
        X_inner_val, y_inner_val = X_train[inner_val_idx], y_train[inner_val_idx]
        select_on_macro_f1 = False
        print(
            f"Inner-val (FOV-grouped, disjoint from test): {len(inner_val_idx)} samples "
            f"from {len(np.unique(train_fov_array[inner_val_idx]))} FOVs; "
            f"inner-train: {len(inner_train_idx)} samples"
        )

    # Feature normalization (stats from the (inner-)train set; the FULL
    # --split_file train when --val_split_file is set, applied to selection-val
    # and test). Default: train-set z-score then /255 (DCT-safe); --no_znorm for
    # /255-only.
    print(f"\nNormalizing features ({'z-score + /255' if znorm else '/255 only'})...")
    X_inner_train_norm, train_mean, train_std = normalize_features(
        X_inner_train, znorm=znorm
    )
    X_inner_val_norm, _, _ = normalize_features(
        X_inner_val, mean=train_mean, std=train_std, znorm=znorm
    )
    X_test_norm, _, _ = normalize_features(
        X_test, mean=train_mean, std=train_std, znorm=znorm
    )

    # Convert to tensors (float64 to match canonical mahmoodlab/MAPS trainer.py:133)
    X_inner_train_tensor = torch.from_numpy(X_inner_train_norm.astype(np.float64))
    y_inner_train_tensor = torch.from_numpy(y_inner_train.astype(np.int64))
    X_inner_val_tensor = torch.from_numpy(X_inner_val_norm.astype(np.float64))
    X_test_tensor = torch.from_numpy(X_test_norm.astype(np.float64))

    # Class-balanced sampling for the WeightedRandomSampler (inner-train only).
    # Default 'dct' matches the main DeepCell-Types model and the other
    # baselines (sqrt-inverse-frequency with a 1000-count floor) so all methods
    # share one sampler; 'full_inv_freq' is the faithful mahmoodlab/MAPS scheme;
    # 'none' disables balancing (uniform shuffle).
    if class_balance == "none":
        sampler = None
        print("Class balancing: none (uniform shuffle)")
    else:
        if class_balance == "dct":
            print("Class balancing: dct (sqrt-inverse-frequency, 1000-count floor)")
            sample_weights = compute_sample_weights_dct(y_inner_train)
        else:  # full_inv_freq
            print("Class balancing: full_inv_freq (faithful mahmoodlab/MAPS, n/count)")
            sample_weights = compute_class_weights(y_inner_train)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # Create training dataloader. drop_last=True matches original MAPS.
    train_dataset = TensorDataset(X_inner_train_tensor, y_inner_train_tensor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=0,  # Features already in memory
    )

    # Create model (float64 to match canonical mahmoodlab/MAPS trainer.py:133)
    model = MAPSModel(
        input_dim=input_dim,
        num_classes=n_classes_compact,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device, dtype=torch.float64)

    print(f"\nModel architecture:")
    print(
        f"  Input: {input_dim} -> Hidden: {hidden_dim} -> Output: {n_classes_compact}"
    )
    print(f"  Dropout: {dropout}")

    # Loss and optimizer (canonical mahmoodlab/MAPS uses constant LR; no scheduler)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    run_config = {
        "normalization": "zscore_then_div255" if znorm else "div255_only",
        "znorm": bool(znorm),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }

    # Training loop — up to max_epochs with early stopping on inner-validation
    # loss (FOV-grouped, disjoint from test), which also drives checkpoint
    # selection. Schedule (max=500/min=250/patience=100) matches canonical
    # mahmoodlab/MAPS trainer.py; upstream early-stops on *train* loss, we use
    # the held-out inner-val loss to avoid touching the reported set.
    _sel_desc = (
        "best-by-selection-ct_macro_f1"
        if select_on_macro_f1
        else "best-by-inner-val-loss"
    )
    print(
        f"\nTraining MAPS model for up to {max_epochs} epochs "
        f"(early stop: min_epochs={min_epochs}, patience={patience}, {_sel_desc})..."
    )
    print(f"  LR schedule: constant lr={learning_rate}")
    best_val_loss = float("inf")
    best_sel_macro_f1 = float("-inf")
    best_macro_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    # Pre-define so torch.load(model_path) is reachable even if no improvement ever happens.
    model_path = Path(f"models/maps_{model_name}.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )

        # Evaluate on the inner-val set (selection signal only — never the
        # reported test set). Shared hierarchy collapse (Tcell + Stromal)
        # matches the main model's LossesAndMetrics.compute() exactly.
        y_true, y_pred, y_prob = evaluate(
            model, X_inner_val_tensor, y_inner_val, device
        )
        metrics = compute_baseline_metrics(
            y_true,
            y_pred,
            y_prob,
            n_classes_compact,
            hierarchy=CELL_TYPE_HIERARCHY,
            ct2idx=compact_ct2idx,
        )
        try:
            from sklearn.metrics import roc_auc_score

            metrics["auroc"] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovo",
                    labels=list(range(n_classes_compact)),
                )
            )
        except ValueError:
            metrics["auroc"] = float("nan")

        # Compute inner-validation loss (drives checkpoint selection)
        model.eval()
        with torch.no_grad():
            X_inner_val_dev = X_inner_val_tensor.to(device)
            y_inner_val_dev = torch.from_numpy(y_inner_val.astype(np.int64)).to(device)
            val_logits, _ = model(X_inner_val_dev)
            val_loss = criterion(val_logits, y_inner_val_dev).item()

        # Canonical selection signal: metrics["macro_f1"] is the hierarchical
        # ct_macro_f1 on the selection-val set (the same reduction the main
        # model reports via LossesAndMetrics.compute, metrics.py:399-419).
        # Printed every epoch so the smoke run (max_epochs=1) always shows it.
        if select_on_macro_f1:
            print(
                f"Epoch {epoch + 1:4d}: selection ct_macro_f1={metrics['macro_f1']:.4f} "
                f"(Val Loss={val_loss:.4f})"
            )

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Macro Acc={metrics['macro_accuracy']:.4f}, Weighted Acc={metrics['weighted_accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}, Weighted F1={metrics['weighted_f1']:.4f}, "
                f"AUROC={metrics['auroc']:.4f}"
            )

        # Model selection. With --val_split_file: keep the checkpoint with the
        # highest selection ct_macro_f1 (argmax hierarchical macro-F1 on the
        # external canonical val), matching scripts/train.py's val_macro_f1
        # selection for the main model. Legacy path: lowest inner-val loss.
        improved = (
            metrics["macro_f1"] > best_sel_macro_f1
            if select_on_macro_f1
            else val_loss < best_val_loss
        )
        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_sel_macro_f1 = max(best_sel_macro_f1, metrics["macro_f1"])
            best_macro_acc = metrics["macro_accuracy"]
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model with canonical key names (mahmoodlab/MAPS trainer.py:165-166)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_parameters": model.state_dict(),
                    "train_data_mean": train_mean,
                    "train_data_std": train_std,
                    "val_loss": best_val_loss,
                    "macro_accuracy": best_macro_acc,
                    "run_config": run_config,
                },
                model_path,
            )
        else:
            patience_counter += 1

        # Early stopping — only after min_epochs, mirroring canonical
        # mahmoodlab/MAPS trainer.py (counter > patience and epoch >= min_epochs).
        if patience_counter > patience and (epoch + 1) >= min_epochs:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no inner-val improvement for {patience_counter} epochs; best epoch {best_epoch})."
            )
            break

    # Load best model for final evaluation
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_parameters"])

    # Final evaluation with shared hierarchy collapse (matches main model)
    print("\nFinal evaluation on test set...")
    y_true_compact, y_pred_compact, y_prob_compact = evaluate(
        model, X_test_tensor, y_test, device
    )
    metrics = compute_baseline_metrics(
        y_true_compact,
        y_pred_compact,
        y_prob_compact,
        n_classes_compact,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )
    try:
        from sklearn.metrics import roc_auc_score

        metrics["auroc"] = float(
            roc_auc_score(
                y_true_compact,
                y_prob_compact,
                multi_class="ovo",
                labels=list(range(n_classes_compact)),
            )
        )
    except ValueError:
        metrics["auroc"] = float("nan")

    print(f"\nFinal Test Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    if select_on_macro_f1:
        print(f"  Best selection ct_macro_f1: {best_sel_macro_f1:.4f}")

    # Map probabilities to ct2idx-sorted columns for saving
    # save_baseline_predictions expects y_prob with len(ct2idx) columns,
    # one per cell type sorted by ct2idx value
    y_true_orig = np.array([compact_to_label[y] for y in y_true_compact])
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob_orig = np.zeros(
        (len(y_true_compact), len(dct_config.ct2idx)), dtype=np.float32
    )
    for compact_idx, orig_idx in compact_to_label.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob_orig[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]

    # Save predictions
    output_path = Path(f"output/{model_name}_maps_prediction.csv")
    save_baseline_predictions(
        y_true_orig,
        y_prob_orig,
        test_cell_indices,
        test_dataset_names,
        test_fov_names,
        dct_config.ct2idx,
        output_path,
        run_metadata={"method": "maps", "class_balance": class_balance},
    )

    # Save normalization stats for inference
    stats_path = Path(f"models/maps_{model_name}_stats.npz")
    np.savez(stats_path, mean=train_mean, std=train_std, **run_config)
    print(f"Normalization stats saved to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
