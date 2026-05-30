"""
CellSighter training and evaluation pipeline.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-40066-7
- Code: https://github.com/KerenLab/CellSighter
"""

import os
import click
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.dataset import create_dataloader
from deepcell_types.training.utils import BatchData
from deepcell_types.training.baseline_features import compute_baseline_metrics, save_baseline_predictions
from deepcell_types.training.metrics import build_label_remap

from .model import CellSighterModel, convert_batch_for_cellsighter


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_remap: torch.Tensor,
    num_markers: int = 269,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: CellSighter model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        label_remap: Lookup tensor mapping original ct2idx values to compact 0-indexed labels
            (should already be on device)
        num_markers: Total number of unique markers for global channel alignment
        scaler: GradScaler for mixed precision (None to disable AMP)
        amp_dtype: Autocast dtype (torch.float16 or torch.bfloat16)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    for batch in tqdm(dataloader, desc="Training"):
        batch_data = BatchData(*batch)
        batch_data.sample = batch_data.sample.to(device, non_blocking=True)
        batch_data.spatial_context = batch_data.spatial_context.to(device, non_blocking=True)
        batch_data.mask = batch_data.mask.to(device, non_blocking=True)
        batch_data.ch_idx = batch_data.ch_idx.to(device, non_blocking=True)
        ct_idx = batch_data.ct_idx.to(device, non_blocking=True)

        # Remap labels to contiguous 0-indexed (label_remap already on device)
        compact_labels = label_remap[ct_idx]

        # Convert to CellSighter format (globally aligned channels)
        x = convert_batch_for_cellsighter(batch_data, num_markers)

        # Forward pass with optional AMP
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, compact_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, compact_labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_remap: torch.Tensor,
    num_markers: int = 269,
    amp_dtype: torch.dtype | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[int]]:
    """
    Evaluate model on dataloader.

    Args:
        model: CellSighter model
        dataloader: Evaluation dataloader
        device: Device to use
        label_remap: Lookup tensor mapping original ct2idx values to compact 0-indexed labels
            (should already be on device)
        num_markers: Total number of unique markers for global channel alignment
        amp_dtype: Autocast dtype for mixed precision (None to disable)

    Returns:
        y_true: True labels (compact 0-indexed)
        y_pred: Predicted labels (compact 0-indexed)
        y_prob: Predicted probabilities (N, num_classes)
        dataset_names: Dataset names
        fov_names: FOV names
        cell_indices: Cell indices
    """
    model.eval()
    all_true = []
    all_pred = []
    all_prob = []
    all_dataset_names = []
    all_fov_names = []
    all_cell_indices = []
    use_amp = amp_dtype is not None

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch_data = BatchData(*batch)
        batch_data.sample = batch_data.sample.to(device, non_blocking=True)
        batch_data.spatial_context = batch_data.spatial_context.to(device, non_blocking=True)
        batch_data.mask = batch_data.mask.to(device, non_blocking=True)
        batch_data.ch_idx = batch_data.ch_idx.to(device, non_blocking=True)

        # Remap labels to compact 0-indexed (label_remap on device, move result to CPU)
        compact_true = label_remap[batch_data.ct_idx].cpu().numpy()

        # Convert to CellSighter format (globally aligned channels)
        x = convert_batch_for_cellsighter(batch_data, num_markers)

        # Forward pass (returns softmax probabilities in eval mode)
        if use_amp:
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                probs = model(x)
        else:
            probs = model(x)

        # Get predictions (already in compact space since model outputs num_classes)
        preds = probs.argmax(dim=-1)

        all_true.append(compact_true)
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.float().cpu().numpy())
        all_dataset_names.extend(batch_data.dataset_name)
        all_fov_names.extend(batch_data.fov_name)
        all_cell_indices.extend(batch_data.cell_index.numpy().tolist())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    return y_true, y_pred, y_prob, all_dataset_names, all_fov_names, all_cell_indices


@click.command()
@click.option("--model_name", type=str, default="cellsighter_0")
@click.option("--device_num", type=str, default="cuda:0")
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
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate (constant, matching original CellSighter)",
)
@click.option(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size",
)
@click.option(
    "--pretrained",
    type=bool,
    default=False,
    help="Use ImageNet pretrained weights (default False to match original CellSighter)",
)
@click.option("--model_size", type=click.Choice(["resnet18", "resnet50"]), default="resnet50",
              help="ResNet variant: 'resnet50' (default, matches paper) or 'resnet18' (faster)")
@click.option("--split_mode", type=click.Choice(["fov", "patch"]), default="fov",
              help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)")
@click.option("--split_file", type=str, default=None,
              help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)")
@click.option("--val_every_n_epochs", type=int, default=10,
              help="Validate every N epochs (default 10, matching original CellSighter paper)")
@click.option("--no_amp", is_flag=True, default=False,
              help="Disable automatic mixed precision (AMP is enabled by default on CUDA)")
@click.option("--no_compile", is_flag=True, default=False,
              help="Disable torch.compile optimization")
@click.option("--min_channels", type=int, default=3, help="Min non-DAPI channels per dataset (filters 2-channel datasets)")
@click.option("--test_split_file", type=str, default=None,
              help="If set, after training+selection on --split_file's val, run the FINAL "
                   "evaluation/prediction on THIS split's val set instead. Lets selection "
                   "(early stopping) use a held-out val while predictions are written on a "
                   "separate frozen test set.")
def main(
    model_name: str,
    device_num: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    pretrained: bool,
    model_size: str,
    split_mode: str,
    split_file: str,
    val_every_n_epochs: int,
    no_amp: bool,
    no_compile: bool,
    min_channels: int,
    test_split_file: str,
):
    """Train CellSighter baseline for cell type classification."""
    # Set device
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    if enable_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project="deepcelltypes-temp-train",
            dir="wandb_tmp",
            job_type="train",
            name=f"{model_name}_cellsighter",
            config={
                "model_type": "cellsighter",
                "model_size": model_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "pretrained": pretrained,
                "split_mode": split_mode,
                "val_every_n_epochs": val_every_n_epochs,
                "amp": not no_amp,
                "compile": not no_compile,
            },
        )

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES
    # Input channels = NUM_MARKERS (271) + 2 (cell mask + neighbor mask)
    # Channels are globally aligned via marker2idx so each marker always
    # occupies the same input position across datasets.
    num_markers = dct_config.NUM_MARKERS
    input_channels = num_markers + 2

    # Build compact label mapping (ct2idx values are not 0-indexed,
    # but CrossEntropyLoss requires contiguous 0-indexed labels)
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    compact_to_orig = {i: v for i, v in enumerate(sorted_ct_values)}
    label_remap = build_label_remap(dct_config.ct2idx)
    compact_ct2idx = {name: label_remap[idx].item() for name, idx in dct_config.ct2idx.items()}

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(f"Input channels: {input_channels} ({num_markers} markers + cell mask + neighbor mask)")
    print(f"Model: {model_size}, AMP: {not no_amp}, torch.compile: {not no_compile}")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    use_cuda = device.type == "cuda"

    # Load train and test data
    train_loader, test_loader, metadata = create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        batch_size=batch_size,
        num_dropout_channels=0,  # No channel dropout for CellSighter
        num_workers=8,
        only_test=False,
        use_fov_splits=(split_mode == "fov"),
        split_file=split_file,
        skip_distance_transform=True,  # CellSighter doesn't use distance transform
        persistent_workers=True,
        max_samples_per_epoch=500_000,  # Cap iterations (~2K batches/epoch at bs=256)
        multiprocessing_context="spawn",  # zarr v3 is not fork-safe
        pin_memory=use_cuda,  # Faster CPU→GPU transfers
        min_channels=min_channels,
    )

    print(f"Active datasets: {metadata['active_datasets']}")
    print(f"Number of samples: {metadata['num_samples']}")

    # Create model
    model = CellSighterModel(
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        model_size=model_size,
    ).to(device)

    # Fix 6: torch.compile for fused operations (PyTorch 2.x+)
    if not no_compile and hasattr(torch, "compile"):
        print("Applying torch.compile...")
        model = torch.compile(model)

    # Fix 1: Mixed precision (AMP)
    use_amp = use_cuda and not no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16 if use_amp else None

    # Fix 5: Move label_remap to device once (avoid per-batch CPU→GPU transfer)
    label_remap = label_remap.to(device)

    # Loss and optimizer (matching CellSighter paper: constant lr=0.001)
    # Note: the original CellSighter repo creates an ExponentialLR scheduler
    # but never calls scheduler.step(), so it trains with constant lr.
    # We match that behavior here.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nTraining CellSighter model...")
    # Use -inf so the first val pass always wins, even if macro_accuracy is exactly 0
    # (happens on smoke runs that don't train long enough to predict the majority class).
    best_macro_acc = float("-inf")
    model_path = Path(f"models/cellsighter_{model_name}.pth")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_remap,
            num_markers, scaler=scaler, amp_dtype=amp_dtype,
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate every N epochs + final epoch (matches original paper)
        is_val_epoch = ((epoch + 1) % val_every_n_epochs == 0) or (epoch + 1 == epochs)
        if is_val_epoch:
            # Evaluate (returns compact 0-indexed labels)
            y_true, y_pred, y_prob, _, _, _ = evaluate(
                model, test_loader, device, label_remap, num_markers,
                amp_dtype=amp_dtype,
            )
            # Shared hierarchy collapse (Tcell + Stromal) — matches main model
            metrics = compute_baseline_metrics(
                y_true, y_pred, y_prob, num_classes,
                hierarchy=CELL_TYPE_HIERARCHY, ct2idx=compact_ct2idx,
            )

            print(f"  Test Macro Accuracy: {metrics['macro_accuracy']:.4f}")
            print(f"  Test Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
            print(f"  Test Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Test Weighted F1: {metrics['weighted_f1']:.4f}")

            # Log to wandb
            if enable_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "test/macro_accuracy": metrics["macro_accuracy"],
                    "test/weighted_accuracy": metrics["weighted_accuracy"],
                    "test/macro_f1": metrics["macro_f1"],
                    "test/weighted_f1": metrics["weighted_f1"],
                })

            # Save best model
            if metrics["macro_accuracy"] > best_macro_acc:
                best_macro_acc = metrics["macro_accuracy"]
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # Strip _orig_mod. prefix from torch.compile'd models
                state_dict = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_accuracy": best_macro_acc,
                }, model_path)
                print(f"  Saved best model to {model_path}")
        else:
            # Log training metrics only
            if enable_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                })

    # Load best model before final evaluation
    best_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Load into unwrapped model if torch.compile was used (keys saved without _orig_mod. prefix)
    load_target = getattr(model, "_orig_mod", model)
    load_target.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"Loaded best model from {model_path} (epoch {best_checkpoint['epoch']}, macro_acc={best_checkpoint['macro_accuracy']:.4f})")

    # Optional: swap in a separate frozen test split for the final prediction pass.
    if test_split_file is not None:
        print(f"Loading separate test split for final predictions: {test_split_file}")
        # Restrict the test loader to exactly the test split's FOVs. load_fov_splits
        # is strict (loaded set must equal train ∪ val), so we derive keep from the
        # test split itself rather than reusing the CLI --keep_datasets, which scopes
        # the *training* loader to the (different) selection split.
        import json as _json
        _td = _json.loads(Path(test_split_file).read_text())
        _test_keep = list(_td.get("train", {})) + list(_td.get("val", {}))
        _, test_loader, _ = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=skip_datasets,
            keep_datasets=_test_keep,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=8,
            only_test=False,
            use_fov_splits=(split_mode == "fov"),
            split_file=test_split_file,
            skip_distance_transform=True,
            persistent_workers=True,
            max_samples_per_epoch=500_000,
            multiprocessing_context="spawn",
            pin_memory=use_cuda,
            min_channels=min_channels,
        )

    # Final evaluation (returns compact 0-indexed labels and probabilities)
    print("\nFinal evaluation on test set...")
    y_true_compact, y_pred_compact, y_prob_compact, test_dataset_names, test_fov_names, test_cell_indices = evaluate(
        model, test_loader, device, label_remap, num_markers=num_markers,
        amp_dtype=amp_dtype,
    )
    # Shared hierarchy collapse (Tcell + Stromal) — matches main model
    metrics = compute_baseline_metrics(
        y_true_compact, y_pred_compact, y_prob_compact, num_classes,
        hierarchy=CELL_TYPE_HIERARCHY, ct2idx=compact_ct2idx,
    )

    print(f"\nFinal Test Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Best Macro Accuracy: {best_macro_acc:.4f}")

    # Log final metrics to wandb
    if enable_wandb:
        wandb.log({
            "final/macro_accuracy": metrics["macro_accuracy"],
            "final/weighted_accuracy": metrics["weighted_accuracy"],
            "final/macro_f1": metrics["macro_f1"],
            "final/weighted_f1": metrics["weighted_f1"],
            "final/best_macro_accuracy": best_macro_acc,
        })

    # Map compact labels back to original ct2idx values for saving
    y_true_orig = np.array([compact_to_orig[int(y)] for y in y_true_compact])

    # Map probabilities to ct2idx-sorted columns for saving
    # save_baseline_predictions expects y_prob with len(ct2idx) columns,
    # one per cell type sorted by ct2idx value
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob = np.zeros((len(y_true_compact), len(dct_config.ct2idx)), dtype=np.float32)
    for compact_idx, orig_idx in compact_to_orig.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]

    # Save predictions
    output_path = Path(f"output/{model_name}_cellsighter_prediction.csv")
    save_baseline_predictions(
        y_true_orig,
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
