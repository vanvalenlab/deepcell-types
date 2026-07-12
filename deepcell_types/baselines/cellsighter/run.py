"""
CellSighter training and evaluation pipeline.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-40066-7
- Code: https://github.com/KerenLab/CellSighter
"""

import os
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Default data directory from environment
DATA_DIR = Path(
    os.environ.get("DEEPCELL_TYPES_ZARR_PATH") or os.environ.get("DATA_DIR", "")
)

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.dataset import create_dataloader
from deepcell_types.training.utils import BatchData
from deepcell_types.training.baseline_features import (
    compute_baseline_metrics,
    save_baseline_predictions,
)
from deepcell_types.training.metrics import build_label_remap

from .model import CellSighterModel, convert_batch_for_cellsighter
from .transforms import build_cellsighter_train_transform


def channel_presence(batch_data: BatchData, num_markers: int) -> torch.Tensor:
    """Per-sample presence mask over global marker channels.

    A global marker channel g is "present" for sample b iff some real (non-padded)
    local channel of that sample scatters into g (i.e. ch_idx[b] == g for a valid
    local channel). Absent channels are zero-padded by convert_batch_for_cellsighter
    and must be excluded from normalization statistics.

    Returns:
        present: (B, num_markers) bool — True where a real marker channel exists.
    """
    ch_idx = batch_data.ch_idx  # (B, C_max), global marker idx or -1 for padding
    valid = ch_idx >= 0  # (B, C_max)
    B = ch_idx.shape[0]
    present = ch_idx.new_zeros(B, num_markers, dtype=torch.bool)
    ch_idx_clamped = ch_idx.clamp(min=0)
    present.scatter_(1, ch_idx_clamped, valid)
    return present  # (B, num_markers)


def fit_per_modality_norm(
    dataloader: DataLoader,
    device: torch.device,
    num_markers: int,
    num_domains: int,
    max_batches: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming per-(domain, global-marker-channel) z-score statistics.

    Accumulates sum, sumsq and pixel-count over VALID pixels only (real channels
    present for that cell; zero-padded absent channels excluded) across up to
    max_batches training batches.

    Returns:
        mean: (num_domains, num_markers)
        std:  (num_domains, num_markers) — floored at 1e-6.
    """
    csum = torch.zeros(num_domains, num_markers, dtype=torch.float64, device=device)
    csumsq = torch.zeros(num_domains, num_markers, dtype=torch.float64, device=device)
    ccount = torch.zeros(num_domains, num_markers, dtype=torch.float64, device=device)

    for i, batch in enumerate(tqdm(dataloader, desc="Fitting norm stats")):
        if i >= max_batches:
            break
        batch_data = BatchData(*batch)
        batch_data.sample = batch_data.sample.to(device, non_blocking=True)
        batch_data.spatial_context = batch_data.spatial_context.to(
            device, non_blocking=True
        )
        batch_data.mask = batch_data.mask.to(device, non_blocking=True)
        batch_data.ch_idx = batch_data.ch_idx.to(device, non_blocking=True)
        domain_idx = batch_data.domain_idx.to(device, non_blocking=True)  # (B,)

        x = convert_batch_for_cellsighter(batch_data, num_markers)
        markers = x[:, :num_markers].double()  # (B, num_markers, H, W)
        B, _, H, W = markers.shape
        present = channel_presence(batch_data, num_markers)  # (B, num_markers) bool

        # Per-(sample, channel) reductions over spatial pixels, gated by presence.
        npix = float(H * W)
        ch_sum = markers.sum(dim=(2, 3)) * present  # (B, num_markers)
        ch_sumsq = (markers * markers).sum(dim=(2, 3)) * present
        ch_count = present.double() * npix  # (B, num_markers)

        # Scatter-add per-sample contributions into the sample's domain row.
        idx = domain_idx.view(B, 1).expand(B, num_markers)  # (B, num_markers)
        csum.scatter_add_(0, idx, ch_sum)
        csumsq.scatter_add_(0, idx, ch_sumsq)
        ccount.scatter_add_(0, idx, ch_count)

    denom = ccount.clamp(min=1.0)
    mean = csum / denom
    var = (csumsq / denom) - mean * mean
    std = var.clamp(min=0.0).sqrt().clamp(min=1e-6)
    # Domains/channels never seen: mean 0, std 1 (identity-ish, no NaNs).
    unseen = ccount == 0
    mean = mean.masked_fill(unseen, 0.0)
    std = std.masked_fill(unseen, 1.0)
    return mean.float(), std.float()


def apply_per_modality_norm(
    x: torch.Tensor,
    domain_idx: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_markers: int,
) -> torch.Tensor:
    """Z-score the marker channels of x in place by the sample's domain.

    Mask channels (x[:, num_markers:]) are left untouched.
    """
    m = mean[domain_idx][:, :, None, None]  # (B, num_markers, 1, 1)
    s = std[domain_idx][:, :, None, None]
    x[:, :num_markers] = (x[:, :num_markers] - m) / s
    return x


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
    norm_stats: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        batch_data.spatial_context = batch_data.spatial_context.to(
            device, non_blocking=True
        )
        batch_data.mask = batch_data.mask.to(device, non_blocking=True)
        batch_data.ch_idx = batch_data.ch_idx.to(device, non_blocking=True)
        ct_idx = batch_data.ct_idx.to(device, non_blocking=True)

        # Remap labels to contiguous 0-indexed (label_remap already on device)
        compact_labels = label_remap[ct_idx]

        # Convert to CellSighter format (globally aligned channels)
        x = convert_batch_for_cellsighter(batch_data, num_markers)

        # Optional per-modality z-score of marker channels
        if norm_stats is not None:
            domain_idx = batch_data.domain_idx.to(device, non_blocking=True)
            if not getattr(train_one_epoch, "_norm_dbg_printed", False):
                mean_before = x[:, :num_markers].mean().item()
                x = apply_per_modality_norm(
                    x, domain_idx, norm_stats[0], norm_stats[1], num_markers
                )
                mean_after = x[:, :num_markers].mean().item()
                print(
                    f"  [per_modality_norm] x[:, :num_markers] mean "
                    f"before={mean_before:.6f} after={mean_after:.6f}"
                )
                train_one_epoch._norm_dbg_printed = True
            else:
                x = apply_per_modality_norm(
                    x, domain_idx, norm_stats[0], norm_stats[1], num_markers
                )

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
    norm_stats: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        batch_data.spatial_context = batch_data.spatial_context.to(
            device, non_blocking=True
        )
        batch_data.mask = batch_data.mask.to(device, non_blocking=True)
        batch_data.ch_idx = batch_data.ch_idx.to(device, non_blocking=True)

        # Remap labels to compact 0-indexed (label_remap on device, move result to CPU)
        compact_true = label_remap[batch_data.ct_idx].cpu().numpy()

        # Convert to CellSighter format (globally aligned channels)
        x = convert_batch_for_cellsighter(batch_data, num_markers)

        # Optional per-modality z-score of marker channels
        if norm_stats is not None:
            domain_idx = batch_data.domain_idx.to(device, non_blocking=True)
            x = apply_per_modality_norm(
                x, domain_idx, norm_stats[0], norm_stats[1], num_markers
            )

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
@click.option(
    "--model_size",
    type=click.Choice(["resnet18", "resnet50"]),
    default="resnet50",
    help="ResNet variant: 'resnet50' (default, matches paper) or 'resnet18' (faster)",
)
@click.option(
    "--crop_size",
    type=int,
    default=60,
    help="Patch crop size. Default 60 matches the original CellSighter paper.",
)
@click.option(
    "--mask_self",
    is_flag=True,
    default=False,
    help="Ablation: zero all non-target-cell pixels (single-cell input, like "
    "DCT/MAPS). Default OFF — faithful CellSighter sees neighbor intensities.",
)
@click.option(
    "--cifar_stem",
    is_flag=True,
    default=False,
    help="Ablation: use the 3x3/s1 CIFAR stem (for 32x32 crops) instead of the "
    "faithful ImageNet 7x7/s2 ResNet50 stem.",
)
@click.option(
    "--allow_split_mismatch",
    is_flag=True,
    default=False,
    help="Downgrade the split archive-fingerprint check to a warning. Use when "
    "the split file's FOVs all exist in the current archive but a non-FOV "
    "archive attr changed since the split was created.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for weight init, augmentation, and the training sampler. "
    "Vary across runs to build an ensemble of diverse members.",
)
@click.option(
    "--split_mode",
    type=click.Choice(["fov", "patch"]),
    default="fov",
    help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)",
)
@click.option(
    "--split_file",
    type=str,
    default=None,
    help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)",
)
@click.option(
    "--val_split_file",
    type=str,
    default=None,
    help="If set, train on the FULL --split_file train (no inner carve) and "
    "select the best epoch on the 'val' FOVs of THIS file, capped to 200k cells "
    "at seed 42 (mirroring deepcell_types/training/dataloader.py:269-273 "
    "max_val_samples) and scored by the hierarchical ct_macro_f1 "
    "(LossesAndMetrics.compute, metrics.py:399-419). The reported set stays "
    "--split_file 'val' (unless --test_split_file overrides it). When unset, "
    "keeps the legacy 10% FOV-grouped inner-val carve + selection.",
)
@click.option(
    "--test_split_file",
    type=str,
    default=None,
    help="If set, the final evaluation + prediction CSV run on the 'val' FOVs "
    "of THIS split (e.g. the held-out v10 test split), using the same faithful "
    "crop/mask settings. Training and model selection still use --split_file. "
    "This yields a CSV directly comparable to the published baseline numbers.",
)
@click.option(
    "--val_every_n_epochs",
    type=int,
    default=10,
    help="Validate every N epochs (default 10, matching original CellSighter paper)",
)
@click.option(
    "--no_amp",
    is_flag=True,
    default=False,
    help="Disable automatic mixed precision (AMP is enabled by default on CUDA)",
)
@click.option(
    "--no_compile",
    is_flag=True,
    default=False,
    help="Disable torch.compile optimization",
)
@click.option(
    "--class_balance",
    type=click.Choice(["sqrt", "equal", "none"]),
    default="sqrt",
    help="Training class-balancing scheme. 'sqrt' (default): DCT sampler — "
    "sqrt-inverse-frequency with a 1000-count floor, identical to the main "
    "DeepCell-Types model and the other baselines (shared comparison footing). "
    "'equal' (FAITHFUL, ablation): full-inverse-frequency WeightedRandomSampler "
    "(weight=total/count) over a per-class pool capped at --size_data, "
    "reproducing the original CellSighter subsample_const_size + define_sampler. "
    "'none': uniform sampling (ablation).",
)
@click.option(
    "--size_data",
    type=int,
    default=1000,
    help="Faithful CellSighter per-class training-pool cap (subsample_const_size; "
    "paper config size_data=1000). Only applied when --class_balance=equal. "
    "Pass 0 to disable the cap (pure full-inverse-frequency over all cells).",
)
@click.option(
    "--no_weighted_sampler",
    is_flag=True,
    default=False,
    help="Deprecated alias for --class_balance none (uniform sampling). When "
    "passed, overrides --class_balance.",
)
@click.option(
    "--per_modality_norm",
    is_flag=True,
    default=False,
    help="Apply per-(modality, global-marker-channel) z-score normalization to the "
    "marker input channels. Stats are fit by a streaming pass over the training "
    "loader before training. Default OFF — faithful CellSighter feeds raw [0,1].",
)
@click.option(
    "--max_samples_per_epoch",
    type=int,
    default=500_000,
    help="Cap cells drawn per training epoch (default 500000). Lower it for a fast "
    "undertrained signal sweep.",
)
@click.option(
    "--num_workers",
    type=int,
    default=8,
    help="DataLoader workers (default 8). Lower it to reduce memory under "
    "concurrent runs.",
)
def main(
    model_name: str,
    device_num: str,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    pretrained: bool,
    model_size: str,
    crop_size: int,
    mask_self: bool,
    cifar_stem: bool,
    allow_split_mismatch: bool,
    seed: int,
    split_mode: str,
    split_file: str,
    val_split_file: str,
    test_split_file: str,
    val_every_n_epochs: int,
    no_amp: bool,
    no_compile: bool,
    class_balance: str,
    size_data: int,
    no_weighted_sampler: bool,
    per_modality_norm: bool,
    max_samples_per_epoch: int,
    num_workers: int,
):
    """Train CellSighter baseline for cell type classification."""
    # Set device
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Seed weight init + augmentation RNG; the training sampler/splits get the
    # same seed via create_dataloader below. Vary --seed for ensemble members.
    torch.manual_seed(seed)

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
    compact_ct2idx = {
        name: label_remap[idx].item() for name, idx in dct_config.ct2idx.items()
    }

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(
        f"Input channels: {input_channels} ({num_markers} markers + cell mask + neighbor mask)"
    )
    print(f"Model: {model_size}, AMP: {not no_amp}, torch.compile: {not no_compile}")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    use_cuda = device.type == "cuda"

    # --no_weighted_sampler is a deprecated alias for --class_balance none and
    # overrides it when passed.
    if no_weighted_sampler:
        if class_balance != "none":
            print(
                "  [deprecation] --no_weighted_sampler overrides "
                f"--class_balance {class_balance} -> none"
            )
        class_balance = "none"
    # size_data=0 disables the per-class cap (pure full-inverse-frequency).
    size_data_cap = size_data if size_data and size_data > 0 else None
    # The --size_data cap is only applied for the faithful 'equal' scheme
    # (subsample_const_size); warn if it was set under another scheme so it is
    # not silently inert.
    if size_data_cap is not None and class_balance != "equal":
        print(
            f"  [warning] --size_data {size_data_cap} is ignored under "
            f"--class_balance {class_balance} (only applied for 'equal')"
        )

    # Faithful CellSighter: full crop incl. neighbor intensities (mask_self=False),
    # 60x60 crops, the original's geometric augmentation pipeline, and
    # equal-proportion balancing (subsample_const_size + define_sampler).
    balance_desc = {
        "equal": (
            f"equal-proportion / full-inv-freq, size_data="
            f"{size_data_cap if size_data_cap else 'off'} (faithful)"
        ),
        "sqrt": "sqrt-inv-freq + 1000 floor (ablation)",
        "none": "uniform (ablation)",
    }[class_balance]
    print(
        f"Crop: {crop_size}x{crop_size} | "
        f"intensities: {'self-masked (ablation)' if mask_self else 'unmasked (faithful)'} | "
        f"stem: {'CIFAR (ablation)' if cifar_stem else 'ImageNet (faithful)'} | "
        f"balance: {balance_desc}"
    )
    cs_train_transform = build_cellsighter_train_transform()

    # Model-selection validation. Two protocols:
    #   (1) --val_split_file (canonical, matches the main DCT model): train on
    #       the FULL --split_file train (inner_val_ratio=0.0, no carve) and select
    #       the best epoch on the 'val' FOVs of --val_split_file, built by a
    #       second create_dataloader call with max_val_samples=200000, seed=42
    #       (mirrors dataloader.py:269-273). The reported (test) set is
    #       --split_file 'val' (unchanged), unless --test_split_file overrides.
    #   (2) no --val_split_file (legacy back-compat): inner_val_ratio=0.1 carves
    #       a FOV-grouped inner-validation set out of the TRAIN FOVs; the held-out
    #       inner-val loader comes back via metadata["inner_val_loader"].
    inner_val_ratio = 0.0 if val_split_file else 0.1
    train_loader, test_loader, metadata = create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        batch_size=batch_size,
        num_dropout_channels=0,  # No channel dropout for CellSighter
        num_workers=num_workers,
        only_test=False,
        use_fov_splits=(split_mode == "fov"),
        split_file=split_file,
        skip_distance_transform=True,  # CellSighter doesn't use distance transform
        persistent_workers=True,
        max_samples_per_epoch=max_samples_per_epoch,
        multiprocessing_context="fork",  # zarr v3 is not fork-safe
        pin_memory=use_cuda,  # Faster CPU→GPU transfers
        inner_val_ratio=inner_val_ratio,
        crop_size=crop_size,
        output_size=crop_size,  # extract directly at crop_size (no resize)
        mask_intensities=mask_self,  # faithful CellSighter: False -> keep neighbors
        train_transform=cs_train_transform,
        split_strict=not allow_split_mismatch,
        seed=seed,
        class_balance=class_balance,  # "equal" (faithful) | "sqrt" | "none"
        size_data=size_data_cap,  # faithful subsample_const_size cap
    )

    if val_split_file:
        # External canonical selection-val: the 'val' FOVs of --val_split_file,
        # capped to 200k cells at seed 42 via create_dataloader's max_val_samples
        # (dataloader.py:269-273). The train loader from this call is discarded
        # (class_balance="none" skips the unused weighted-sampler build); we only
        # take its unaugmented val loader as the selection set.
        _, sel_loader, sel_meta = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=skip_datasets,
            keep_datasets=keep_datasets,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=num_workers,
            only_test=False,
            use_fov_splits=True,
            split_file=val_split_file,
            skip_distance_transform=True,
            persistent_workers=True,
            max_samples_per_epoch=max_samples_per_epoch,
            multiprocessing_context="fork",
            pin_memory=use_cuda,
            inner_val_ratio=0.0,
            max_val_samples=200000,  # cap @ seed 42, mirrors dataloader.py:269-273
            crop_size=crop_size,
            output_size=crop_size,
            mask_intensities=mask_self,
            split_strict=not allow_split_mismatch,
            seed=42,  # DCT training seed for the cap (independent of --seed)
            class_balance="none",  # train loader is discarded; skip weight build
        )
        import json as _json

        with open(split_file) as _f:
            _sj = _json.load(_f)
        _kept = set(keep_datasets) if keep_datasets else None
        _train_fovs = [
            k for k in _sj["train"] if (_kept is None or k in _kept)
        ]
        print(f"Active datasets: {metadata['active_datasets']}")
        print(f"Number of samples: {metadata['num_samples']}")
        print(
            f"Selection protocol: train on FULL --split_file train "
            f"({len(_train_fovs)} FOVs, no inner carve); selection-val from "
            f"{val_split_file} capped to {len(sel_loader.dataset)} cells "
            f"(cap 200000 @ seed 42); test (report) = {len(test_loader.dataset)} "
            f"cells from --split_file 'val'."
        )
    else:
        sel_loader = metadata.get("inner_val_loader")
        if sel_loader is None:
            # FOV splits are required for a leakage-free inner-val carve.
            raise click.UsageError(
                "CellSighter model selection requires FOV splits (--split_mode fov "
                "with a --split_file) so an inner-validation set can be carved from "
                "the training FOVs (or pass --val_split_file for the canonical "
                "external selection-val protocol)."
            )
        print(f"Active datasets: {metadata['active_datasets']}")
        print(f"Number of samples: {metadata['num_samples']}")
        print(
            f"Inner-val cells (FOV-grouped, for selection): "
            f"{metadata['num_inner_val']}"
        )

    # Create model
    model = CellSighterModel(
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        model_size=model_size,
        cifar_stem=cifar_stem,
    ).to(device)

    # Compile supported operations for the fused PyTorch execution path.
    if not no_compile and hasattr(torch, "compile"):
        print("Applying torch.compile...")
        model = torch.compile(model)

    # Mixed precision (AMP).
    use_amp = use_cuda and not no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16 if use_amp else None

    # Move label_remap to the device once to avoid per-batch transfers.
    label_remap = label_remap.to(device)

    # Loss and optimizer (matching CellSighter paper: constant lr=0.001)
    # Note: the original CellSighter repo creates an ExponentialLR scheduler
    # but never calls scheduler.step(), so it trains with constant lr.
    # We match that behavior here.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: fit per-(modality, marker-channel) z-score stats on the train loader
    # before training. num_domains comes from the config's domain2idx mapping.
    norm_stats = None
    if per_modality_norm:
        num_domains = dct_config.NUM_DOMAINS
        print(
            f"\nFitting per-modality norm stats "
            f"(num_domains={num_domains}, num_markers={num_markers}, max 200 batches)..."
        )
        norm_mean, norm_std = fit_per_modality_norm(
            train_loader, device, num_markers, num_domains, max_batches=200
        )
        print(
            f"  norm_mean shape: {tuple(norm_mean.shape)}, "
            f"norm_std shape: {tuple(norm_std.shape)}"
        )
        # Sample value from a populated (domain, channel) cell for evidence
        # (unseen cells are left at mean=0/std=1; show a real fitted one).
        seen = norm_mean.abs() > 0
        if seen.any():
            d0, c0 = (seen.nonzero()[0]).tolist()
        else:
            d0, c0 = 0, 0
        print(
            f"  sample mean[{d0},{c0}]={norm_mean[d0, c0].item():.6f}, "
            f"std[{d0},{c0}]={norm_std[d0, c0].item():.6f}"
        )
        norm_stats = (norm_mean, norm_std)
        # Reap the fit-pass dataloader workers before training spawns a fresh
        # set — otherwise spawn workers briefly double (~16) and OOM under
        # concurrent runs ("DataLoader worker exited unexpectedly").
        import gc

        gc.collect()

    # Training loop
    print("\nTraining CellSighter model...")
    # Select on macro-F1 — the headline metric, matching the main model
    # (scripts/train.py selects on val_macro_f1) and the other baselines. Use
    # -inf so the first val pass always wins, even if macro_f1 is exactly 0
    # (happens on smoke runs that don't train long enough to predict the majority class).
    best_macro_f1 = float("-inf")
    model_path = Path(f"models/cellsighter_{model_name}.pth")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_remap,
            num_markers,
            scaler=scaler,
            amp_dtype=amp_dtype,
            norm_stats=norm_stats,
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate every N epochs + final epoch (matches original paper).
        # Selection runs on the held-out inner-val set (never the reported test
        # set) so the best checkpoint is not chosen on the set it is reported on.
        is_val_epoch = ((epoch + 1) % val_every_n_epochs == 0) or (epoch + 1 == epochs)
        if is_val_epoch:
            # Evaluate on inner-val (returns compact 0-indexed labels)
            y_true, y_pred, y_prob, _, _, _ = evaluate(
                model,
                sel_loader,
                device,
                label_remap,
                num_markers,
                amp_dtype=amp_dtype,
                norm_stats=norm_stats,
            )
            # Shared hierarchy collapse (Tcell + Stromal) — matches main model
            metrics = compute_baseline_metrics(
                y_true,
                y_pred,
                y_prob,
                num_classes,
                hierarchy=CELL_TYPE_HIERARCHY,
                ct2idx=compact_ct2idx,
            )

            # metrics["macro_f1"] is the hierarchical ct_macro_f1 on the
            # selection-val set (the same reduction the main model reports via
            # LossesAndMetrics.compute, metrics.py:399-419) — the canonical
            # selection signal.
            _sel_tag = "external-val" if val_split_file else "inner-val"
            print(
                f"  Epoch {epoch + 1}: selection ct_macro_f1="
                f"{metrics['macro_f1']:.4f} ({_sel_tag})"
            )
            print(f"  Inner-val Macro Accuracy: {metrics['macro_accuracy']:.4f}")
            print(f"  Inner-val Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
            print(f"  Inner-val Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Inner-val Weighted F1: {metrics['weighted_f1']:.4f}")

            # Save best model (selected on inner-val macro-F1)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # Strip _orig_mod. prefix from torch.compile'd models
                state_dict = {
                    k.removeprefix("_orig_mod."): v
                    for k, v in model.state_dict().items()
                }
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "macro_f1": best_macro_f1,
                    },
                    model_path,
                )
                print(f"  Saved best model to {model_path}")

    # Load best model before final evaluation
    best_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Load into unwrapped model if torch.compile was used (keys saved without _orig_mod. prefix)
    load_target = getattr(model, "_orig_mod", model)
    load_target.load_state_dict(best_checkpoint["model_state_dict"])
    print(
        f"Loaded best model from {model_path} (epoch {best_checkpoint['epoch']}, macro_f1={best_checkpoint['macro_f1']:.4f})"
    )

    # Optionally swap in a dedicated held-out test split for the final eval,
    # using the SAME faithful crop/mask settings the model was trained with.
    # The val loader from create_dataloader is unaugmented (AugmentedDataset
    # wraps only the train subset), so this is a clean test-time pass.
    if test_split_file:
        # Fairness guard: the held-out test FOVs (the 'val' set of
        # test_split_file) must be disjoint from the training FOVs (the 'train'
        # set of split_file), or the reported number leaks training data.
        # load_fov_splits only checks overlap *within* a single file, so this
        # cross-file overlap must be checked explicitly.
        if split_file:
            import json

            with open(split_file) as f:
                _train_fovs = {
                    (ds, fov)
                    for ds, fovs in json.load(f).get("train", {}).items()
                    for fov in fovs
                }
            with open(test_split_file) as f:
                _test_fovs = {
                    (ds, fov)
                    for ds, fovs in json.load(f).get("val", {}).items()
                    for fov in fovs
                }
            _leak = _train_fovs & _test_fovs
            if _leak:
                raise click.UsageError(
                    f"{len(_leak)} FOV(s) appear in both the training split "
                    f"(--split_file 'train') and the held-out test split "
                    f"(--test_split_file 'val'), e.g. {sorted(_leak)[:5]}. The "
                    "reported number would leak training data; use coordinated "
                    "splits (see scripts/split_val_for_test.py)."
                )
        print(f"\nRebuilding eval loader on held-out test split: {test_split_file}")
        _, test_loader, _ = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=skip_datasets,
            keep_datasets=keep_datasets,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=num_workers,
            only_test=False,
            use_fov_splits=True,
            split_file=test_split_file,
            skip_distance_transform=True,
            persistent_workers=True,
            max_samples_per_epoch=max_samples_per_epoch,
            multiprocessing_context="fork",
            pin_memory=use_cuda,
            crop_size=crop_size,
            output_size=crop_size,
            mask_intensities=mask_self,
            split_strict=not allow_split_mismatch,
        )
    elif val_split_file:
        # Canonical protocol: selection used the external --val_split_file 'val',
        # so the test_loader (--split_file 'val') is a held-out report set that
        # never drove selection — no selection-on-eval-set warning needed.
        print(
            "\nFinal evaluation on --split_file 'val' (held out from selection, "
            "which used --val_split_file 'val'). This is a comparable reported set."
        )
    else:
        print(
            "\nWARNING: --test_split_file not set. Final evaluation reuses the "
            "val loader that was also used for checkpoint selection, so this "
            "number is selection-on-the-eval-set and is NOT comparable to a "
            "held-out published baseline. Pass --test_split_file for any "
            "reported number (see scripts/split_val_for_test.py)."
        )

    # Final evaluation (returns compact 0-indexed labels and probabilities)
    print("\nFinal evaluation on test set...")
    (
        y_true_compact,
        y_pred_compact,
        y_prob_compact,
        test_dataset_names,
        test_fov_names,
        test_cell_indices,
    ) = evaluate(
        model,
        test_loader,
        device,
        label_remap,
        num_markers=num_markers,
        amp_dtype=amp_dtype,
        norm_stats=norm_stats,
    )
    # Shared hierarchy collapse (Tcell + Stromal) — matches main model
    metrics = compute_baseline_metrics(
        y_true_compact,
        y_pred_compact,
        y_prob_compact,
        num_classes,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    print("\nFinal Test Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Best Inner-val Macro F1: {best_macro_f1:.4f}")

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
        run_metadata={
            "method": "cellsighter",
            "class_balance": class_balance,
            "size_data": size_data_cap,
        },
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
