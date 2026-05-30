"""Training pipeline for CellTypeAnnotator.

Features:
- Direct FocalLoss classification
- FOV-level splits (no spatial leakage)
- Per-epoch validation + early stopping
- sqrt-frequency weighted sampling
- Tissue-aware exclusion during training
- Full reproducibility seeding
- Loss weights: ct:1, marker_pos:1, domain:0 (DANN disabled by default)
- AdamW + OneCycleLR (cosine with 5% warmup)
"""

import json
import logging
import os
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchinfo import summary
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)

from deepcell_types.training.config import TissueNetConfig, WARMUP_PCT, CELL_TYPE_HIERARCHY
from deepcell_types.training.dataset import create_dataloader, AugmentedDataset, FullImageDataset
from deepcell_types.model import create_model
from deepcell_types.training.losses import FocalLoss
from deepcell_types.training.utils import (
    BatchData, LossesAndMetrics, MPMetricsTracker, PredLogger,
    log_epoch_metrics, log_confusion_matrix, seed_everything,
    get_tissue_ct_exclude, build_label_remap,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))

# Default loss weights for multi-task training
DEFAULT_LOSS_WEIGHTS = {"ct": 1.0, "domain": 0.0, "marker_pos": 1.0}


def compute_class_weights(dct_config, dataset, label_remap):
    """Compute sqrt-inverse-frequency class weights for FocalLoss.

    Weights are indexed in compact 0-indexed label space.
    """
    ct_counts = dataset.ct_counts
    total = sum(ct_counts.values())
    n_classes = len(dct_config.ct2idx)

    weights = torch.ones(n_classes)
    for ct, idx in dct_config.ct2idx.items():
        compact_idx = label_remap[idx].item()
        count = ct_counts.get(ct, 0)
        if count > 0:
            weights[compact_idx] = np.sqrt(total / count)
        else:
            weights[compact_idx] = 1.0

    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


def forward_one_batch(
    batch_data: BatchData,
    device: torch.device,
    model: torch.nn.Module,
    prefix: str,
    losses_metrics: LossesAndMetrics,
    predlogger: PredLogger = None,
    ct_exclude=None,
    loss_weights=None,
    label_remap: torch.Tensor = None,
    enable_amp: bool = False,
    hierarchical_loss_fn=None,
    tumor_dataset_set: set = None,
    dct_config=None,
):
    """Process one batch through model.

    Args:
        loss_weights: dict with keys 'ct', 'domain', 'marker_pos', optionally 'tumor'
        label_remap: Lookup tensor to remap ct2idx values to compact 0-indexed labels.
            Identity for 0-indexed ct2idx (post-migration), kept for compatibility.
        enable_amp: Whether to use Automatic Mixed Precision (autocast)
        tumor_dataset_set: Set of dataset keys with tumor cells (for tumor loss masking)
        dct_config: TissueNetConfig (needed for tumor label index)
    """
    if loss_weights is None:
        loss_weights = DEFAULT_LOSS_WEIGHTS

    # Move tensors to device
    batch_data = batch_data.to(device)

    # Remap cell type labels to compact 0-indexed
    # (ct2idx values are 1-indexed, but FocalLoss/metrics need 0-indexed)
    orig_ct_idx = batch_data.ct_idx
    if label_remap is not None:
        compact_ct_idx = label_remap[batch_data.ct_idx]
    else:
        compact_ct_idx = batch_data.ct_idx

    # Forward pass + loss computations wrapped in autocast for AMP
    with torch.amp.autocast('cuda', enabled=enable_amp):
        ct_logits, domain_logits, marker_pos_logits, cls_embedding, _, tumor_logit = model(
            batch_data.sample,
            batch_data.spatial_context,
            batch_data.ch_idx,
            batch_data.mask,
            ct_exclude,
            domain_idx=batch_data.domain_idx,
        )

        # Defensive check: target class should never be in the exclude set
        # (ct_exclude is already None when no_ct_exclude=True, so checking is not None suffices)
        if ct_exclude is not None:
            for i, excl in enumerate(ct_exclude):
                if excl and compact_ct_idx[i].item() in excl:
                    import logging
                    logging.warning(f"Target class {compact_ct_idx[i].item()} is in ct_exclude for sample {i}. Check tissue_celltype_mapping.")

        # Cell type loss (FocalLoss with class weights)
        ct_loss = losses_metrics.ct_loss_fn(ct_logits, compact_ct_idx)
        if hierarchical_loss_fn is not None:
            ct_loss = ct_loss + hierarchical_loss_fn(ct_logits, compact_ct_idx)

        # Domain loss
        domain_loss = losses_metrics.domain_loss_fn(domain_logits, batch_data.domain_idx)

        # Marker positivity loss (independent BCE, only on valid channels)
        valid_channels = ~batch_data.mask  # True = real channel
        valid_mp = batch_data.marker_positivity_mask  # True = not "?"
        compute_loss_mask = valid_channels & valid_mp

        if compute_loss_mask.any():
            marker_pos_target = batch_data.marker_positivity[compute_loss_mask]
            mp_logits_sel = marker_pos_logits[compute_loss_mask]
            marker_pos_loss = F.binary_cross_entropy_with_logits(
                mp_logits_sel, marker_pos_target
            )
        else:
            marker_pos_loss = torch.tensor(0.0, device=device)

        # Tumor binary prediction loss (masked to tumor_datasets)
        tumor_loss = torch.tensor(0.0, device=device)
        if tumor_logit is not None and loss_weights.get("tumor", 0) > 0:
            tumor_raw = dct_config.ct2idx.get("Tumor", -1)
            if tumor_raw >= 0:
                # Remap raw ct2idx value to compact label space to match compact_ct_idx
                if label_remap is not None:
                    tumor_compact = label_remap[tumor_raw].item()
                else:
                    tumor_compact = tumor_raw
            else:
                tumor_compact = -1
            # Only compute tumor loss if the class exists in ct2idx
            if tumor_compact >= 0:
                tumor_label = (compact_ct_idx == tumor_compact).float()
                tumor_mask = torch.tensor(
                    [dn in tumor_dataset_set for dn in batch_data.dataset_name],
                    dtype=torch.float, device=device,
                )
                if tumor_mask.sum() > 0:
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        tumor_logit.squeeze(-1), tumor_label, reduction='none'
                    )
                    tumor_loss = (per_sample_loss * tumor_mask).sum() / tumor_mask.sum()

        # Combined loss with corrected weights
        loss = (
            loss_weights["ct"] * ct_loss
            + loss_weights["domain"] * domain_loss
            + loss_weights["marker_pos"] * marker_pos_loss
            + loss_weights.get("tumor", 0) * tumor_loss
        )

    # Compute probabilities for metrics
    probs = F.softmax(ct_logits, dim=-1)

    if predlogger is not None:
        # PredLogger stores original ct2idx labels (for CSV output)
        predlogger.log(
            labels=orig_ct_idx.detach().cpu().numpy(),
            probs=probs.cpu().detach().numpy(),
            cell_index=batch_data.cell_index.detach().cpu().numpy(),
            dataset_name=batch_data.dataset_name,
            fov_name=batch_data.fov_name,
        )

    # Update metrics (use compact labels)
    losses_metrics.acc_domain_metric(domain_logits, batch_data.domain_idx)

    # Per-marker MP metrics on channels with known MP labels only
    valid_mp_channels = valid_channels & valid_mp
    if valid_mp_channels.any():
        mp_pred = torch.sigmoid(marker_pos_logits[valid_mp_channels])
        mp_target = batch_data.marker_positivity[valid_mp_channels]
        mp_ch_indices = batch_data.ch_idx[valid_mp_channels]
        losses_metrics.mp_metrics.update(mp_pred, mp_target, mp_ch_indices)

    # Confusion matrix (all splits — required for macro/weighted accuracy computation)
    losses_metrics.conf_mat_ct_metric(probs, compact_ct_idx)

    return loss, {
        "loss": loss.item(),
        "ct_loss": ct_loss.item(),
        "domain_loss": domain_loss.item(),
        "marker_pos_loss": marker_pos_loss.item(),
        "tumor_loss": tumor_loss.item(),
    }


@click.command()
@click.option("--model_name", type=str, default="deepcell-types")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--enable_wandb", type=bool, default=False)
@click.option("--zarr_dir", type=str, default=str(DATA_DIR))
@click.option("--skip_datasets", type=str, multiple=True, default=[])
@click.option("--keep_datasets", type=str, multiple=True, default=[])
@click.option("--epochs", type=int, default=50)
@click.option("--batch_size", type=int, default=256)
@click.option("--lr", type=float, default=3e-4)
@click.option("--patience", type=int, default=10, help="Early stopping patience in validation checks (not epochs). Effective patience in epochs = patience * val_every.")
@click.option("--seed", type=int, default=42)
@click.option("--debug", is_flag=True, help="Enable anomaly detection")
@click.option("--num_workers", type=int, default=16)
@click.option("--svd_embeddings_path", type=str, default=None, help="Path to SVD-reduced embeddings .npz")
@click.option("--pretrained_path", type=str, default=None, help="Path to pre-trained backbone weights (from pretrain.py) — backbone-only load for fine-tuning")
@click.option("--resume_path", type=str, default=None, help="Path to a full training checkpoint to resume (restores model/optimizer/scheduler/scaler/epoch). Distinct from --pretrained_path which is backbone-only.")
@click.option("--num_dropout_channels", type=int, default=8, help="Number of channels to randomly drop during training")
@click.option("--split_mode", type=click.Choice(["fov", "patch"]), default="fov",
              help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)")
@click.option("--split_file", type=str, default=None,
              help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)")
@click.option("--max_samples_per_epoch", type=int, default=None, help="Cap samples per epoch (e.g. 500000)")
@click.option("--max_val_samples", type=int, default=None, help="Cap val set size (fixed random subset, e.g. 200000)")
@click.option("--skip_distance_transform", is_flag=True, help="Skip distance transform computation (zeros instead)")
@click.option("--val_every", type=int, default=1, help="Validate every N epochs (default 1, use 10 to match CellSighter). Note: --patience counts validation checks, so effective patience in training epochs = patience * val_every.")
@click.option("--domain_weight", type=float, default=0.1, help="Weight for domain adversarial loss (0 = disabled). Default 0.1 enables DANN as part of the canonical recipe.")
@click.option("--marker_pos_weight", type=float, default=1.0, help="Weight for marker positivity auxiliary loss (0 = disabled)")
@click.option("--tumor_weight", type=float, default=0.0, help="Weight for binary tumor prediction loss (0 = disabled)")
@click.option("--no_ct_exclude", is_flag=True, help="Disable tissue-aware cell type exclusion (matches baseline behavior)")
@click.option("--no_class_weights", is_flag=True, help="Disable per-class weights in FocalLoss (use when WeightedRandomSampler is active to avoid double-weighting)")
@click.option("--min_channels", type=int, default=0, help="Min model-visible marker channels per dataset (default 0 = no filter; retained as a no-op for legacy split-file compatibility)")
@click.option("--hierarchical_weight", type=float, default=0.0, help="Weight for hierarchical coarse-grained loss (0 = disabled)")
@click.option("--enable_amp", type=bool, default=True, help="Enable Automatic Mixed Precision (AMP) training (~2x speedup on CUDA, disabled automatically on CPU)")
@click.option("--spatial_pool_size", type=int, default=1, help="Spatial pooling grid size (1=global avg, 4=4x4 spatial)")
@click.option("--focal_gamma", type=float, default=2.0, help="FocalLoss gamma (0=CE, 2=default focal)")
@click.option("--warmup_pct", type=float, default=None, help="Override warmup percentage for OneCycleLR (default from config.py)")
@click.option("--resnet_channels", type=int, default=48, help="Base channels for PerChannelResNet (canonical: 48)")
@click.option("--mean_intensity_mode", type=click.Choice(["none", "cls_residual", "per_channel", "both"]), default="cls_residual",
              help="Add a mean-intensity-per-channel side input. Canonical: cls_residual=scatter intensities to global marker positions, MLP→add to CLS. Other modes: per_channel=project per-channel scalar intensity into d_model and add to fused tokens before the transformer; both=apply both; none=disable. All branches zero-init their output projection so warm-start from a baseline ckpt preserves predictions at step 0.")
@click.option("--freeze_backbone", is_flag=True,
              help="Freeze everything except the mean-intensity branches (and re-enable only intensity_cls_branch / intensity_per_channel_proj). Use with a warm-started ckpt to train only the new side input.")
@click.option("--unfreeze_ct_head", is_flag=True,
              help="Also keep the CT classifier head and CLS-token / final-norm trainable when --freeze_backbone is set. "
                   "Lets the CT head adapt to the new adapter output without unfreezing the full transformer / per-channel encoder backbone.")
def main(
    model_name, device_num, enable_wandb, zarr_dir, skip_datasets, keep_datasets,
    epochs, batch_size, lr, patience, seed, debug, num_workers,
    svd_embeddings_path, pretrained_path, resume_path, num_dropout_channels, split_mode,
    split_file,
    max_samples_per_epoch, max_val_samples, skip_distance_transform, val_every,
    domain_weight, marker_pos_weight, tumor_weight, no_ct_exclude, no_class_weights, min_channels, hierarchical_weight, enable_amp,
    spatial_pool_size, focal_gamma, warmup_pct, resnet_channels,
    mean_intensity_mode, freeze_backbone, unfreeze_ct_head,
):
    # Seed everything
    seed_everything(seed)

    # Lazy wandb init
    import wandb
    if enable_wandb:
        wandb.login()
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "deepcell-types"),
        dir="wandb_tmp",
        job_type="train",
        mode="online" if enable_wandb else "disabled",
        name=model_name,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "seed": seed,
            "architecture": "CellTypeAnnotator",
            "split_mode": split_mode,
            "domain_weight": domain_weight,
            "tumor_weight": tumor_weight,
            "no_ct_exclude": no_ct_exclude,
            "no_class_weights": no_class_weights,
        },
    )

    if debug:
        torch.autograd.set_detect_anomaly(True)

    device = torch.device(device_num)
    # AMP is only supported on CUDA; disable automatically on CPU
    enable_amp = enable_amp and device.type == 'cuda'
    dct_config = TissueNetConfig(zarr_dir)
    d_model = 256

    # Fail fast if tumor loss is requested but Tumor class is absent from the archive
    if tumor_weight > 0 and "Tumor" not in dct_config.ct2idx:
        raise ValueError(
            "tumor_weight > 0 but 'Tumor' is not in dct_config.ct2idx; "
            "cannot compute tumor loss on an archive with no Tumor class."
        )

    # Load marker embeddings
    marker_embeddings = dct_config.load_marker_embeddings_array(svd_path=svd_embeddings_path)

    use_cuda = device.type == "cuda"

    # Create dataloaders
    train_loader, val_loader, metadata = create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        skip_datasets=list(skip_datasets) if skip_datasets else None,
        keep_datasets=list(keep_datasets) if keep_datasets else None,
        batch_size=batch_size,
        num_dropout_channels=num_dropout_channels,
        num_workers=num_workers,
        only_test=False,
        use_fov_splits=(split_mode == "fov"),
        train_ratio=0.8,
        seed=seed,
        use_weighted_sampler=True,
        split_file=split_file,
        skip_distance_transform=skip_distance_transform,
        persistent_workers=num_workers > 0,
        max_samples_per_epoch=max_samples_per_epoch,
        max_val_samples=max_val_samples,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=use_cuda,
        min_channels=min_channels,
    )

    wandb.config.update(metadata)
    print(f"Train: {metadata.get('num_train', '?')}, Val: {metadata.get('num_val', '?')}")

    # Build label remap (ct2idx values are 1-indexed, need 0-indexed for loss/metrics)
    # Move to device once to avoid per-batch CPU→GPU transfer
    label_remap = build_label_remap(dct_config.ct2idx).to(device)

    # Compute class weights from training data (in compact label space)
    # Defensive layout checks: create_dataloader wraps FullImageDataset in
    # Subset, then AugmentedDataset. Raise explicitly (not `assert`) so a
    # silent break under `python -O` (which strips assert statements) can't
    # produce a cryptic AttributeError several lines later.
    if not isinstance(train_loader.dataset, AugmentedDataset):
        raise TypeError(
            f"Expected train_loader.dataset to be AugmentedDataset, "
            f"got {type(train_loader.dataset).__name__}. "
            "create_dataloader wrapping order may have changed."
        )
    if not isinstance(train_loader.dataset.dataset, torch.utils.data.Subset):
        raise TypeError(
            f"Expected train_loader.dataset.dataset to be torch.utils.data.Subset, "
            f"got {type(train_loader.dataset.dataset).__name__}. "
            "create_dataloader wrapping order may have changed."
        )
    if not isinstance(train_loader.dataset.dataset.dataset, FullImageDataset):
        raise TypeError(
            f"Expected train_loader.dataset.dataset.dataset to be FullImageDataset, "
            f"got {type(train_loader.dataset.dataset.dataset).__name__}. "
            "create_dataloader wrapping order may have changed."
        )
    train_dataset_ref = train_loader.dataset.dataset.dataset  # Augmented -> Subset -> FullImageDataset
    class_weights = compute_class_weights(dct_config, train_dataset_ref, label_remap).to(device)

    # Build model
    model = create_model(dct_config, marker_embeddings, d_model=d_model,
                         spatial_pool_size=spatial_pool_size,
                         resnet_base_channels=resnet_channels,
                         tumor_head=(tumor_weight > 0),
                         mean_intensity_mode=mean_intensity_mode)
    # Load pre-trained backbone weights (from masked marker pre-training)
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pre-trained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
        # Accept both the legacy plain-state_dict and the new bundled checkpoint
        # (which stores the backbone under the "model" key).
        if isinstance(pretrained_state, dict) and "model" in pretrained_state and isinstance(pretrained_state["model"], dict):
            pretrained_state = pretrained_state["model"]
        # Load only matching keys (skip MaskedMarkerHead / optimizer / scheduler keys)
        model_state = model.state_dict()
        loaded = 0
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded}/{len(model_state)} parameters from pre-trained model")

    model.to(device)

    if freeze_backbone:
        n_total = sum(p.numel() for p in model.parameters())
        for p in model.parameters():
            p.requires_grad = False
        # Re-enable the mean-intensity branches (always, under freeze_backbone)
        unfreeze_modules = {"intensity_cls_branch", "intensity_per_channel_proj"}
        # Optionally also unfreeze CT-task layers (head + CLS token + final norm).
        # Leaves the heavy backbone (transformer, per-channel encoder, marker
        # embedder LoRA, spatial encoder) still frozen.
        if unfreeze_ct_head:
            unfreeze_modules.update({"ct_head", "final_norm"})
        for name, module in model.named_modules():
            if name in unfreeze_modules:
                for p in module.parameters():
                    p.requires_grad = True
        # cls_token is a top-level Parameter (not a Module), handle separately
        if unfreeze_ct_head and hasattr(model, "cls_token"):
            model.cls_token.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  freeze_backbone: trainable params = {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)"
              + (" (with --unfreeze_ct_head)" if unfreeze_ct_head else ""))
        if n_trainable == 0:
            raise click.UsageError("--freeze_backbone with no branch trainable — did you also pass --mean_intensity_mode?")

    summary(model, col_names=["trainable"])

    # Optimizer + scheduler
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=lr, weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    effective_warmup_pct = warmup_pct if warmup_pct is not None else WARMUP_PCT
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=effective_warmup_pct,
        anneal_strategy="cos",
    )

    # GradScaler for AMP (no-op when enable_amp=False)
    scaler = torch.amp.GradScaler('cuda', enabled=enable_amp)

    # ---------- Checkpoint helpers (R4 H1: full training-state checkpoints) ----------
    # Snapshot of training-time config that must match on resume. If these differ we
    # can't restore optimizer/scheduler state sanely, so we raise.
    # CKPT_CONFIG is bundled into every saved checkpoint AND written to a
    # sidecar JSON at models/model_<name>_config.json. The set of keys here
    # must include EVERY argument that materially changes training behaviour
    # — otherwise a re-run with a stored checkpoint can silently diverge.
    CKPT_CONFIG = {
        "d_model": d_model,
        "resnet_channels": resnet_channels,
        "use_conditioned_mp_head": True,  # create_model default; toggled only at inference
        "n_celltypes": dct_config.NUM_CELLTYPES,
        "format_version": "1.1",
        "seed": seed,
        # Data / split provenance — without these, "reproduce this run" is
        # underspecified because two different splits or two different
        # embedding files trivially change the numbers.
        "split_file": split_file,
        "split_mode": split_mode,
        "svd_embeddings_path": svd_embeddings_path,
        "min_channels": min_channels,
        "max_samples_per_epoch": max_samples_per_epoch,
        "max_val_samples": max_val_samples,
        # Optimization-shape hyperparameters that move metrics by ≥0.5pp.
        "focal_gamma": focal_gamma,
        "warmup_pct": warmup_pct,
        "lr": lr,
        "batch_size": batch_size,
        "domain_weight": domain_weight,
        "marker_pos_weight": marker_pos_weight,
        "tumor_weight": tumor_weight,
        "hierarchical_weight": hierarchical_weight,
        "no_ct_exclude": bool(no_ct_exclude),
        "no_class_weights": bool(no_class_weights),
        "num_dropout_channels": num_dropout_channels,
        "spatial_pool_size": spatial_pool_size,
        "val_every": val_every,
        "enable_amp": bool(enable_amp),
        "skip_distance_transform": bool(skip_distance_transform),
    }

    # Sidecar JSON for offline lookups without loading the (~3-4GB) checkpoint
    # pickle. Written once at training start so it exists even if the run
    # crashes before the first checkpoint save.
    config_sidecar = Path("models") / f"model_{model_name}_config.json"
    config_sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(config_sidecar, "w") as f:
        json.dump(CKPT_CONFIG, f, indent=2, default=str)

    def build_checkpoint(epoch_val, best_val, epochs_no_improve):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch_val,
            # Best validation macro-F1 (the repo's single CT selection metric).
            # The legacy "best_val_macro_acc" key is still read on resume for
            # backwards-compat with checkpoints written before the switch.
            "best_val_macro_f1": best_val,
            "best_metric": "macro_f1",
            "epochs_without_improvement": epochs_no_improve,
            "config": CKPT_CONFIG,
            # Bundle the canonical-channel registry so inference can size
            # marker2idx without consulting a vendored YAML or the archive.
            "canonical_channels": list(dct_config.marker2idx.keys()),
        }

    # Loss and metrics
    # When WeightedRandomSampler is active, avoid double-weighting by not using
    # class weights in FocalLoss. Use --no_class_weights to match CellSighter behavior.
    focal_alpha = None if no_class_weights else class_weights
    # Build compact ct2idx for hierarchical evaluation (maps names to confusion matrix indices)
    compact_ct2idx = {name: label_remap[idx].item() for name, idx in dct_config.ct2idx.items()}
    losses_metrics = LossesAndMetrics(
        ct_loss_fn=FocalLoss(alpha=focal_alpha, gamma=focal_gamma),
        domain_loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.01),
        marker_pos_loss_fn=None,  # handled inline in forward_one_batch
        acc_domain_metric=MulticlassAccuracy(num_classes=dct_config.NUM_DOMAINS).to(device),
        conf_mat_ct_metric=MulticlassConfusionMatrix(
            num_classes=dct_config.NUM_CELLTYPES
        ).to(device),
        mp_metrics=MPMetricsTracker(),
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    hierarchical_loss_fn = None
    if hierarchical_weight > 0:
        from deepcell_types.training.losses import HierarchicalLoss
        from deepcell_types.training.config import CONFIG_DIR as _TRAIN_CONFIG_DIR
        hierarchical_loss_fn = HierarchicalLoss(
            config_path=str(_TRAIN_CONFIG_DIR / "combined_celltypes.yaml"),
            ct2idx=dct_config.ct2idx,
            weight=hierarchical_weight,
        ).to(device)

    loss_weights = {**DEFAULT_LOSS_WEIGHTS, "domain": domain_weight, "marker_pos": marker_pos_weight, "tumor": tumor_weight}
    tumor_dataset_set = dct_config.tumor_datasets if tumor_weight > 0 else set()
    wandb.watch(model)

    # Early stopping state
    best_val_macro_f1 = 0.0
    epochs_without_improvement = 0
    best_model_path = Path(f"models/model_{model_name}_best.pt")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- R4 H1: full training-state resume ----------
    start_epoch = 0
    if resume_path is not None:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"--resume_path {resume_path} does not exist")
        logger.info("Resuming from %s", resume_path)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=True)

        if not isinstance(resume_ckpt, dict) or "optimizer" not in resume_ckpt:
            # Legacy checkpoint (only `model` key, or plain state_dict): fall back to
            # backbone-only semantics. Log a clear warning that optimizer state is lost.
            logger.warning(
                "Legacy checkpoint at %s lacks optimizer state; "
                "falling back to --pretrained_path semantics (backbone-only load, "
                "optimizer/scheduler/scaler/epoch NOT restored).",
                resume_path,
            )
            legacy_state = resume_ckpt["model"] if isinstance(resume_ckpt, dict) and "model" in resume_ckpt else resume_ckpt
            model_state = model.state_dict()
            loaded = 0
            for k, v in legacy_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded += 1
            model.load_state_dict(model_state)
            logger.info("Loaded %d/%d params (backbone-only).", loaded, len(model_state))
        else:
            # Full checkpoint: validate config
            ckpt_config = resume_ckpt.get("config", {})
            for key in ("resnet_channels", "d_model"):
                want = CKPT_CONFIG[key]
                have = ckpt_config.get(key)
                if have is not None and have != want:
                    raise ValueError(
                        f"--resume_path config mismatch for '{key}': checkpoint has {have!r}, "
                        f"current run has {want!r}. Cannot restore optimizer/scheduler state "
                        f"when model architecture differs. Use --pretrained_path for "
                        f"backbone-only load, or re-run with matching CLI args."
                    )
            if ckpt_config.get("seed") is not None and ckpt_config["seed"] != seed:
                logger.warning(
                    "Resumed checkpoint was trained with seed=%s but current run uses seed=%s; "
                    "RNG state will diverge from original run.",
                    ckpt_config["seed"], seed,
                )

            model.load_state_dict(resume_ckpt["model"])
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            start_epoch = int(resume_ckpt["epoch"]) + 1
            # New key "best_val_macro_f1"; fall back to the legacy
            # "best_val_macro_acc" key for checkpoints written before the switch.
            best_val_macro_f1 = float(
                resume_ckpt.get(
                    "best_val_macro_f1",
                    resume_ckpt.get("best_val_macro_acc", 0.0),
                )
            )
            epochs_without_improvement = int(resume_ckpt.get("epochs_without_improvement", 0))
            logger.info(
                "Resumed: start_epoch=%d, best_val_macro_f1=%.4f, epochs_without_improvement=%d",
                start_epoch, best_val_macro_f1, epochs_without_improvement,
            )
            if start_epoch >= epochs:
                logger.warning(
                    "start_epoch (%d) >= --epochs (%d); nothing to do. Increase --epochs to continue training.",
                    start_epoch, epochs,
                )

    # Training loop
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", initial=start_epoch, total=epochs):
        # ===================== TRAIN =====================
        model.train()
        train_losses = defaultdict(list)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} (train)", leave=False):
            batch_data = BatchData(*batch)
            ct_exclude = None if no_ct_exclude else get_tissue_ct_exclude(batch_data, dct_config, label_remap)

            loss, batch_losses = forward_one_batch(
                batch_data, device, model, "train", losses_metrics,
                ct_exclude=ct_exclude, loss_weights=loss_weights,
                label_remap=label_remap,
                enable_amp=enable_amp,
                hierarchical_loss_fn=hierarchical_loss_fn,
                tumor_dataset_set=tumor_dataset_set,
                dct_config=dct_config,
            )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            # Only advance scheduler if optimizer actually stepped. When AMP detects
            # inf/NaN grads, scaler.step() skips optimizer.step() and scaler.update()
            # drops the scale factor. Advancing OneCycleLR unconditionally would
            # desynchronize LR schedule from actual optimizer steps over 50 epochs.
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                scheduler.step()

            for k, v in batch_losses.items():
                train_losses[k].append(v)

        # Log training metrics
        train_epoch_metrics = losses_metrics.compute()
        log_epoch_metrics(train_epoch_metrics, "train")
        wandb.log({
            "train/loss_epoch": np.mean(train_losses["loss"]),
            "train/ct_loss_epoch": np.mean(train_losses["ct_loss"]),
            "train/domain_loss_epoch": np.mean(train_losses["domain_loss"]),
            "train/marker_pos_loss_epoch": np.mean(train_losses["marker_pos_loss"]),
            "train/tumor_loss_epoch": np.mean(train_losses["tumor_loss"]),
            "train/lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })
        losses_metrics.reset_metrics()

        # ===================== VALIDATION =====================
        is_val_epoch = ((epoch + 1) % val_every == 0) or (epoch + 1 == epochs)
        if is_val_epoch:
            model.eval()
            val_losses = defaultdict(list)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} (val)", leave=False):
                    batch_data = BatchData(*batch)
                    ct_exclude = None if no_ct_exclude else get_tissue_ct_exclude(batch_data, dct_config, label_remap)

                    loss, batch_losses = forward_one_batch(
                        batch_data, device, model, "val", losses_metrics,
                        ct_exclude=ct_exclude, loss_weights=loss_weights,
                        label_remap=label_remap,
                        enable_amp=enable_amp,
                        hierarchical_loss_fn=hierarchical_loss_fn,
                        tumor_dataset_set=tumor_dataset_set,
                        dct_config=dct_config,
                    )

                    for k, v in batch_losses.items():
                        val_losses[k].append(v)

            val_epoch_metrics = losses_metrics.compute()
            log_epoch_metrics(val_epoch_metrics, "val")
            wandb.log({
                "val/loss_epoch": np.mean(val_losses["loss"]),
                "val/ct_loss_epoch": np.mean(val_losses["ct_loss"]),
                "val/tumor_loss_epoch": np.mean(val_losses["tumor_loss"]),
                "epoch": epoch,
            })
            losses_metrics.reset_metrics()

            val_macro_f1 = val_epoch_metrics["ct_macro_f1"]
            print(
                f"Epoch {epoch}: train_macro_f1={train_epoch_metrics['ct_macro_f1']:.4f}, "
                f"val_macro_f1={val_macro_f1:.4f}"
            )

            # Model selection — macro-F1 is the repo's single CT metric. It is
            # robust to class-balance gaming where over-predicting majority
            # classes inflates per-class accuracy.
            current = val_macro_f1
            if current > best_val_macro_f1:
                best_val_macro_f1 = current
                epochs_without_improvement = 0
                # Atomic save: write to .tmp then os.replace to avoid corrupt files on SIGTERM
                tmp_path = best_model_path.with_suffix(best_model_path.suffix + ".tmp")
                torch.save(
                    build_checkpoint(epoch, best_val_macro_f1, epochs_without_improvement),
                    tmp_path,
                )
                os.replace(tmp_path, best_model_path)
                print(f"  -> New best model saved (macro_f1={current:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break
        else:
            print(f"Epoch {epoch}: train_macro_f1={train_epoch_metrics['ct_macro_f1']:.4f}")

        # Save checkpoint every epoch (atomic: .tmp → os.replace)
        epoch_path = Path(f"models/model_{model_name}_epoch_{epoch}.pt")
        epoch_tmp = epoch_path.with_suffix(epoch_path.suffix + ".tmp")
        torch.save(
            build_checkpoint(epoch, best_val_macro_f1, epochs_without_improvement),
            epoch_tmp,
        )
        os.replace(epoch_tmp, epoch_path)

    # ===================== TEST (on validation set with best model) =====================
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Final eval ALWAYS runs on the full val set (no max_val_samples cap), even when
    # per-epoch val used a cap for speed. This keeps the headline test number
    # apples-to-apples with baseline runs (which never cap their val set).
    if max_val_samples is not None:
        print(f"\nBuilding full-val dataloader for final eval (per-epoch val was capped at {max_val_samples})...")
        _, final_val_loader, _ = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=list(skip_datasets) if skip_datasets else None,
            keep_datasets=list(keep_datasets) if keep_datasets else None,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=num_workers,
            only_test=False,
            use_fov_splits=(split_mode == "fov"),
            train_ratio=0.8,
            seed=seed,
            use_weighted_sampler=False,
            split_file=split_file,
            skip_distance_transform=skip_distance_transform,
            persistent_workers=num_workers > 0,
            max_samples_per_epoch=None,
            max_val_samples=None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            pin_memory=use_cuda,
            min_channels=min_channels,
        )
    else:
        final_val_loader = val_loader

    predlogger = PredLogger(dct_config.ct2idx)
    with torch.no_grad():
        for batch in tqdm(final_val_loader, desc="Final eval"):
            batch_data = BatchData(*batch)
            ct_exclude = None if no_ct_exclude else get_tissue_ct_exclude(batch_data, dct_config, label_remap)

            loss, _ = forward_one_batch(
                batch_data, device, model, "test", losses_metrics,
                predlogger=predlogger, ct_exclude=ct_exclude,
                loss_weights=loss_weights, label_remap=label_remap,
                enable_amp=enable_amp,
                hierarchical_loss_fn=hierarchical_loss_fn,
                tumor_dataset_set=tumor_dataset_set,
                dct_config=dct_config,
            )

    final_metrics = losses_metrics.compute()
    print(f"\nFinal metrics (best model): {final_metrics}")
    log_epoch_metrics(final_metrics, "test")

    log_confusion_matrix(
        losses_metrics.conf_mat_ct_metric, "test",
        sorted(dct_config.ct2idx, key=dct_config.ct2idx.get),
        metric_name="confusion_matrix_ct",
    )
    losses_metrics.reset_metrics()

    output_path = Path(f"output/{model_name}_val_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predlogger.save(output_path)

    run.finish()


if __name__ == "__main__":
    main()
