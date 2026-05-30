"""Masked Marker Pre-training for CellTypeAnnotator.

Self-supervised pre-training that learns cross-marker biological relationships
(e.g., CD3-CD4 co-occurrence, CD4-CD8 mutual exclusion) by randomly masking
30% of marker channels and training the model to predict their mean expression
from the remaining channels.

Analogous to Masked Language Modeling (BERT) but for continuous marker intensity
values in multiplexed imaging data.

Usage:
    python pretrain.py --model_name pretrain --epochs 20 --device_num cuda:0

Then fine-tune with:
    python train.py --model_name finetuned --pretrained_path models/model_pretrain_best.pt
"""

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

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import create_dataloader
from deepcell_types.model import create_model, MaskedMarkerHead, mask_marker_channels
from deepcell_types.training.utils import BatchData, seed_everything, load_matching_state_dict

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))


@click.command()
@click.option("--model_name", type=str, default="pretrain")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--enable_wandb", type=bool, default=False)
@click.option("--zarr_dir", type=str, default=str(DATA_DIR))
@click.option("--skip_datasets", type=str, multiple=True, default=[])
@click.option("--keep_datasets", type=str, multiple=True, default=[])
@click.option("--epochs", type=int, default=20)
@click.option("--batch_size", type=int, default=256)
@click.option("--lr", type=float, default=3e-4)
@click.option("--patience", type=int, default=5, help="Early stopping patience")
@click.option("--seed", type=int, default=42)
@click.option("--num_workers", type=int, default=16)
@click.option(
    "--mask_ratio", type=float, default=0.3, help="Fraction of valid channels to mask"
)
@click.option("--svd_embeddings_path", type=str, default=None)
@click.option(
    "--recon_weight", type=float, default=1.0, help="Weight for reconstruction loss"
)
@click.option(
    "--marker_pos_weight",
    type=float,
    default=0.5,
    help="Weight for marker positivity auxiliary loss",
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
    "--enable_amp",
    type=bool,
    default=True,
    help="Enable Automatic Mixed Precision (AMP) training (~2x speedup on CUDA, disabled automatically on CPU)",
)
@click.option(
    "--min_channels",
    type=int,
    default=0,
    help="Min model-visible marker channels per dataset (default 0 = no filter)",
)
@click.option(
    "--max_samples_per_epoch",
    type=int,
    default=500000,
    help="Cap training samples per epoch (random subsample)",
)
@click.option(
    "--resume_path",
    type=str,
    default=None,
    help="Path to a full training checkpoint to resume (restores model/optimizer/scheduler/scaler/epoch).",
)
def main(
    model_name,
    device_num,
    enable_wandb,
    zarr_dir,
    skip_datasets,
    keep_datasets,
    epochs,
    batch_size,
    lr,
    patience,
    seed,
    num_workers,
    mask_ratio,
    svd_embeddings_path,
    recon_weight,
    marker_pos_weight,
    split_mode,
    split_file,
    enable_amp,
    min_channels,
    max_samples_per_epoch,
    resume_path,
):
    seed_everything(seed)

    import wandb

    if enable_wandb:
        wandb.login()
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "deepcell-types"),
        dir="wandb_tmp",
        job_type="pretrain",
        mode="online" if enable_wandb else "disabled",
        name=model_name,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "mask_ratio": mask_ratio,
            "recon_weight": recon_weight,
            "marker_pos_weight": marker_pos_weight,
            "architecture": "CellTypeAnnotator_pretrain",
            "split_mode": split_mode,
        },
    )

    device = torch.device(device_num)
    # AMP is only supported on CUDA; disable automatically on CPU
    enable_amp = enable_amp and device.type == "cuda"
    dct_config = TissueNetConfig(zarr_dir)
    d_model = 256

    # Load marker embeddings
    marker_embeddings = dct_config.load_marker_embeddings_array(
        svd_path=svd_embeddings_path
    )

    use_cuda = device.type == "cuda"

    # Create dataloaders (no label supervision needed, but we still use FOV splits)
    train_loader, val_loader, metadata = create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        skip_datasets=list(skip_datasets) if skip_datasets else None,
        keep_datasets=list(keep_datasets) if keep_datasets else None,
        batch_size=batch_size,
        num_dropout_channels=0,  # No dropout during pretraining — masking handles regularization
        num_workers=num_workers,
        only_test=False,
        use_fov_splits=(split_mode == "fov"),
        train_ratio=0.8,
        seed=seed,
        use_weighted_sampler=True,  # Enables RandomSampler with max_samples_per_epoch
        split_file=split_file,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=use_cuda,
        min_channels=min_channels,
        max_samples_per_epoch=max_samples_per_epoch,
    )

    print(
        f"Pre-training data: {metadata.get('num_train', '?')} train, {metadata.get('num_val', '?')} val"
    )

    # Build model + reconstruction head
    model = create_model(dct_config, marker_embeddings, d_model=d_model)
    recon_head = MaskedMarkerHead(d_model=d_model)

    model.to(device)
    recon_head.to(device)
    summary(model, col_names=["trainable"])
    print(
        f"Reconstruction head params: {sum(p.numel() for p in recon_head.parameters()):,}"
    )

    # Optimizer: jointly optimize model backbone + reconstruction head
    all_params = list(model.parameters()) + list(recon_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
    )

    # GradScaler for AMP (no-op when enable_amp=False)
    scaler = torch.amp.GradScaler("cuda", enabled=enable_amp)

    # ---------- Checkpoint helpers (R4 H1: full training-state checkpoints) ----------
    CKPT_CONFIG = {
        "d_model": d_model,
        "resnet_channels": 32,  # pretrain uses create_model defaults
        "use_conditioned_mp_head": True,
        "n_celltypes": dct_config.NUM_CELLTYPES,
        "format_version": "1.0",
        "seed": seed,
    }

    def build_checkpoint(epoch_val, best_val, epochs_no_improve):
        return {
            "model": model.state_dict(),
            "recon_head": recon_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch_val,
            "best_val_loss": best_val,
            "epochs_without_improvement": epochs_no_improve,
            "config": CKPT_CONFIG,
        }

    # Early stopping
    best_val_loss = float("inf")
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
            # Legacy pretrain checkpoint is a plain state_dict (model backbone only).
            logger.warning(
                "Legacy checkpoint at %s lacks optimizer state; "
                "loading model backbone only (optimizer/scheduler/scaler/epoch NOT restored).",
                resume_path,
            )
            legacy_state = resume_ckpt["model"] if isinstance(resume_ckpt, dict) and "model" in resume_ckpt else resume_ckpt
            loaded = load_matching_state_dict(model, legacy_state)
            logger.info("Loaded %d/%d params (backbone-only).", loaded, len(model.state_dict()))
        else:
            ckpt_config = resume_ckpt.get("config", {})
            for key in ("resnet_channels", "d_model"):
                want = CKPT_CONFIG[key]
                have = ckpt_config.get(key)
                if have is not None and have != want:
                    raise ValueError(
                        f"--resume_path config mismatch for '{key}': checkpoint has {have!r}, "
                        f"current run has {want!r}. Cannot restore optimizer/scheduler state "
                        f"when model architecture differs."
                    )
            if ckpt_config.get("seed") is not None and ckpt_config["seed"] != seed:
                logger.warning(
                    "Resumed checkpoint was trained with seed=%s but current run uses seed=%s.",
                    ckpt_config["seed"],
                    seed,
                )
            model.load_state_dict(resume_ckpt["model"])
            if "recon_head" in resume_ckpt:
                recon_head.load_state_dict(resume_ckpt["recon_head"])
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            start_epoch = int(resume_ckpt["epoch"]) + 1
            best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf")))
            epochs_without_improvement = int(
                resume_ckpt.get("epochs_without_improvement", 0)
            )
            logger.info(
                "Resumed: start_epoch=%d, best_val_loss=%.6f, epochs_without_improvement=%d",
                start_epoch,
                best_val_loss,
                epochs_without_improvement,
            )

    for epoch in tqdm(
        range(start_epoch, epochs),
        desc="Pre-train epochs",
        initial=start_epoch,
        total=epochs,
    ):
        # ===================== TRAIN =====================
        model.train()
        recon_head.train()
        train_losses = defaultdict(list)
        train_valid_mp_batches = 0  # R5 L1: track batches where MP loss was non-trivial
        val_valid_mp_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} (pretrain)", leave=False):
            batch_data = BatchData(*batch).to(device)

            sample = batch_data.sample
            spatial = batch_data.spatial_context
            ch_idx = batch_data.ch_idx
            pad_mask = batch_data.mask
            mp_target = batch_data.marker_positivity
            mp_validity = batch_data.marker_positivity_mask

            # Mask channels
            masked_sample, masked_indices, mean_expr = mask_marker_channels(
                sample, pad_mask, mask_ratio=mask_ratio
            )
            mean_expr = mean_expr.to(device)

            # Forward pass + loss computations wrapped in autocast for AMP
            with torch.amp.autocast("cuda", enabled=enable_amp):
                # Forward pass (with masked input)
                _, _, marker_pos_logits, _, channel_outputs = model(
                    masked_sample,
                    spatial,
                    ch_idx,
                    pad_mask,
                    domain_idx=batch_data.domain_idx,
                )

                # Reconstruction loss: MSE on masked channels only
                pred_expr = recon_head(channel_outputs)
                if masked_indices.any():
                    recon_loss = F.mse_loss(
                        pred_expr[masked_indices], mean_expr[masked_indices]
                    )
                else:
                    recon_loss = torch.tensor(0.0, device=device)

                # Auxiliary: marker positivity on UNMASKED valid channels
                # This provides additional biological signal during pretraining
                valid_unmasked = (~pad_mask) & (~masked_indices) & mp_validity
                if valid_unmasked.any() and marker_pos_weight > 0:
                    mp_logits = marker_pos_logits[valid_unmasked]
                    mp_tgt = mp_target[valid_unmasked]
                    mp_loss = F.binary_cross_entropy_with_logits(mp_logits, mp_tgt)
                    train_valid_mp_batches += 1
                else:
                    mp_loss = torch.tensor(0.0, device=device)

                loss = recon_weight * recon_loss + marker_pos_weight * mp_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            # Only advance scheduler if optimizer actually stepped. When AMP detects
            # inf/NaN grads, scaler.step() skips optimizer.step() and scaler.update()
            # drops the scale factor. Advancing OneCycleLR unconditionally would
            # desynchronize LR schedule from actual optimizer steps.
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                scheduler.step()

            train_losses["loss"].append(loss.item())
            train_losses["recon_loss"].append(recon_loss.item())
            train_losses["mp_loss"].append(mp_loss.item())

        train_summary = {k: np.mean(v) for k, v in train_losses.items()}
        print(f"Epoch {epoch} train: {train_summary}")
        wandb.log({f"pretrain/train/{k}": v for k, v in train_summary.items()})
        wandb.log({"pretrain/lr": scheduler.get_last_lr()[0], "epoch": epoch})

        # ===================== VALIDATION =====================
        model.eval()
        recon_head.eval()
        val_losses = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} (val)", leave=False):
                batch_data = BatchData(*batch).to(device)

                sample = batch_data.sample
                spatial = batch_data.spatial_context
                ch_idx = batch_data.ch_idx
                pad_mask = batch_data.mask
                mp_target = batch_data.marker_positivity
                mp_validity = batch_data.marker_positivity_mask

                masked_sample, masked_indices, mean_expr = mask_marker_channels(
                    sample, pad_mask, mask_ratio=mask_ratio
                )
                mean_expr = mean_expr.to(device)

                _, _, marker_pos_logits, _, channel_outputs = model(
                    masked_sample,
                    spatial,
                    ch_idx,
                    pad_mask,
                    domain_idx=batch_data.domain_idx,
                )

                pred_expr = recon_head(channel_outputs)
                if masked_indices.any():
                    recon_loss = F.mse_loss(
                        pred_expr[masked_indices], mean_expr[masked_indices]
                    )
                else:
                    recon_loss = torch.tensor(0.0, device=device)

                valid_unmasked = (~pad_mask) & (~masked_indices) & mp_validity
                if valid_unmasked.any() and marker_pos_weight > 0:
                    mp_logits = marker_pos_logits[valid_unmasked]
                    mp_tgt = mp_target[valid_unmasked]
                    mp_loss = F.binary_cross_entropy_with_logits(mp_logits, mp_tgt)
                    val_valid_mp_batches += 1
                else:
                    mp_loss = torch.tensor(0.0, device=device)

                loss = recon_weight * recon_loss + marker_pos_weight * mp_loss

                val_losses["loss"].append(loss.item())
                val_losses["recon_loss"].append(recon_loss.item())
                val_losses["mp_loss"].append(mp_loss.item())

        val_summary = {k: np.mean(v) for k, v in val_losses.items()}
        print(f"Epoch {epoch} val: {val_summary}")
        wandb.log({f"pretrain/val/{k}": v for k, v in val_summary.items()})

        # R5 L1: warn if MP-loss mask was empty for the entire epoch (train or val);
        # otherwise the reported mp_loss average is effectively zero with no signal.
        if marker_pos_weight > 0:
            if train_valid_mp_batches == 0:
                logger.warning(
                    "Pretrain epoch %d saw 0 valid MP samples in training; MP-loss contribution is effectively zero",
                    epoch,
                )
            if val_valid_mp_batches == 0:
                logger.warning(
                    "Pretrain epoch %d saw 0 valid MP samples in validation; MP-loss contribution is effectively zero",
                    epoch,
                )

        # Model selection on validation reconstruction loss
        val_loss = val_summary["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Atomic save: write to .tmp then os.replace to avoid corrupt files on SIGTERM.
            # Save FULL state (model + recon_head + optimizer + scheduler + scaler) so
            # training can be resumed. Fine-tuning still only consumes the "model" key,
            # which is backward compatible.
            tmp_path = best_model_path.with_suffix(best_model_path.suffix + ".tmp")
            torch.save(
                build_checkpoint(epoch, best_val_loss, epochs_without_improvement),
                tmp_path,
            )
            os.replace(tmp_path, best_model_path)
            print(f"  -> New best model saved (val_loss={val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nPre-training complete. Best val_loss={best_val_loss:.6f}")
    print(f"Pre-trained backbone saved to {best_model_path}")
    print("\nTo fine-tune:")
    print(
        f"  python train.py --model_name finetuned --pretrained_path {best_model_path}"
    )

    run.finish()


if __name__ == "__main__":
    main()
