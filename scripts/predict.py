"""Prediction pipeline for CellTypeAnnotator.

Simple forward pass -> softmax.
Same tissue-aware constraint as training.
"""

import logging
import os
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torchinfo import summary
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)  # MulticlassAccuracy used for domain only

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.dataset import create_dataloader
from deepcell_types.model import create_model
from deepcell_types.training.losses import FocalLoss
from deepcell_types.training.utils import (
    BatchData, LossesAndMetrics, MPMetricsTracker, PredLogger,
    log_epoch_metrics, log_confusion_matrix, seed_everything,
    get_tissue_ct_exclude, build_label_remap,
)


DATA_DIR = Path(os.environ.get("DATA_DIR", ""))


@click.command()
@click.option("--model_name", type=str, default="deepcell-types")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--enable_wandb", type=bool, default=False)
@click.option("--zarr_dir", type=str, default=str(DATA_DIR))
@click.option("--skip_datasets", type=str, multiple=True, default=[])
@click.option("--keep_datasets", type=str, multiple=True, default=[])
@click.option("--exclude_ct_tissue", type=bool, default=False,
              help="Apply per-tissue cell-type exclusion at inference. Default False to match "
                   "canonical `--no_ct_exclude` training recipe; set True only if the checkpoint "
                   "was trained without `--no_ct_exclude`.")
@click.option("--batch_size", type=int, default=256)
@click.option("--num_workers", type=int, default=16)
@click.option("--svd_embeddings_path", type=str, default=None)
@click.option("--model_path", type=str, default=None, help="Explicit path to model weights")
@click.option("--resnet_channels", type=int, default=48, help="PerChannelResNet base channels (canonical: 48)")
@click.option("--mean_intensity_mode", type=click.Choice(["auto", "none", "cls_residual", "per_channel", "both"]), default="auto",
              help="Mean-intensity side-input mode (canonical: cls_residual). 'auto' detects from ckpt keys.")
@click.option("--split_file", type=str, default=None, help="FOV split JSON; evaluates val set only")
@click.option("--min_channels", type=int, default=0, help="Min model-visible marker channels per dataset (default 0 = no filter)")
@click.option("--spatial_pool_size", type=int, default=1, help="Spatial pooling grid size (must match training)")
@click.option("--apply_tissue_mask", is_flag=True, help="Mask tissue-inappropriate cell type logits before softmax (post-hoc fix for models trained with --no_ct_exclude)")
@click.option("--strict_tissue_mask", is_flag=True, help="Use training-split-based tissue mapping (stricter); requires --split_file and implies --apply_tissue_mask")
@click.option(
    "--learn_mp_thresholds",
    is_flag=True,
    help=(
        "Learn per-marker MP thresholds from the train set before evaluating val. "
        "Thresholds are learned from the training split that produced this checkpoint; "
        "for a fair cross-model comparison, use fixed thresholds via --mp_threshold_file instead."
    ),
)
@click.option("--mp_threshold_file", type=str, default=None, help="Path to pre-computed per-marker MP thresholds JSON")
@click.option(
    "--save_attention",
    is_flag=True,
    default=False,
    help="Save CLS→channel attention artifacts as output/{model_name}_mp_artifacts.npz (~390 MB per run). Off by default.",
)
@click.option("--seed", type=int, default=42, help="Random seed for inference reproducibility (matters when comparing predictions across seeds).")
@click.option(
    "--ct_abstention_k",
    type=float,
    default=0.2,
    help=(
        "Per-(tissue, modality) IQR-fence abstention on max-softmax confidence. "
        "Default k=0.2 — the published headline setting, chosen to maximise "
        "macro_F1 separation against the strongest baseline (XGBoost-tuned) "
        "while keeping a sizeable cohort of confident cells. "
        "Cells whose max-softmax falls below Q1 - k*IQR within their "
        "(tissue, modality) group are flagged as abstained (predicted_ct = -1, "
        "original kept in predicted_ct_raw). Set k <= 0 or pass 'none' to "
        "disable. k=1.5 is the canonical Tukey fence (~no-op). See "
        "docs/reports/ct_iqr_abstention_test.md."
    ),
)
def main(
    model_name, device_num, enable_wandb, zarr_dir, skip_datasets, keep_datasets,
    exclude_ct_tissue, batch_size, num_workers, svd_embeddings_path,
    model_path, resnet_channels, mean_intensity_mode, split_file, min_channels, spatial_pool_size,
    apply_tissue_mask, strict_tissue_mask,
    learn_mp_thresholds, mp_threshold_file,
    save_attention, seed, ct_abstention_k,
):
    seed_everything(seed)

    import wandb
    if enable_wandb:
        wandb.login()
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "deepcell-types"),
        dir="wandb_tmp",
        job_type="predict",
        mode="online" if enable_wandb else "disabled",
        name=model_name + "_predict",
    )

    device = torch.device(device_num)
    dct_config = TissueNetConfig(zarr_dir)
    d_model = 256

    # Load marker embeddings
    marker_embeddings = dct_config.load_marker_embeddings_array(svd_path=svd_embeddings_path)

    use_cuda = device.type == "cuda"

    # Load test data (val split if split_file provided, else all data).
    # When --learn_mp_thresholds is on, use the FOV-grouped train sampler so
    # the one-pass scan over the training split preserves per-worker zarr
    # cache locality. `shuffle=True` over a multi-thousand-FOV archive forces
    # each worker to cold-load ~1 GB of zarr per cell, which under spawn
    # workers manifests as the historical deadlock (issue #79).
    if split_file is not None:
        train_loader, test_loader, metadata = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=list(skip_datasets) if skip_datasets else None,
            keep_datasets=list(keep_datasets) if keep_datasets else None,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=num_workers,
            split_file=split_file,
            min_channels=min_channels,
            use_weighted_sampler=False,
            fov_grouped_train=learn_mp_thresholds,
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            pin_memory=use_cuda,
        )
        if not learn_mp_thresholds:
            del train_loader  # only need val
    else:
        _, test_loader, metadata = create_dataloader(
            zarr_dir=zarr_dir,
            dct_config=dct_config,
            skip_datasets=list(skip_datasets) if skip_datasets else None,
            keep_datasets=list(keep_datasets) if keep_datasets else None,
            batch_size=batch_size,
            num_dropout_channels=0,
            num_workers=num_workers,
            only_test=True,
            min_channels=min_channels,
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            pin_memory=use_cuda,
        )

    # Load weights (supports both old plain state_dict and new bundled checkpoint)
    if model_path is None:
        model_path = f"models/model_{model_name}_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Auto-detect tumor head from checkpoint
    has_tumor_head = False
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        has_tumor_head = any(k.startswith("tumor_head.") for k in checkpoint["model"])
    elif isinstance(checkpoint, dict):
        has_tumor_head = any(k.startswith("tumor_head.") for k in checkpoint)

    # Auto-detect mean_intensity_mode from ckpt keys when requested
    state_for_detect = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else (checkpoint if isinstance(checkpoint, dict) else {})
    if mean_intensity_mode == "auto":
        has_cls = any(k.startswith("intensity_cls_branch.") for k in state_for_detect)
        has_pch = any(k.startswith("intensity_per_channel_proj.") for k in state_for_detect)
        if has_cls and has_pch:
            mean_intensity_mode = "both"
        elif has_cls:
            mean_intensity_mode = "cls_residual"
        elif has_pch:
            mean_intensity_mode = "per_channel"
        else:
            mean_intensity_mode = "none"
        print(f"Auto-detected mean_intensity_mode = {mean_intensity_mode}")

    # Build model
    model = create_model(dct_config, marker_embeddings, d_model=d_model,
                         resnet_base_channels=resnet_channels,
                         tumor_head=has_tumor_head, spatial_pool_size=spatial_pool_size,
                         mean_intensity_mode=mean_intensity_mode)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Legacy format: plain state_dict — detect old Linear MP head
        if "marker_pos_head.weight" in checkpoint and "marker_pos_head.film_scale.weight" not in checkpoint:
            model = create_model(dct_config, marker_embeddings, d_model=d_model,
                                resnet_base_channels=resnet_channels,
                                use_conditioned_mp_head=False,
                                mean_intensity_mode=mean_intensity_mode).to(device)
        model.load_state_dict(checkpoint)
        print("Legacy checkpoint")

    model.to(device)
    summary(model, col_names=["trainable"])

    # Learn or load per-marker MP thresholds
    mp_thresholds = None
    idx2marker = {v: k for k, v in dct_config.marker2idx.items()}

    if learn_mp_thresholds:
        if split_file is None:
            raise click.UsageError("--learn_mp_thresholds requires --split_file")
        # Only ~5% of cells have MP labels, so 500K samples is enough for stable thresholds
        max_threshold_batches = 500_000 // batch_size
        total_batches = len(train_loader)
        n_batches = min(max_threshold_batches, total_batches)
        print(f"Learning per-marker MP thresholds from train set ({n_batches}/{total_batches} batches)...")
        train_mp_tracker = MPMetricsTracker()
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader, desc="Learning MP thresholds (train)", total=n_batches)):
                if i >= n_batches:
                    break
                batch_data = BatchData(*batch).to(device)
                _, _, marker_pos_logits, _, _, _ = model(
                    batch_data.sample, batch_data.spatial_context,
                    batch_data.ch_idx, batch_data.mask, None,
                    domain_idx=batch_data.domain_idx,
                )
                valid_mp_channels = ~batch_data.mask & batch_data.marker_positivity_mask
                if valid_mp_channels.any():
                    mp_pred = torch.sigmoid(marker_pos_logits[valid_mp_channels])
                    mp_target = batch_data.marker_positivity[valid_mp_channels]
                    mp_ch_indices = batch_data.ch_idx[valid_mp_channels]
                    train_mp_tracker.update(mp_pred, mp_target, mp_ch_indices)

        mp_thresholds = train_mp_tracker.find_optimal_thresholds()
        named_thresholds = {idx2marker.get(k, str(k)): v for k, v in mp_thresholds.items()}

        import json
        threshold_path = Path(f"output/{model_name}_mp_thresholds.json")
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(threshold_path, 'w') as f:
            json.dump(named_thresholds, f, indent=2, sort_keys=True)

        t_values = list(mp_thresholds.values())
        print(f"Learned thresholds for {len(mp_thresholds)} markers, saved to {threshold_path}")
        print(f"  Range: [{min(t_values):.3f}, {max(t_values):.3f}], mean={np.mean(t_values):.3f}")

        # Report train set metrics at learned vs fixed thresholds
        train_learned = train_mp_tracker.compute()
        train_fixed = train_mp_tracker.compute_at_fixed_threshold(0.5)
        print(f"  Train MP macro F1: {train_fixed['mp_macro_f1']:.4f} (fixed 0.5) → {train_learned['mp_macro_f1']:.4f} (learned)")
        del train_loader, train_mp_tracker

    elif mp_threshold_file is not None:
        import json
        with open(mp_threshold_file) as f:
            named_thresholds = json.load(f)
        mp_thresholds = {}
        for name, t in named_thresholds.items():
            if name in dct_config.marker2idx:
                mp_thresholds[dct_config.marker2idx[name]] = t
        print(f"Loaded {len(mp_thresholds)} per-marker thresholds from {mp_threshold_file}")

    # Metrics (with hierarchical eval to match train.py)
    label_remap = build_label_remap(dct_config.ct2idx).to(device)
    compact_ct2idx = {name: label_remap[idx].item() for name, idx in dct_config.ct2idx.items()}
    losses_metrics = LossesAndMetrics(
        ct_loss_fn=FocalLoss(gamma=2.0),
        domain_loss_fn=torch.nn.CrossEntropyLoss(),
        marker_pos_loss_fn=None,
        acc_domain_metric=MulticlassAccuracy(num_classes=dct_config.NUM_DOMAINS).to(device),
        conf_mat_ct_metric=MulticlassConfusionMatrix(
            num_classes=dct_config.NUM_CELLTYPES
        ).to(device),
        mp_metrics=MPMetricsTracker(thresholds=mp_thresholds),
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    # Precompute tissue mask: dataset_name -> boolean mask of allowed ct indices
    # Used by --apply_tissue_mask to mask out tissue-inappropriate logits post-hoc
    # --strict_tissue_mask implies --apply_tissue_mask with training-split-based mapping
    if strict_tissue_mask:
        if split_file is None:
            raise click.UsageError("--strict_tissue_mask requires --split_file")
        apply_tissue_mask = True  # strict implies apply

    tissue_mask_cache = {}  # dataset_name -> (n_celltypes,) bool tensor (True = allowed)
    if apply_tissue_mask:
        if strict_tissue_mask:
            tcm = {
                tissue: set(types)
                for tissue, types in dct_config.build_tissue_mapping_from_split(split_file).items()
            }
            print(f"Tissue mask: strict mode (training-split-based, {len(tcm)} tissues)")
        else:
            tcm = {
                tissue: set(types)
                for tissue, types in dct_config.tissue_celltype_mapping.items()
            }
            print(f"Tissue mask: archive-based ({len(tcm)} tissues)")
        ct2idx = dct_config.ct2idx
        n_ct = dct_config.NUM_CELLTYPES
        for ds_name in (metadata.get("active_datasets", []) if metadata else []):
            tissue = dct_config.get_tissue_for_dataset(ds_name)
            if tissue is None or tissue not in tcm or not tcm[tissue]:
                # No tissue info or empty allowed list -> allow all types
                tissue_mask_cache[ds_name] = torch.ones(n_ct, dtype=torch.bool, device=device)
            else:
                allowed = torch.zeros(n_ct, dtype=torch.bool, device=device)
                for ct in tcm[tissue]:
                    if ct in ct2idx:
                        allowed[ct2idx[ct]] = True
                tissue_mask_cache[ds_name] = allowed
        print(f"Tissue mask: precomputed for {len(tissue_mask_cache)} datasets "
              f"({sum(1 for v in tissue_mask_cache.values() if not v.all())} with restrictions)")

    # Predict
    model.eval()

    predlogger = PredLogger(dct_config.ct2idx)

    all_attn_mp = []
    all_tumor_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch_data = BatchData(*batch)

            ct_exclude = None
            if exclude_ct_tissue:
                ct_exclude = get_tissue_ct_exclude(batch_data, dct_config, label_remap)

            # Move to device
            batch_data = batch_data.to(device)

            # Remap labels to compact 0-indexed for metrics
            compact_ct_idx = label_remap[batch_data.ct_idx]

            # Forward — only request attention weights when --save_attention
            # is set. Asking for them unconditionally disables PyTorch's fast
            # SDPA kernel and forces a slower per-head materialised attention
            # path. Per the deep-review (perf): gating saves ~15% wall-clock
            # on a typical predict run because that kernel is the bottleneck.
            outputs = model(
                batch_data.sample,
                batch_data.spatial_context,
                batch_data.ch_idx,
                batch_data.mask,
                ct_exclude,
                return_attn_weights=save_attention,
                domain_idx=batch_data.domain_idx,
            )
            if save_attention:
                ct_logits, domain_logits, marker_pos_logits, cls_embedding, _, tumor_logit, cls_to_channels = outputs
            else:
                ct_logits, domain_logits, marker_pos_logits, cls_embedding, _, tumor_logit = outputs
                cls_to_channels = None

            # Apply post-hoc tissue mask: mask out tissue-inappropriate logits
            if apply_tissue_mask:
                for i, ds_name in enumerate(batch_data.dataset_name):
                    allowed = tissue_mask_cache.get(ds_name)
                    if allowed is not None and not allowed.all():
                        ct_logits[i, ~allowed] = float('-inf')

            probs = F.softmax(ct_logits, dim=-1)

            # Attention-derived MP: average CLS→channel attention across layers.
            # Only accumulate when --save_attention is set — otherwise the ~390 MB
            # concatenated array is never written and just burns RAM.
            if save_attention:
                attn_mp = cls_to_channels.mean(dim=0)  # (B, C_max)
                all_attn_mp.append(attn_mp.cpu().numpy())

            # Tumor probability
            if tumor_logit is not None:
                tumor_prob = torch.sigmoid(tumor_logit.squeeze(-1))  # (B,)
                all_tumor_probs.append(tumor_prob.cpu().numpy())

            # Log predictions (PredLogger stores original ct2idx labels for CSV)
            predlogger.log(
                labels=batch_data.ct_idx.detach().cpu().numpy(),
                probs=probs.cpu().detach().numpy(),
                cell_index=batch_data.cell_index.detach().cpu().numpy(),
                dataset_name=batch_data.dataset_name,
                fov_name=batch_data.fov_name,
            )

            # Update metrics (use compact labels)
            losses_metrics.acc_domain_metric(domain_logits, batch_data.domain_idx)
            losses_metrics.conf_mat_ct_metric(probs, compact_ct_idx)

            valid_mp_channels = ~batch_data.mask & batch_data.marker_positivity_mask
            if valid_mp_channels.any():
                mp_pred = torch.sigmoid(marker_pos_logits[valid_mp_channels])
                mp_target = batch_data.marker_positivity[valid_mp_channels]
                mp_ch_indices = batch_data.ch_idx[valid_mp_channels]
                losses_metrics.mp_metrics.update(mp_pred, mp_target, mp_ch_indices)

    # Compute and display metrics
    epoch_metrics = losses_metrics.compute()
    print(f"\nPrediction metrics: {epoch_metrics}")
    log_epoch_metrics(epoch_metrics, "test")

    log_confusion_matrix(
        losses_metrics.conf_mat_ct_metric, "test",
        sorted(dct_config.ct2idx, key=dct_config.ct2idx.get),
        metric_name="confusion_matrix_ct",
    )
    # Compare learned vs fixed thresholds if applicable
    if mp_thresholds is not None:
        fixed_metrics = losses_metrics.mp_metrics.compute_at_fixed_threshold(0.5)
        print(f"\nVal MP metrics comparison:")
        print(f"  Fixed 0.5:          macro_f1={fixed_metrics['mp_macro_f1']:.4f}  macro_prec={fixed_metrics['mp_macro_precision']:.4f}  macro_rec={fixed_metrics['mp_macro_recall']:.4f}")
        print(f"  Learned thresholds: macro_f1={epoch_metrics['mp_macro_f1']:.4f}  macro_prec={epoch_metrics['mp_macro_precision']:.4f}  macro_rec={epoch_metrics['mp_macro_recall']:.4f}")

    # Per-marker MP breakdown
    per_marker = losses_metrics.mp_metrics.compute_per_marker(idx2marker)
    if per_marker:
        import pandas as pd
        mp_df = pd.DataFrame(per_marker).T
        mp_df = mp_df.sort_values("f1", ascending=False)
        print(f"\nPer-marker MP metrics ({len(mp_df)} markers):")
        print(mp_df.to_string())
        mp_csv_path = Path(f"output/{model_name}_mp_per_marker.csv")
        mp_df.to_csv(mp_csv_path, index_label="marker")
        print(f"Per-marker MP metrics saved to {mp_csv_path}")

    output_path = Path(f"output/{model_name}_prediction.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predlogger.save(output_path)

    # Add tumor_probability column if tumor head was active
    if all_tumor_probs:
        import pandas as pd
        df = pd.read_csv(output_path)
        df["tumor_probability"] = np.concatenate(all_tumor_probs)
        df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

    # ---------------- CT abstention (post-hoc) ----------------
    # On by default with k=0.2 (published headline operating point);
    # set --ct_abstention_k 0 or a negative value to disable. Reads back the
    # saved CSV, derives per-cell predicted_ct + max_softmax from the per-class
    # probability columns, joins (tissue, modality) from the zarr archive,
    # applies an IQR-fence abstention per (tissue, modality) group, and
    # prints the kept-cell trade summary.
    if ct_abstention_k is not None and ct_abstention_k > 0:
        import pandas as pd
        import zarr as _zarr
        from deepcell_types.training.abstention import (
            apply_abstention,
            hierarchical_correct,
            macro_weighted_accuracy,
        )

        df = pd.read_csv(output_path)
        class_cols = sorted(dct_config.ct2idx, key=dct_config.ct2idx.get)
        # Per-class probability columns are written by PredLogger in this same order.
        probs_arr = df[class_cols].to_numpy(dtype=np.float32)
        pred_idx = probs_arr.argmax(axis=1)
        max_p = probs_arr[np.arange(probs_arr.shape[0]), pred_idx]
        df["predicted_ct"] = [class_cols[i] for i in pred_idx]
        df["_max_softmax"] = max_p

        # Tissue/modality come from the zarr archive's per-dataset attrs.
        root = _zarr.open(zarr_dir, mode="r")
        meta_rows = []
        for ds_key in root.group_keys():
            a = dict(root[ds_key].attrs)
            meta_rows.append({
                "dataset_name": ds_key,
                "tissue": a.get("tissue") or a.get("organ") or "unknown",
                "modality": a.get("modality") or "unknown",
            })
        meta_df = pd.DataFrame(meta_rows)
        df = df.merge(meta_df, on="dataset_name", how="left")
        df["tissue"] = df["tissue"].fillna("unknown")
        df["modality"] = df["modality"].fillna("unknown")

        # Baseline (pre-abstention) hierarchical accuracy on the full frame.
        true_labels = df["cell_type_actual"].to_numpy()
        pred_labels_pre = df["predicted_ct"].to_numpy()
        correct_pre = hierarchical_correct(true_labels, pred_labels_pre, CELL_TYPE_HIERARCHY)
        macro_pre, weighted_pre = macro_weighted_accuracy(true_labels, correct_pre, class_cols)

        # Apply abstention. We use the integer -1 as the sentinel per the
        # documented contract; pandas will keep the column dtype=object since
        # the non-abstained rows hold class-name strings.
        df = apply_abstention(
            df,
            k=float(ct_abstention_k),
            group_cols=("tissue", "modality"),
            max_softmax_col="_max_softmax",
            pred_col="predicted_ct",
            sentinel=-1,
        )

        # Kept-cell hierarchical accuracy (skip rows where we abstained).
        kept = ~df["abstained"].to_numpy()
        n_total = int(len(df))
        n_kept = int(kept.sum())
        n_abstained = n_total - n_kept
        coverage = n_kept / n_total if n_total else 0.0
        if n_kept > 0:
            correct_kept = correct_pre[kept]
            macro_post, weighted_post = macro_weighted_accuracy(
                true_labels[kept], correct_kept, class_cols
            )
        else:
            macro_post, weighted_post = 0.0, 0.0

        # Remove the internal helper column before persisting; predicted_ct is
        # already sentinel=-1 for abstained rows, original kept in predicted_ct_raw.
        df = df.drop(columns=["_max_softmax"])
        df.to_csv(output_path, index=False)

        print(f"\nCT abstention enabled (k={ct_abstention_k:.1f})")
        print(f"Coverage: {coverage*100:.2f}% ({n_abstained:,} abstained / {n_total:,} total)")
        delta_pp = (macro_post - macro_pre) * 100
        print(
            f"Macro accuracy on kept cells: {macro_post*100:.2f}% "
            f"(vs {macro_pre*100:.2f}% with no abstention; {delta_pp:+.2f}pp)"
        )
        print(
            f"Weighted accuracy on kept cells: {weighted_post*100:.2f}% "
            f"(vs {weighted_pre*100:.2f}%; {(weighted_post-weighted_pre)*100:+.2f}pp)"
        )

    # Save attention-derived MP as analysis artifact (gated by --save_attention;
    # the concatenated numpy array is ~390 MB per run on the v7 val split)
    if save_attention:
        mp_artifacts_path = Path(f"output/{model_name}_mp_artifacts.npz")
        np.savez(
            mp_artifacts_path,
            attn_mp=np.concatenate(all_attn_mp, axis=0),
        )
        print(f"MP artifacts saved to {mp_artifacts_path}")

    run.finish()


if __name__ == "__main__":
    main()
