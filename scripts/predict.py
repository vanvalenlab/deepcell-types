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
from deepcell_types.predict import (
    validate_checkpoint_vocabulary,
    _infer_ct_head_params,
)
from deepcell_types.training.losses import FocalLoss
from deepcell_types.training.utils import (
    BatchData,
    LossesAndMetrics,
    MPMetricsTracker,
    PredLogger,
    log_epoch_metrics,
    log_confusion_matrix,
    seed_everything,
    build_label_remap,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _resolve_marker_embeddings(dct_config, state_dict, svd_path):
    """Return the marker-embedding array passed to ``create_model``.

    When ``svd_path`` is given, load the SVD-reduced embeddings aligned to the
    config's ``marker2idx``. When it is ``None``, return a correctly-shaped
    zeros placeholder: the model's ``marker_embedder`` weights are restored from
    the checkpoint by ``load_state_dict`` downstream, so the placeholder's
    *values* are never used — only its shape must match. This lets the canonical
    evaluation run without regenerating the OpenAI-derived ``svd_512.npz``
    (whose contents the checkpoint overwrites anyway).
    """
    if svd_path is not None:
        return dct_config.load_marker_embeddings_array(svd_path=svd_path)
    try:
        embed_dim = int(state_dict["marker_embedder.embed_layer.weight"].shape[1])
    except KeyError as e:
        raise ValueError(
            f"Checkpoint is missing {e}; cannot infer the marker-embedding "
            "dimension for the zeros placeholder. Pass --svd_embeddings_path "
            "explicitly."
        ) from e
    return np.zeros((dct_config.NUM_MARKERS, embed_dim), dtype=np.float32)


@click.command()
@click.option("--model_name", type=str, default="deepcell-types")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--zarr_dir", type=str, default=str(DATA_DIR))
@click.option("--skip_datasets", type=str, multiple=True, default=[])
@click.option("--keep_datasets", type=str, multiple=True, default=[])
@click.option("--batch_size", type=int, default=256)
@click.option("--num_workers", type=int, default=16)
@click.option("--svd_embeddings_path", type=str, default=None)
@click.option(
    "--model_path", type=str, default=None, help="Explicit path to model weights"
)
@click.option(
    "--resnet_channels",
    type=int,
    default=48,
    help="PerChannelResNet base channels (canonical: 48)",
)
@click.option(
    "--split_file",
    type=str,
    default=None,
    help="FOV split JSON; evaluates val set only",
)
@click.option(
    "--spatial_pool_size",
    type=int,
    default=1,
    help="Spatial pooling grid size (must match training)",
)
@click.option(
    "--learn_mp_thresholds",
    is_flag=True,
    help=(
        "Learn per-marker MP thresholds from the train set before evaluating val. "
        "Thresholds are learned from the training split that produced this checkpoint; "
        "for a fair cross-model comparison, use fixed thresholds via --mp_threshold_file instead."
    ),
)
@click.option(
    "--mp_threshold_file",
    type=str,
    default=None,
    help="Path to pre-computed per-marker MP thresholds JSON",
)
@click.option(
    "--save_attention",
    is_flag=True,
    default=False,
    help="Save CLS→channel attention artifacts as output/{model_name}_mp_artifacts.npz (~390 MB per run). Off by default.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for inference reproducibility (matters when comparing predictions across seeds).",
)
@click.option(
    "--ct_abstention_k",
    type=float,
    default=0.0,
    help=(
        "Per-FOV IQR-fence abstention on max-softmax confidence. "
        "Default is 0 (disabled) — the current paper headline (Fig 3c) uses no "
        "abstention, full coverage. k=0.2 was an earlier paper draft's "
        "operating point; IQR-fence abstention was removed from the paper "
        "(see analysis/_score_csv.py in the research workspace) and remains "
        "available here only as a historical ablation, opt-in via this flag. "
        "Cells whose max-softmax falls below Q1 - k*IQR within their "
        "(dataset_name, fov_name) group are flagged as abstained "
        "(predicted_ct = 'Unknown', original kept in predicted_ct_raw). "
        "Set k <= 0 to disable. k=1.5 is the canonical Tukey fence (~no-op)."
    ),
)
def main(
    model_name,
    device_num,
    zarr_dir,
    skip_datasets,
    keep_datasets,
    batch_size,
    num_workers,
    svd_embeddings_path,
    model_path,
    resnet_channels,
    split_file,
    spatial_pool_size,
    learn_mp_thresholds,
    mp_threshold_file,
    save_attention,
    seed,
    ct_abstention_k,
):
    seed_everything(seed)

    device = torch.device(device_num)
    dct_config = TissueNetConfig(zarr_dir)
    d_model = 256

    # Load the checkpoint up front: the marker-embedding placeholder is sized
    # from it when no SVD embeddings file is supplied (see below). Supports both
    # the old plain state_dict and the new bundled checkpoint format.
    if model_path is None:
        model_path = f"models/model_{model_name}_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = _checkpoint_state_dict(checkpoint)

    # Marker embeddings. With --svd_embeddings_path, load the SVD-reduced array;
    # without it, build a zeros placeholder of the right shape — load_state_dict
    # below restores the real marker_embedder weights from the checkpoint, so the
    # placeholder values are overwritten and the eval needs no svd_512.npz.
    marker_embeddings = _resolve_marker_embeddings(
        dct_config, state_dict, svd_embeddings_path
    )

    use_cuda = device.type == "cuda"

    # Load test data (val split if split_file provided, else all data).
    # When --learn_mp_thresholds is on, use the FOV-grouped train sampler so
    # the one-pass scan over the training split preserves per-worker zarr
    # cache locality. `shuffle=True` over a multi-thousand-FOV archive forces
    # each worker to cold-load ~1 GB of zarr per cell, which under spawn
    # workers manifests as the historical deadlock.
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
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            pin_memory=use_cuda,
        )

    # Self-describing inference: when the checkpoint bundles a "config" dict
    # (loaded above), read the model-construction params from it so a future
    # retrain with different settings (e.g. compat_marker0_zero=False, a
    # different resnet_base_channels, or the legacy Linear MP head) reconstructs
    # the right architecture. compat_marker0_zero in particular is a pure-Python
    # numerics flag — NOT a state_dict tensor — so strict load can't catch a
    # mismatch; it would silently mis-infer. CLI flags / canonical defaults are
    # used as the fallback when a key (or the whole config) is absent, which
    # keeps the current released checkpoint loading unchanged.
    try:
        ct_head_params = _infer_ct_head_params(state_dict)
    except KeyError as e:
        raise ValueError(
            f"Checkpoint is missing the expected key {e}; this does not look "
            "like a deepcell-types CellTypeAnnotator checkpoint."
        ) from e

    # Guard the cell-type / marker ORDERING before the strict load. A strict
    # load_state_dict only catches shape (count) mismatches; a permuted ct2idx
    # of the right size would load cleanly and silently mislabel every cell in
    # the eval CSV. Same check the Python API runs.
    validate_checkpoint_vocabulary(checkpoint, dct_config.ct2idx, dct_config.marker2idx)

    # Build model
    model = create_model(
        dct_config,
        marker_embeddings,
        d_model=d_model,
        resnet_base_channels=ckpt_config.get("resnet_channels", resnet_channels),
        spatial_pool_size=ckpt_config.get("spatial_pool_size", spatial_pool_size),
        n_heads=ckpt_config.get("n_heads", 8),
        use_conditioned_mp_head=ckpt_config.get("use_conditioned_mp_head", True),
        compat_marker0_zero=ckpt_config.get("compat_marker0_zero", True),
        ct_head_width=ct_head_params["ct_head_width"],
        ct_head_depth=ct_head_params["ct_head_depth"],
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Legacy format: plain state_dict — detect old Linear MP head
        if (
            "marker_pos_head.weight" in checkpoint
            and "marker_pos_head.film_scale.weight" not in checkpoint
        ):
            model = create_model(
                dct_config,
                marker_embeddings,
                d_model=d_model,
                resnet_base_channels=resnet_channels,
                use_conditioned_mp_head=False,
                ct_head_width=ct_head_params["ct_head_width"],
                ct_head_depth=ct_head_params["ct_head_depth"],
            ).to(device)
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
        print(
            f"Learning per-marker MP thresholds from train set ({n_batches}/{total_batches} batches)..."
        )
        train_mp_tracker = MPMetricsTracker()
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(
                    train_loader, desc="Learning MP thresholds (train)", total=n_batches
                )
            ):
                if i >= n_batches:
                    break
                batch_data = BatchData(*batch).to(device)
                marker_pos_logits = model(
                    batch_data.sample,
                    batch_data.spatial_context,
                    batch_data.ch_idx,
                    batch_data.mask,
                    domain_idx=batch_data.domain_idx,
                ).marker_pos_logits
                valid_mp_channels = ~batch_data.mask & batch_data.marker_positivity_mask
                if valid_mp_channels.any():
                    mp_pred = torch.sigmoid(marker_pos_logits[valid_mp_channels])
                    mp_target = batch_data.marker_positivity[valid_mp_channels]
                    mp_ch_indices = batch_data.ch_idx[valid_mp_channels]
                    train_mp_tracker.update(mp_pred, mp_target, mp_ch_indices)

        mp_thresholds = train_mp_tracker.find_optimal_thresholds()
        named_thresholds = {
            idx2marker.get(k, str(k)): v for k, v in mp_thresholds.items()
        }

        import json

        threshold_path = Path(f"output/{model_name}_mp_thresholds.json")
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(threshold_path, "w") as f:
            json.dump(named_thresholds, f, indent=2, sort_keys=True)

        t_values = list(mp_thresholds.values())
        print(
            f"Learned thresholds for {len(mp_thresholds)} markers, saved to {threshold_path}"
        )
        print(
            f"  Range: [{min(t_values):.3f}, {max(t_values):.3f}], mean={np.mean(t_values):.3f}"
        )

        # Report train set metrics at learned vs fixed thresholds
        train_learned = train_mp_tracker.compute()
        train_fixed = train_mp_tracker.compute_at_fixed_threshold(0.5)
        print(
            f"  Train MP macro F1: {train_fixed['mp_macro_f1']:.4f} (fixed 0.5) → {train_learned['mp_macro_f1']:.4f} (learned)"
        )
        del train_loader, train_mp_tracker

    elif mp_threshold_file is not None:
        import json

        with open(mp_threshold_file) as f:
            named_thresholds = json.load(f)
        mp_thresholds = {}
        for name, t in named_thresholds.items():
            if name in dct_config.marker2idx:
                mp_thresholds[dct_config.marker2idx[name]] = t
        print(
            f"Loaded {len(mp_thresholds)} per-marker thresholds from {mp_threshold_file}"
        )

    # Metrics (with hierarchical eval to match train.py)
    label_remap = build_label_remap(dct_config.ct2idx).to(device)
    compact_ct2idx = {
        name: label_remap[idx].item() for name, idx in dct_config.ct2idx.items()
    }
    losses_metrics = LossesAndMetrics(
        ct_loss_fn=FocalLoss(gamma=2.0),
        domain_loss_fn=torch.nn.CrossEntropyLoss(),
        marker_pos_loss_fn=None,
        acc_domain_metric=MulticlassAccuracy(num_classes=dct_config.NUM_DOMAINS).to(
            device
        ),
        conf_mat_ct_metric=MulticlassConfusionMatrix(
            num_classes=dct_config.NUM_CELLTYPES
        ).to(device),
        mp_metrics=MPMetricsTracker(thresholds=mp_thresholds),
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    # Predict
    model.eval()

    predlogger = PredLogger(dct_config.ct2idx)

    all_attn_mp = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch_data = BatchData(*batch)

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
                return_attn_weights=save_attention,
                domain_idx=batch_data.domain_idx,
            )
            ct_logits = outputs.ct_logits
            domain_logits = outputs.domain_logits
            marker_pos_logits = outputs.marker_pos_logits
            cls_to_channels = outputs.cls_to_channels

            probs = F.softmax(ct_logits, dim=-1)

            # Attention-derived MP: average CLS→channel attention across layers.
            # Only accumulate when --save_attention is set — otherwise the ~390 MB
            # concatenated array is never written and just burns RAM.
            if save_attention:
                attn_mp = cls_to_channels.mean(dim=0)  # (B, C_max)
                all_attn_mp.append(attn_mp.cpu().numpy())

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
        losses_metrics.conf_mat_ct_metric,
        "test",
        sorted(dct_config.ct2idx, key=dct_config.ct2idx.get),
        metric_name="confusion_matrix_ct",
    )
    # Compare learned vs fixed thresholds if applicable
    if mp_thresholds is not None:
        fixed_metrics = losses_metrics.mp_metrics.compute_at_fixed_threshold(0.5)
        print("\nVal MP metrics comparison:")
        print(
            f"  Fixed 0.5:          macro_f1={fixed_metrics['mp_macro_f1']:.4f}  macro_prec={fixed_metrics['mp_macro_precision']:.4f}  macro_rec={fixed_metrics['mp_macro_recall']:.4f}"
        )
        print(
            f"  Learned thresholds: macro_f1={epoch_metrics['mp_macro_f1']:.4f}  macro_prec={epoch_metrics['mp_macro_precision']:.4f}  macro_rec={epoch_metrics['mp_macro_recall']:.4f}"
        )

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

    # Assemble predictions in memory so abstention is baked into a single write
    # (no write-raw -> read-back -> overwrite). When abstention is enabled the
    # IQR fence is computed per (dataset_name, fov_name) group, matching the
    # Python API's per-FOV semantics (deepcell_types.predict).
    df = predlogger.to_dataframe()

    # ---------------- CT abstention (immediate, per-FOV) ----------------
    # On by default with k=0.2 (published headline operating point); set
    # --ct_abstention_k 0 or a negative value to disable. When disabled, the
    # frame is written as-is (probability + metadata columns only, no
    # predicted_ct) — matching the historical disabled-path output.
    if ct_abstention_k is not None and ct_abstention_k > 0:
        from deepcell_types.abstention import apply_abstention
        from deepcell_types.training.metrics import hierarchical_macro_f1

        class_cols = sorted(dct_config.ct2idx, key=dct_config.ct2idx.get)
        probs_arr = df[class_cols].to_numpy(dtype=np.float32)
        pred_idx = probs_arr.argmax(axis=1)
        df["predicted_ct"] = [class_cols[i] for i in pred_idx]
        df["_max_softmax"] = probs_arr[np.arange(probs_arr.shape[0]), pred_idx]

        # Baseline (pre-abstention) hierarchical macro-F1 on the full frame.
        true_labels = df["cell_type_actual"].to_numpy()
        pred_labels_pre = df["predicted_ct"].to_numpy()
        macro_f1_pre = hierarchical_macro_f1(
            true_labels, pred_labels_pre, class_cols, CELL_TYPE_HIERARCHY
        )

        # Apply abstention per (dataset_name, fov_name). Sentinel "Unknown"
        # matches the Python API contract (predict.ABSTENTION_LABEL). Adds
        # `abstained` and `predicted_ct_raw`; sentinels predicted_ct in place.
        df = apply_abstention(
            df,
            k=float(ct_abstention_k),
            group_cols=("dataset_name", "fov_name"),
            max_softmax_col="_max_softmax",
            pred_col="predicted_ct",
            sentinel="Unknown",
        )

        # Kept-cell hierarchical macro-F1 (kept cells retain their original
        # prediction, so reuse pred_labels_pre).
        kept = ~df["abstained"].to_numpy()
        n_total = int(len(df))
        n_kept = int(kept.sum())
        n_abstained = n_total - n_kept
        coverage = n_kept / n_total if n_total else 0.0
        if n_kept > 0:
            macro_f1_post = hierarchical_macro_f1(
                true_labels[kept],
                pred_labels_pre[kept],
                class_cols,
                CELL_TYPE_HIERARCHY,
            )
        else:
            macro_f1_post = 0.0

        # Drop the internal helper column before persisting.
        df = df.drop(columns=["_max_softmax"])

    # Single atomic write — when abstention is on, the file already contains
    # the "Unknown" labels (predicted_ct) and the predicted_ct_raw/abstained
    # columns.
    PredLogger.write_csv_atomic(df, output_path)
    print(f"Predictions saved to {output_path}")

    if ct_abstention_k is not None and ct_abstention_k > 0:
        print(f"\nCT abstention enabled (k={ct_abstention_k:.1f})")
        print(
            f"Coverage: {coverage * 100:.2f}% "
            f"({n_abstained:,} abstained / {n_total:,} total)"
        )
        delta_pp = (macro_f1_post - macro_f1_pre) * 100
        print(
            f"Macro F1 on kept cells: {macro_f1_post * 100:.2f}% "
            f"(vs {macro_f1_pre * 100:.2f}% with no abstention; {delta_pp:+.2f}pp)"
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


if __name__ == "__main__":
    main()
