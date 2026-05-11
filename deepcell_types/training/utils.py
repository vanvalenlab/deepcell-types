import logging
import hashlib
import json
import os
import pickle
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from zipfile import BadZipFile

import torch

logger = logging.getLogger(__name__)


def _zarr_group_filesystem_path(group):
    """Return the local filesystem path for a zarr group, when available."""
    store_path = getattr(group, "store_path", None)
    store = getattr(store_path, "store", None)
    root = getattr(store, "root", None)
    if root is None:
        return None
    path = getattr(store_path, "path", "")
    return Path(root) / path if path else Path(root)


def _read_v3_1d_array(array_dir: Path):
    """Read simple one-dimensional zarr v3 arrays without zarr's alpha parser."""
    meta_path = array_dir / "zarr.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    n = int(meta["shape"][0])
    data_type = meta["data_type"]
    if isinstance(data_type, dict) and data_type.get("name") == "fixed_length_utf32":
        dtype = np.dtype(f"<U{int(data_type['configuration']['length_bytes']) // 4}")
    else:
        dtype = np.dtype(data_type)

    chunk_len = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    fill_value = meta.get("fill_value", 0)
    out = np.full((n,), fill_value, dtype=dtype)
    if n == 0:
        return out

    codecs = [codec.get("name") for codec in meta.get("codecs", [])]
    zstd = None
    if "zstd" in codecs:
        from numcodecs import Zstd

        zstd = Zstd(level=0)

    for chunk_idx, start in enumerate(range(0, n, chunk_len)):
        chunk_path = array_dir / "c" / str(chunk_idx)
        if not chunk_path.exists():
            continue
        data = chunk_path.read_bytes()
        if zstd is not None:
            data = zstd.decode(data)
        chunk = np.frombuffer(data, dtype=dtype, count=chunk_len)
        stop = min(start + chunk_len, n)
        out[start:stop] = chunk[: stop - start]
    return out


def _stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _file_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _atomic_pickle_dump(obj, path: Path, *, protocol: int = 4) -> None:
    """Write a pickle cache by replacing the final path atomically."""
    tmp_path = None
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        pickle.dump(obj, tmp, protocol=protocol)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _atomic_np_savez(path: Path, **arrays) -> None:
    """Write an npz cache by replacing the final path atomically."""
    tmp_path = None
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _feature_cache_metadata(
    zarr_dir: str,
    dct_config,
    min_channels: int,
    dataset_keys: list,
    split_file: str | None = None,
) -> dict:
    from deepcell_types.training.config import archive_array_fingerprint

    zarr_path = Path(zarr_dir).expanduser()
    try:
        zarr_path = zarr_path.resolve()
    except OSError:
        pass

    return {
        "cache_version": 5,
        "zarr_dir": str(zarr_path),
        "min_channels": min_channels,
        "dataset_keys_hash": _stable_hash(sorted(dataset_keys)),
        "marker2idx_hash": _stable_hash(dct_config.marker2idx),
        "ct2idx_hash": _stable_hash(dct_config.ct2idx),
        "split_file_hash": _file_hash(split_file),
        "archive_fingerprint": archive_array_fingerprint(zarr_path, dataset_keys),
    }


def _cache_metadata_mismatches(saved: dict | None, expected: dict) -> list[str]:
    if not saved:
        return ["missing metadata"]
    return [key for key, value in expected.items() if saved.get(key) != value]


def _format_examples(values, limit: int = 5) -> str:
    examples = sorted(values)[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(str(v) for v in examples) + suffix


@dataclass
class BatchData:
    """Standardized batch format for all training/inference scripts.

    Fields (factored representation):
        sample: (B, C_max, 1, H, W) - raw intensity * self_mask per channel
        spatial_context: (B, 3, H, W) - [self_mask, neighbor_mask, distance_transform]
        ch_idx: (B, C_max) - channel indices
        mask: (B, C_max) - padding mask (True = padding)
        ct_idx: (B,) - cell type indices
        domain_idx: (B,) - domain (modality) indices
        marker_positivity: (B, C_max) - marker positivity labels
        marker_positivity_mask: (B, C_max) - mask for "?" labels (True = valid, compute loss)
        cell_index: (B,) - cell index in FOV
        dataset_name: tuple of str - dataset names
        fov_name: tuple of str - FOV names
        tissue_idx: (B,) - tissue indices (index 0 = ``__null__``); defaults to
            zeros so older datasets that don't ship a tissue lookup still load.
    """

    sample: torch.Tensor
    spatial_context: torch.Tensor
    ch_idx: torch.Tensor
    mask: torch.Tensor
    ct_idx: torch.Tensor
    domain_idx: torch.Tensor
    marker_positivity: torch.Tensor
    marker_positivity_mask: torch.Tensor
    cell_index: torch.Tensor
    dataset_name: Any
    fov_name: Any
    tissue_idx: Optional[torch.Tensor] = None

    def to(self, device):
        """Move all tensor fields to device, pass through non-tensor fields."""
        return BatchData(
            sample=self.sample.to(device),
            spatial_context=self.spatial_context.to(device),
            ch_idx=self.ch_idx.to(device),
            mask=self.mask.to(device),
            ct_idx=self.ct_idx.to(device),
            domain_idx=self.domain_idx.to(device),
            marker_positivity=self.marker_positivity.to(device),
            marker_positivity_mask=self.marker_positivity_mask.to(device),
            cell_index=self.cell_index.to(device),
            dataset_name=self.dataset_name,
            fov_name=self.fov_name,
            tissue_idx=self.tissue_idx.to(device) if self.tissue_idx is not None else None,
        )


def adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx):
    """Adjust confusion matrix so child predictions count as correct for parent types.

    For each parent in hierarchy, moves parent->child counts to the diagonal.

    Args:
        conf_mat: (N, N) confusion matrix (rows=true, cols=predicted)
        hierarchy: dict mapping parent type names to lists of child type names
        ct2idx: dict mapping cell type names to indices used in conf_mat
            (must match the index space of conf_mat, e.g. compact 0-indexed)
    """
    adjusted = conf_mat.copy()
    for parent, children in hierarchy.items():
        if parent not in ct2idx:
            continue
        parent_idx = ct2idx[parent]
        for child in children:
            if child not in ct2idx:
                continue
            child_idx = ct2idx[child]
            adjusted[parent_idx, parent_idx] += adjusted[parent_idx, child_idx]
            adjusted[parent_idx, child_idx] = 0
    return adjusted


def summarize_mp_per_marker(per_marker_counts: dict) -> dict:
    """Single source of truth for marker-positivity macro/micro reduction.

    Same reduction shared by the main model (`MPMetricsTracker`) and the
    Nimbus baseline (`compute_marker_positivity_metrics`) so the two
    headline numbers are directly comparable.

    Args:
        per_marker_counts: dict mapping marker_id -> {"tp", "fp", "fn", "tn"}
            (ints). Keys can be channel indices or marker name strings; only
            the values matter.

    Returns:
        dict with keys mp_macro_f1, mp_micro_f1, mp_micro_precision,
        mp_micro_recall, mp_macro_precision, mp_macro_recall,
        mp_macro_accuracy, mp_num_markers, mp_num_markers_excluded_from_macro_f1.

    macro_f1 follows the convention: a marker is *excluded* (NaN) when
    n_positive_gt == 0 AND n_positive_pred == 0 (F1 is vacuous in that
    case and would otherwise drag the mean to 0).
    """
    if not per_marker_counts:
        return {
            "mp_macro_f1": 0.0,
            "mp_micro_f1": 0.0,
            "mp_micro_precision": 0.0,
            "mp_micro_recall": 0.0,
            "mp_macro_precision": 0.0,
            "mp_macro_recall": 0.0,
            "mp_macro_accuracy": 0.0,
            "mp_num_markers": 0,
            "mp_num_markers_excluded_from_macro_f1": 0,
        }

    f1s, precisions, recalls, accuracies = [], [], [], []
    total_tp = total_fp = total_fn = total_tn = 0
    excluded = 0

    for counts in per_marker_counts.values():
        tp = int(counts["tp"])
        fp = int(counts["fp"])
        fn = int(counts["fn"])
        tn = int(counts["tn"])

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        accuracies.append(acc)

        # Symmetric vacuous-marker exclusion across precision/recall/F1 — otherwise
        # the triangle identity macro_f1 ≈ 2pr/(p+r) breaks because prec/rec would
        # include 0.0 for vacuous markers while F1 used nanmean (PR #55 fix).
        n_pos_gt = tp + fn
        n_pos_pred = tp + fp
        if n_pos_gt == 0 and n_pos_pred == 0:
            f1s.append(np.nan)
            precisions.append(np.nan)
            recalls.append(np.nan)
            excluded += 1
        else:
            f1s.append(f1)
            precisions.append(prec)
            recalls.append(rec)

    denom_f1 = 2 * total_tp + total_fp + total_fn
    denom_prec = total_tp + total_fp
    denom_rec = total_tp + total_fn
    macro_f1 = (
        float(np.nanmean(f1s))
        if len(f1s) > 0 and not np.all(np.isnan(f1s))
        else 0.0
    )

    return {
        "mp_macro_f1": macro_f1,
        "mp_micro_f1": float(2 * total_tp / denom_f1) if denom_f1 > 0 else 0.0,
        "mp_micro_precision": float(total_tp / denom_prec) if denom_prec > 0 else 0.0,
        "mp_micro_recall": float(total_tp / denom_rec) if denom_rec > 0 else 0.0,
        "mp_macro_precision": (
            float(np.nanmean(precisions))
            if len(precisions) > 0 and not np.all(np.isnan(precisions))
            else 0.0
        ),
        "mp_macro_recall": (
            float(np.nanmean(recalls))
            if len(recalls) > 0 and not np.all(np.isnan(recalls))
            else 0.0
        ),
        "mp_macro_accuracy": float(np.mean(accuracies)),
        "mp_num_markers": len(per_marker_counts),
        "mp_num_markers_excluded_from_macro_f1": excluded,
    }


class MPMetricsTracker:
    """Per-marker marker positivity metrics with macro/micro averaging.

    Accumulates raw per-marker scores and targets. Metrics are computed lazily
    at compute() time using configured per-marker thresholds (default 0.5).
    Supports learning optimal thresholds from accumulated data.
    """

    def __init__(self, thresholds=None):
        """
        Args:
            thresholds: Optional dict {ch_idx: float} of per-marker thresholds.
                If None, uses 0.5 for all markers.
        """
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self._score_buffers = defaultdict(
            list
        )  # {ch_idx: [(preds_arr, targets_arr), ...]}

    def update(self, pred_scores, targets, ch_indices):
        """Update with a batch of predictions.

        Args:
            pred_scores: (N,) sigmoid probabilities
            targets: (N,) binary targets (0 or 1)
            ch_indices: (N,) channel indices (from marker2idx)
        """
        pred_scores = pred_scores.detach().cpu()
        targets = targets.detach().cpu().long()
        ch_indices = ch_indices.detach().cpu()

        for ch in ch_indices.unique():
            ch_val = ch.item()
            mask = ch_indices == ch
            self._score_buffers[ch_val].append(
                (pred_scores[mask].numpy(), targets[mask].numpy())
            )

    def _get_scores(self, ch_idx):
        preds = np.concatenate([s[0] for s in self._score_buffers[ch_idx]])
        targets = np.concatenate([s[1] for s in self._score_buffers[ch_idx]])
        return preds, targets

    def _get_threshold(self, ch_idx):
        if self.thresholds and ch_idx in self.thresholds:
            return self.thresholds[ch_idx]
        return 0.5

    @staticmethod
    def _marker_metrics(tp, fp, fn, tn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        return prec, rec, f1, acc

    def _compute_impl(self, fixed_threshold=None):
        """Core compute. If fixed_threshold is set, use it for all markers."""
        if not self._score_buffers:
            return {
                **summarize_mp_per_marker({}),
                "mp_macro_auroc": 0.0,
            }

        per_marker_counts = {}
        aurocs = []

        for ch_idx in self._score_buffers:
            preds, targets = self._get_scores(ch_idx)
            threshold = (
                fixed_threshold
                if fixed_threshold is not None
                else self._get_threshold(ch_idx)
            )
            pred_binary = (preds >= threshold).astype(int)

            per_marker_counts[ch_idx] = {
                "tp": int(((pred_binary == 1) & (targets == 1)).sum()),
                "fp": int(((pred_binary == 1) & (targets == 0)).sum()),
                "fn": int(((pred_binary == 0) & (targets == 1)).sum()),
                "tn": int(((pred_binary == 0) & (targets == 0)).sum()),
            }

            if len(np.unique(targets)) == 2:
                from sklearn.metrics import roc_auc_score

                aurocs.append(float(roc_auc_score(targets, preds)))

        # Reduce through the shared helper so MP eval is bit-exact across
        # main model and Nimbus baseline.
        result = summarize_mp_per_marker(per_marker_counts)

        if result["mp_num_markers_excluded_from_macro_f1"] > 0:
            logger.info(
                "MPMetricsTracker: excluded %d/%d markers from macro F1 "
                "(no positive GT and no positive predictions)",
                result["mp_num_markers_excluded_from_macro_f1"],
                len(self._score_buffers),
            )

        # AUROC requires continuous scores, so it lives outside the shared
        # count-based helper.
        result["mp_macro_auroc"] = float(np.mean(aurocs)) if aurocs else 0.0
        return result

    def compute(self):
        """Return macro/micro summary metrics using configured thresholds."""
        return self._compute_impl()

    def compute_at_fixed_threshold(self, threshold=0.5):
        """Return metrics at a fixed threshold for all markers (ignoring learned thresholds)."""
        return self._compute_impl(fixed_threshold=threshold)

    def compute_per_marker(self, idx2marker=None, fixed_threshold=None):
        """Return detailed per-marker metrics as a dict of dicts."""
        results = {}
        for ch_idx in sorted(self._score_buffers.keys()):
            preds, targets = self._get_scores(ch_idx)
            threshold = (
                fixed_threshold
                if fixed_threshold is not None
                else self._get_threshold(ch_idx)
            )
            pred_binary = (preds >= threshold).astype(int)

            tp = int(((pred_binary == 1) & (targets == 1)).sum())
            fp = int(((pred_binary == 1) & (targets == 0)).sum())
            fn = int(((pred_binary == 0) & (targets == 1)).sum())
            tn = int(((pred_binary == 0) & (targets == 0)).sum())

            prec, rec, f1, acc = self._marker_metrics(tp, fp, fn, tn)
            auroc = None
            if len(np.unique(targets)) == 2:
                from sklearn.metrics import roc_auc_score

                auroc = float(roc_auc_score(targets, preds))

            name = idx2marker.get(ch_idx, str(ch_idx)) if idx2marker else str(ch_idx)
            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auroc": auroc,
                "threshold": threshold,
                "n_samples": tp + fp + fn + tn,
            }
        return results

    def find_optimal_thresholds(self, n_thresholds=50):
        """Find per-marker thresholds that maximize F1 from accumulated scores.

        Markers with zero positive ground-truth labels in the calibration set
        cannot have F1 maximized — every sweep value gives F1=0. The fallback
        is the neutral 0.5 threshold, but that is uncalibrated, so this method
        emits a warning naming the affected channels (capped at 20 names to
        avoid log spam) and the count is logged.

        Returns:
            dict {ch_idx: optimal_threshold}
        """
        sweep = np.linspace(0.01, 0.99, n_thresholds)
        thresholds = {}
        uncalibrated: list[int] = []

        for ch_idx in self._score_buffers:
            preds, targets = self._get_scores(ch_idx)
            n_pos = int((targets == 1).sum())
            best_f1, best_t = 0.0, 0.5

            for t in sweep:
                pred_binary = (preds >= t).astype(int)
                tp = ((pred_binary == 1) & (targets == 1)).sum()
                denom = (
                    2 * tp
                    + ((pred_binary == 1) & (targets == 0)).sum()
                    + ((pred_binary == 0) & (targets == 1)).sum()
                )
                f1 = float(2 * tp / denom) if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)

            if n_pos == 0:
                uncalibrated.append(ch_idx)
            thresholds[ch_idx] = best_t

        if uncalibrated:
            logger.warning(
                "find_optimal_thresholds: %d/%d markers had zero positive "
                "GT labels in the calibration set; their thresholds default "
                "to 0.5 (uncalibrated). Affected ch_idx (first 20): %s",
                len(uncalibrated),
                len(self._score_buffers),
                uncalibrated[:20],
            )

        return thresholds


@dataclass
class LossesAndMetrics:
    ct_loss_fn: Any
    domain_loss_fn: Any
    marker_pos_loss_fn: Any
    acc_domain_metric: Any
    conf_mat_ct_metric: Any
    mp_metrics: Any  # MPMetricsTracker instance
    hierarchy: dict = field(default=None)
    ct2idx: dict = field(default=None)

    def __post_init__(self):
        # Warn if FocalLoss class_weights are active without a frequency floor.
        # WeightedRandomSampler (compute_sample_weights in dataset.py) floors
        # per-class counts at 1000 to prevent sqrt-inv-frequency from producing
        # runaway amplification (raw sqrt-inv-frequency would otherwise boost
        # the rarest classes ~19000x). If FocalLoss class_weights are derived from raw
        # sqrt-inv-frequency without the same floor and the sampler is also
        # on, rare classes get double-weighted.
        alpha = getattr(self.ct_loss_fn, "alpha", None)
        if alpha is not None and isinstance(alpha, torch.Tensor) and alpha.numel() > 0:
            # Heuristic: compare max/min; sqrt-inv-frequency with no floor
            # will produce ratios on the order of sqrt(N_majority / N_minority),
            # which can exceed 100x for rare classes in this dataset.
            alpha_f = alpha.detach().float().cpu()
            finite = alpha_f[torch.isfinite(alpha_f) & (alpha_f > 0)]
            if finite.numel() >= 2:
                ratio = float(finite.max() / finite.min())
            else:
                ratio = 1.0
            logger.warning(
                "LossesAndMetrics: FocalLoss class_weights are active "
                "(max/min=%.1fx across %d classes). If WeightedRandomSampler "
                "is also enabled, you are double-weighting rare classes "
                "(sampler floors counts at 1000 but FocalLoss does not). "
                "Use --no_class_weights when the "
                "sampler is on, or match the sampler's floor when computing "
                "class_weights.",
                ratio,
                int(alpha_f.numel()),
            )

    def reset_metrics(self):
        self.acc_domain_metric.reset()
        self.conf_mat_ct_metric.reset()
        self.mp_metrics.reset()

    def compute(self):
        # Derive macro/weighted accuracy + macro/weighted F1 from a single
        # confusion matrix so baseline and main-model numbers are directly
        # comparable. See _conf_mat_summary() for the formula.
        conf_mat = self.conf_mat_ct_metric.compute().cpu().numpy()
        if self.hierarchy and self.ct2idx:
            conf_mat = adjust_conf_mat_hierarchy(conf_mat, self.hierarchy, self.ct2idx)
        summary = _conf_mat_summary(conf_mat)
        mp = self.mp_metrics.compute()
        return {
            "ct_macro_accuracy": summary["macro_accuracy"],
            "ct_weighted_accuracy": summary["weighted_accuracy"],
            "ct_macro_f1": summary["macro_f1"],
            "ct_weighted_f1": summary["weighted_f1"],
            "domain_accuracy": self.acc_domain_metric.compute().item(),
            **mp,
        }


def build_label_remap(ct2idx):
    """Build lookup tensor to remap ct2idx values to contiguous 0-indexed labels.

    ct2idx values may not be 0-indexed (e.g. 1..N), but CrossEntropyLoss,
    FocalLoss, and torchmetrics require contiguous 0-indexed labels (0..N-1).

    Args:
        ct2idx: dict mapping cell type names to integer indices

    Returns:
        label_remap: torch.LongTensor of shape (max_value+1,) where
            label_remap[orig_idx] = compact_idx
    """
    sorted_ct_values = sorted(ct2idx.values())
    max_orig = max(sorted_ct_values) + 1
    label_remap = torch.zeros(max_orig, dtype=torch.long)
    for compact, orig in enumerate(sorted_ct_values):
        label_remap[orig] = compact
    return label_remap


class PredLogger:
    def __init__(self, ct2idx):
        self.ct2idx = ct2idx
        self.labels = []
        self.probs = []
        self.cell_index = []
        self.dataset_name = []
        self.fov_name = []

    def log(self, labels, probs, cell_index, dataset_name, fov_name):
        self.labels.append(labels)
        self.probs.append(probs)
        self.cell_index.append(cell_index)
        self.dataset_name.append(dataset_name)
        self.fov_name.append(fov_name)

    def save(self, path_name):
        columns = sorted(self.ct2idx, key=self.ct2idx.get)
        idx2ct = {v: k for k, v in self.ct2idx.items()}
        labels = np.concatenate(self.labels)
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        dataset_name = np.concatenate(self.dataset_name)
        fov_name = np.concatenate(self.fov_name)
        df = pd.DataFrame(probs, columns=columns)
        df["cell_type_actual"] = [idx2ct[label] for label in labels]
        df["cell_index"] = cell_index
        df["dataset_name"] = dataset_name
        df["fov_name"] = fov_name
        # Atomic write: a disk-full or SIGTERM mid-write would otherwise leave a
        # truncated CSV that pandas reads silently, producing wrong abstention
        # numbers in downstream analysis.
        final_path = Path(path_name)
        tmp_path: Optional[Path] = None
        with tempfile.NamedTemporaryFile(
            "w",
            dir=final_path.parent,
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            df.to_csv(tmp, index=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            tmp_path.replace(final_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


def log_epoch_metrics(epoch_metrics, prefix, wandb_run=None):
    """Log epoch-level metrics to wandb.

    Args:
        epoch_metrics: Dict of metric name -> value
        prefix: "train", "val", or "test"
        wandb_run: Optional wandb run object. If None, imports wandb and logs directly.
    """
    # Only network/IO failures from wandb should be swallowed. Logic errors
    # (AttributeError/KeyError/TypeError) must propagate — they indicate a
    # bug in the caller, not an expected runtime condition.
    try:
        import wandb
        import wandb.errors as _wandb_errors
    except ImportError as exc:
        logger.warning(
            "log_epoch_metrics: wandb import failed (prefix=%s): %s", prefix, exc
        )
        return

    for metric_name, metric_value in epoch_metrics.items():
        try:
            wandb.log({f"{prefix}/{metric_name}_epoch": metric_value})
        except (_wandb_errors.CommError, OSError) as exc:
            logger.warning(
                "log_epoch_metrics failed for prefix=%s metric=%s: %s",
                prefix,
                metric_name,
                exc,
            )


def log_confusion_matrix(
    metric, prefix, class_names, metric_name="confusion_matrix", tmp_dir="./tmp_images"
):
    """Log confusion matrix to wandb.

    Args:
        metric: torchmetrics confusion matrix metric
        prefix: "train", "val", or "test"
        class_names: List of class names for axis labels
        metric_name: Name for the wandb log entry
        tmp_dir: Directory for temporary image files
    """
    # Compute outside try/except so torchmetrics / numpy errors propagate loudly.
    conf_mat = metric.compute().cpu().numpy()
    conf_mat_norm = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-8)

    try:
        import wandb
        import wandb.errors as _wandb_errors
        import plotly.express as px
    except ImportError as exc:
        logger.warning(
            "log_confusion_matrix: required import failed (prefix=%s): %s",
            prefix,
            exc,
        )
        return

    side = 1500
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_path = tmp_dir / f"{metric_name}.png"
    norm_path = tmp_dir / f"{metric_name}_norm.png"

    # Only wandb network/IO and plotly image-writer errors are swallowed.
    # Logic errors (AttributeError, KeyError, TypeError) propagate.
    try:
        fig = px.imshow(
            conf_mat,
            x=class_names,
            y=class_names,
            labels=dict(x="Predicted", y="Actual"),
            width=side,
            height=side,
        )
        fig.write_image(base_path)
        wandb.log({metric_name: wandb.Image(str(base_path))})

        fig_norm = px.imshow(
            conf_mat_norm,
            x=class_names,
            y=class_names,
            labels=dict(x="Predicted", y="Actual"),
            width=side,
            height=side,
        )
        fig_norm.write_image(norm_path)
        wandb.log({f"{metric_name}_normalized": wandb.Image(str(norm_path))})
    except (_wandb_errors.CommError, OSError, RuntimeError) as exc:
        logger.warning(
            "log_confusion_matrix failed for prefix=%s metric=%s: %s",
            prefix,
            metric_name,
            exc,
        )


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Seed python, numpy, torch, and cuda RNGs for reproducibility.

    What this guarantees:
        - ``random``, ``numpy.random``, and ``torch`` (CPU + all CUDA devices)
          are seeded in the calling process.
        - cuDNN is placed in deterministic mode (``cudnn.deterministic=True``,
          ``cudnn.benchmark=False``).

    What this does NOT guarantee:
        - DataLoader worker reproducibility. Each worker process has its own
          ``random`` / ``numpy.random`` state that this function cannot reach.
          Pair a ``torch.Generator`` from :func:`make_generator` with
          :func:`worker_init_fn` on the DataLoader so worker-side augmentations
          are reproducible.
        - Bit-exact determinism. Many CUDA kernels (e.g. scatter_add, atomic
          ops inside transformer attention, some convolutions) are
          non-deterministic even with ``cudnn.deterministic=True``. Enabling
          full bit-determinism via ``torch.use_deterministic_algorithms(True)``
          is intentionally not done here: it costs ~15-25% throughput on this
          model without closing the non-determinism gap (augmentations are
          stochastic at the dataset level, so training is only reproducible
          when DataLoader workers are seeded via ``worker_init_fn``).

    Args:
        seed: Seed to use for all RNGs.
        deterministic: When True, also sets ``CUBLAS_WORKSPACE_CONFIG`` so that
            cuBLAS reductions are deterministic. Off by default because it
            slightly reduces throughput on cuBLAS-heavy layers (e.g. the
            transformer MLPs).
    """
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Required for deterministic cuBLAS on CUDA >= 10.2. See
        # https://pytorch.org/docs/stable/notes/randomness.html
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int):
    """DataLoader ``worker_init_fn`` that seeds RNGs inside each worker.

    PyTorch already derives a per-worker seed (``torch.initial_seed()``) from
    the DataLoader's ``generator``; this helper propagates that seed to the
    ``random`` and ``numpy.random`` module-level RNGs used by augmentations
    and dataset code. Without this, two runs with the same
    ``seed_everything(42)`` can differ by ~0.1-0.3pp macro accuracy because
    worker-side augmentation RNGs are not seeded.

    Usage::

        gen = make_generator(seed=42)
        loader = DataLoader(..., generator=gen, worker_init_fn=worker_init_fn)
    """
    import random

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded to ``seed``.

    Pair with :func:`worker_init_fn` on the DataLoader so that worker
    processes inherit a deterministic sub-seed from this generator::

        gen = make_generator(seed=42)
        loader = DataLoader(..., generator=gen, worker_init_fn=worker_init_fn)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def get_tissue_ct_exclude(batch_data, dct_config, label_remap=None):
    """Build per-sample tissue-aware ct exclusion list.

    Args:
        batch_data: BatchData instance
        dct_config: TissueNetConfig instance
        label_remap: Optional lookup tensor from build_label_remap(). If provided,
            remaps ct2idx values to compact 0-indexed space (required when model
            outputs use compact indices).
    """
    ct_exclude = []
    for ds_name in batch_data.dataset_name:
        excluded = dct_config.get_excluded_ct_indices(ds_name)
        if label_remap is not None and excluded:
            excluded = [label_remap[idx].item() for idx in excluded]
        ct_exclude.append(excluded)
    return ct_exclude if any(ct_exclude) else None


# Shared baseline utilities
def _conf_mat_summary(conf_mat: np.ndarray) -> dict:
    """Shared confusion-matrix → {macro/weighted accuracy, macro/weighted F1}.

    Derives accuracy and F1 from the same (optionally hierarchy-adjusted)
    confusion matrix so that baselines and the main model report identically.
    F1 follows sklearn's macro / weighted reductions:
      - per-class precision[i] = TP[i] / max(TP[i] + FP[i], 1)
      - per-class recall[i]    = TP[i] / max(TP[i] + FN[i], 1)
      - per-class f1[i]        = 2 * P[i] * R[i] / max(P[i] + R[i], 1)
      - macro_f1               = mean(f1[i] for classes with support > 0)
      - weighted_f1            = sum(support[i] * f1[i]) / sum(support[i])
    """
    support = conf_mat.sum(axis=1)
    diag = np.diag(conf_mat).astype(np.float64)
    has_support = support > 0

    # Accuracy
    per_class_acc = diag / (support + 1e-8)
    macro_acc = float(np.mean(per_class_acc[has_support])) if has_support.any() else 0.0
    weighted_acc = float(diag.sum() / (conf_mat.sum() + 1e-8))

    # F1 (sklearn convention; safe denominators)
    predicted = conf_mat.sum(axis=0).astype(np.float64)
    precision = np.where(predicted > 0, diag / np.maximum(predicted, 1), 0.0)
    recall = np.where(support > 0, diag / np.maximum(support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    macro_f1 = float(np.mean(f1[has_support])) if has_support.any() else 0.0
    total_support = float(support.sum())
    weighted_f1 = (
        float(np.sum(support[has_support] * f1[has_support]) / max(total_support, 1.0))
        if total_support > 0
        else 0.0
    )

    return {
        "macro_accuracy": macro_acc,
        "weighted_accuracy": weighted_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    hierarchy: dict = None,
    ct2idx: dict = None,
) -> dict:
    """
    Compute classification metrics for baselines (matches main model contract).

    Reports macro and weighted accuracy + macro and weighted F1, derived from
    a single (optionally hierarchy-adjusted) confusion matrix. The same logic
    runs inside ``LossesAndMetrics.compute()`` for the main model, so a
    side-by-side table of baseline vs main numbers is apples-to-apples.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (N, num_classes)
        num_classes: Number of classes
        hierarchy: Optional cell type hierarchy for hierarchical evaluation.
            Predictions of child types count as correct for parent types.
        ct2idx: Optional mapping from cell type names to indices matching
            the label space used in y_true/y_pred (compact 0-indexed).

    Returns:
        Dictionary with keys macro_accuracy, weighted_accuracy, macro_f1,
        weighted_f1, confusion_matrix.
    """
    from sklearn.metrics import confusion_matrix

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if hierarchy and ct2idx:
        conf_mat = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    summary = _conf_mat_summary(conf_mat)
    summary["confusion_matrix"] = conf_mat
    return summary


def save_baseline_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cell_indices: list,
    dataset_names: list,
    fov_names: list,
    ct2idx: dict,
    output_path: Path,
):
    """
    Save predictions in the same format as PredLogger (for baselines).

    Args:
        y_true: True labels (N,)
        y_prob: Predicted probabilities (N, num_classes)
        cell_indices: Cell indices
        dataset_names: Dataset names
        fov_names: FOV names
        ct2idx: Cell type to index mapping
        output_path: Output CSV path
    """
    # Get column names sorted by index
    columns = sorted(ct2idx, key=ct2idx.get)
    idx2ct = {v: k for k, v in ct2idx.items()}

    df = pd.DataFrame(y_prob, columns=columns)
    df["cell_type_actual"] = [idx2ct[label] for label in y_true]
    df["cell_index"] = cell_indices
    df["dataset_name"] = dataset_names
    df["fov_name"] = fov_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def _extract_all_dataset_features(
    zarr_dir: str,
    dct_config,
    dataset_keys: list,
    min_channels: int = 0,
    global_cache_path: str = None,
) -> dict:
    """Extract per-dataset features from zarr, with split-agnostic caching.

    This is the expensive I/O step. Results are cached so that multiple
    holdout experiments can reuse the same extracted features with
    different train/val splits.

    Args:
        zarr_dir: Path to tissuenet zarr archive
        dct_config: TissueNetConfig instance
        dataset_keys: List of dataset keys to process
        min_channels: Minimum model-visible marker channels per dataset
        global_cache_path: If provided, cache all per-dataset features here.
            Subsequent calls with the same path skip extraction entirely.

    Returns:
        dict mapping dataset_key -> {features, labels, cell_indices}
        where features is (N, num_markers), labels is (N,), cell_indices is (N,)
    """
    import zarr
    from tqdm import tqdm

    expected_cache_meta = _feature_cache_metadata(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        min_channels=min_channels,
        dataset_keys=dataset_keys,
    )

    # Check global cache
    if global_cache_path is not None:
        cache_file = Path(global_cache_path)
        if cache_file.exists():
            # Reject any cache not owned by the current user before pickle.load.
            # The path is often predictable (passed as a CLI arg), so on a shared
            # filesystem another user could replace it with a crafted pickle that
            # would execute arbitrary code at load time. Owner check +
            # world-writable rejection blocks that vector. Mirrors the cell-data
            # cache fix introduced in PR #55.
            try:
                st = cache_file.stat()
            except OSError as e:
                logger.warning(
                    "global feature cache stat failed (%s); rebuilding", e
                )
                st = None
            if st is not None and st.st_uid != os.getuid():
                logger.warning(
                    "global feature cache at %s not owned by current user "
                    "(uid=%d, expected %d) — rejecting and rebuilding",
                    cache_file, st.st_uid, os.getuid(),
                )
                st = None  # skip load
            if st is not None and st.st_mode & 0o002:
                logger.warning(
                    "global feature cache at %s is world-writable — "
                    "rejecting and rebuilding",
                    cache_file,
                )
                st = None
            if st is None:
                cached = None
                _load_failed = True
            else:
                print(f"Loading global feature cache from {global_cache_path}")
                _load_failed = False
                try:
                    with open(cache_file, "rb") as f:
                        cached = pickle.load(f)
                except (OSError, EOFError, pickle.UnpicklingError) as e:
                    print(
                        f"Ignoring unreadable global feature cache ({e}); rebuilding"
                    )
                    _load_failed = True
            if _load_failed:
                pass  # fall through to rebuild
            else:
                saved_meta = None
                if isinstance(cached, dict) and "__metadata__" in cached:
                    saved_meta = cached.get("__metadata__")
                    cached = cached.get("datasets", {})
                mismatches = _cache_metadata_mismatches(saved_meta, expected_cache_meta)
                if mismatches:
                    print(
                        "Ignoring stale global feature cache "
                        f"({', '.join(mismatches)}); rebuilding"
                    )
                else:
                    # Filter to requested datasets
                    result = {k: v for k, v in cached.items() if k in set(dataset_keys)}
                    print(f"Loaded features for {len(result)} datasets from cache")
                    return result

    zf = zarr.open_group(zarr_dir, mode="r")
    ct_mapping = dct_config.celltype_mapping
    ct2idx = dct_config.ct2idx
    marker2idx = dct_config.marker2idx
    num_markers = len(marker2idx)

    def resolve_marker_idx(ch_name):
        # Strict canonical contract: direct marker2idx lookup only.
        # Source-data variants must be canonicalized at ingestion.
        return marker2idx.get(ch_name)

    per_dataset = {}
    for dataset_key in tqdm(dataset_keys, desc="Extracting features from zarr"):
        ds = zf[dataset_key]
        if "preprocessed" not in ds:
            continue

        preproc = ds["preprocessed"]
        channel_names = list(preproc.attrs.get("channel_names", []))
        if not channel_names:
            continue

        if min_channels > 0:
            num_real = sum(
                1 for c in channel_names if resolve_marker_idx(c) is not None
            )
            if num_real < min_channels:
                continue

        cell_data = _get_cell_data_from_ds(ds, dataset_key, preproc)
        if cell_data is None:
            continue
        cell_types_raw, cell_indices = cell_data

        ds_ct_mapping = ct_mapping.get(dataset_key, {})
        valid_cells = []
        for ct_label, cell_idx in zip(cell_types_raw, cell_indices):
            ct_label = str(ct_label)
            ct_standard = ds_ct_mapping.get(ct_label, ct_label)
            if ct_standard in ct2idx:
                valid_cells.append((ct_standard, cell_idx))

        if not valid_cells:
            continue

        mask = preproc["mask"][:]
        mask_flat = mask.ravel()
        max_label = int(mask_flat.max()) + 1
        counts = np.bincount(mask_flat, minlength=max_label)

        ct_labels = [vc[0] for vc in valid_cells]
        cell_idxs = np.array([vc[1] for vc in valid_cells])

        features = np.zeros((len(valid_cells), num_markers), dtype=np.float32)
        cell_counts = counts[cell_idxs]
        cell_counts_safe = np.where(cell_counts > 0, cell_counts, 1)
        for c, ch_name in enumerate(channel_names):
            global_idx = resolve_marker_idx(ch_name)
            if global_idx is None:
                continue
            channel_data = preproc["raw"][c]
            sums = np.bincount(
                mask_flat,
                weights=channel_data.ravel().astype(np.float64),
                minlength=max_label,
            )
            features[:, global_idx] = sums[cell_idxs] / cell_counts_safe

        labels = np.array([ct2idx[ct] for ct in ct_labels], dtype=np.int64)

        per_dataset[dataset_key] = {
            "features": features,
            "labels": labels,
            "cell_indices": cell_idxs,
            # Per-cell pixel area (cellSize) — canonical mahmoodlab/MAPS feeds
            # this as an extra input column (paper Methods + canonical README).
            # Stored separately so consumers that don't need it can ignore it.
            "cell_sizes": cell_counts.astype(np.float32),
        }

    # Save global cache
    if global_cache_path is not None:
        cache_file = Path(global_cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_pickle_dump(
            {
                "__metadata__": expected_cache_meta,
                "datasets": per_dataset,
            },
            cache_file,
            protocol=4,
        )
        print(
            f"Saved global feature cache ({len(per_dataset)} datasets) to {global_cache_path}"
        )

    return per_dataset


def extract_features_from_zarr(
    zarr_dir: str,
    dct_config,
    split_file: str = None,
    skip_datasets=None,
    keep_datasets=None,
    cache_path: str = None,
    min_channels: int = 0,
    global_cache_path: str = None,
    strict_split: bool = True,
) -> dict:
    """Extract mean intensity features directly from zarr (fast path for baselines).

    Processes whole FOVs at once using np.bincount, avoiding the slow
    per-cell patch extraction pipeline. ~20-50x faster than DataLoader approach.

    Args:
        zarr_dir: Path to tissuenet zarr archive
        dct_config: TissueNetConfig instance
        split_file: Path to pre-computed FOV split JSON
        skip_datasets: Dataset keys to skip
        keep_datasets: Dataset keys to keep
        cache_path: If provided, cache split-specific features to this .npz file.
            On subsequent calls, loads from cache if it exists.
            NOTE: This cache is split-specific. Use global_cache_path for
            cross-holdout reuse.
        min_channels: Minimum number of model-visible marker channels required per dataset.
            Datasets with fewer are excluded. Default 0 (no filtering).
        global_cache_path: If provided, cache ALL per-dataset features to this
            pickle file (split-agnostic). Subsequent calls with any split_file
            skip the expensive zarr I/O and just apply the split to cached data.
            This is the recommended approach for holdout experiments where
            multiple splits share the same underlying data.
        strict_split: If True, reject split files whose train/val FOVs are
            absent from the archive, absent after feature extraction, or overlap.

    Returns:
        dict with keys:
            X_train, y_train, train_dataset_names, train_fov_names, train_cell_indices
            X_val, y_val, val_dataset_names, val_fov_names, val_cell_indices
            metadata: dict with active_datasets, num_samples
    """
    import json
    import zarr

    # Determine which datasets to process. This happens before cache loading so
    # cache provenance includes the exact requested dataset set.
    zf = zarr.open_group(zarr_dir, mode="r")
    from deepcell_types.training.config import _discover_fov_keys

    all_dataset_keys = _discover_fov_keys(zf)

    if keep_datasets:
        dataset_keys = [k for k in all_dataset_keys if k in set(keep_datasets)]
    elif skip_datasets:
        skip_set = (
            set(skip_datasets) if not isinstance(skip_datasets, set) else skip_datasets
        )
        dataset_keys = [k for k in all_dataset_keys if k not in skip_set]
    else:
        dataset_keys = all_dataset_keys

    print(f"Found {len(dataset_keys)} datasets in tissuenet archive")

    expected_cache_meta = _feature_cache_metadata(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        min_channels=min_channels,
        dataset_keys=dataset_keys,
        split_file=split_file,
    )

    # Check for split-specific cached features (legacy behavior)
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"Loading cached features from {cache_path}")
            try:
                with np.load(cache_file, allow_pickle=True) as cached:
                    saved_meta = None
                    if "cache_metadata" in cached.files:
                        saved_meta = json.loads(str(cached["cache_metadata"].item()))
                    mismatches = _cache_metadata_mismatches(
                        saved_meta, expected_cache_meta
                    )
                    if mismatches:
                        print(
                            "Ignoring stale split-specific feature cache "
                            f"({', '.join(mismatches)}); rebuilding"
                        )
                    else:
                        out = {}
                        for key in cached.files:
                            if key == "cache_metadata":
                                continue
                            val = cached[key]
                            if val.dtype == object:
                                out[key] = list(val)
                            else:
                                out[key] = val
                        print(
                            f"Loaded {len(out['X_train'])} train, {len(out['X_val'])} val samples from cache"
                        )
                        return out
            except (
                OSError,
                ValueError,
                EOFError,
                BadZipFile,
                json.JSONDecodeError,
                pickle.UnpicklingError,
            ) as e:
                print(
                    "Ignoring unreadable split-specific feature cache "
                    f"({e}); rebuilding"
                )

    # Load split file
    if split_file is None:
        raise ValueError("split_file is required for extract_features_from_zarr")
    with open(split_file) as f:
        split_data = json.load(f)

    train_fov_set = set()
    for ds_name, fov_names in split_data["train"].items():
        for fov_name in fov_names:
            train_fov_set.add((ds_name, fov_name))
    val_fov_set = set()
    for ds_name, fov_names in split_data["val"].items():
        for fov_name in fov_names:
            val_fov_set.add((ds_name, fov_name))
    heldout_fov_set = set()
    for ds_name, fov_names in split_data.get("heldout", {}).items():
        for fov_name in fov_names:
            heldout_fov_set.add((ds_name, fov_name))

    train_val_overlap = train_fov_set & val_fov_set
    if train_val_overlap:
        msg = (
            f"{len(train_val_overlap)} FOVs appear in both train and val splits: "
            f"{_format_examples(train_val_overlap)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    heldout_overlap = heldout_fov_set & (train_fov_set | val_fov_set)
    if heldout_overlap:
        msg = (
            f"{len(heldout_overlap)} heldout FOVs also appear in train/val: "
            f"{_format_examples(heldout_overlap)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    split_fov_set = train_fov_set | val_fov_set
    unsupported_fov_keys = {
        pair for pair in (split_fov_set | heldout_fov_set) if pair[0] != pair[1]
    }
    if unsupported_fov_keys:
        msg = (
            "baseline feature extraction requires split dataset names and FOV names "
            f"to match archive keys; unsupported entries: {_format_examples(unsupported_fov_keys)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    split_dataset_keys = {
        ds_name for ds_name, fov_name in split_fov_set if ds_name == fov_name
    }
    heldout_dataset_keys = {
        ds_name for ds_name, fov_name in heldout_fov_set if ds_name == fov_name
    }
    all_dataset_key_set = set(all_dataset_keys)
    dataset_key_set = set(dataset_keys)

    missing_archive = split_dataset_keys - all_dataset_key_set
    if missing_archive:
        msg = (
            f"{len(missing_archive)} split FOVs are not present in the archive: "
            f"{_format_examples(missing_archive)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    missing_heldout_archive = heldout_dataset_keys - all_dataset_key_set
    if missing_heldout_archive:
        msg = (
            f"{len(missing_heldout_archive)} heldout FOVs are not present in the archive: "
            f"{_format_examples(missing_heldout_archive)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    filtered_split_keys = split_dataset_keys - dataset_key_set
    if filtered_split_keys:
        msg = (
            f"{len(filtered_split_keys)} split FOVs are excluded by "
            f"keep_datasets/skip_datasets filters: {_format_examples(filtered_split_keys)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    # Extract features — use global cache if available
    per_dataset = _extract_all_dataset_features(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        dataset_keys=dataset_keys,
        min_channels=min_channels,
        global_cache_path=global_cache_path,
    )

    processed_keys = set(per_dataset)
    active_split_keys = split_dataset_keys & dataset_key_set
    missing_features = active_split_keys - processed_keys
    if missing_features:
        msg = (
            f"{len(missing_features)} split FOVs produced no features after "
            f"filters/annotation loading: {_format_examples(missing_features)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    unassigned_features = processed_keys - split_dataset_keys - heldout_dataset_keys
    if unassigned_features:
        msg = (
            f"{len(unassigned_features)} processed FOVs are absent from the split file: "
            f"{_format_examples(unassigned_features)}"
        )
        if strict_split:
            raise ValueError(msg)
        logger.warning(msg)

    # Apply split to per-dataset features
    results = {
        "train": {
            "features": [],
            "labels": [],
            "ds_names": [],
            "fov_names": [],
            "cell_indices": [],
            "cell_sizes": [],
        },
        "val": {
            "features": [],
            "labels": [],
            "ds_names": [],
            "fov_names": [],
            "cell_indices": [],
            "cell_sizes": [],
        },
    }

    for dataset_key, data in per_dataset.items():
        if (dataset_key, dataset_key) in train_fov_set:
            split = "train"
        elif (dataset_key, dataset_key) in val_fov_set:
            split = "val"
        else:
            continue

        n_cells = len(data["labels"])
        acc = results[split]
        acc["features"].append(data["features"])
        acc["labels"].append(data["labels"])
        acc["ds_names"].extend([dataset_key] * n_cells)
        acc["fov_names"].extend([dataset_key] * n_cells)
        acc["cell_indices"].extend(data["cell_indices"].tolist())
        # Defensive: missing in pre-cache_v5 entries — recompute as 1.0 fallback if absent
        cell_sizes = data.get("cell_sizes")
        if cell_sizes is None:
            cell_sizes = np.ones(n_cells, dtype=np.float32)
        acc["cell_sizes"].append(cell_sizes)

    # Concatenate
    num_markers = len(dct_config.marker2idx)
    out = {"metadata": {"active_datasets": dataset_keys, "num_samples": 0}}
    for split in ("train", "val"):
        acc = results[split]
        if acc["features"]:
            X = np.concatenate(acc["features"], axis=0)
            y = np.concatenate(acc["labels"], axis=0)
            cell_sizes = np.concatenate(acc["cell_sizes"], axis=0)
        else:
            X = np.zeros((0, num_markers), dtype=np.float32)
            y = np.zeros(0, dtype=np.int64)
            cell_sizes = np.zeros(0, dtype=np.float32)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        out[f"{split}_dataset_names"] = acc["ds_names"]
        out[f"{split}_fov_names"] = acc["fov_names"]
        out[f"{split}_cell_indices"] = acc["cell_indices"]
        out[f"{split}_cell_sizes"] = cell_sizes
        out["metadata"]["num_samples"] = out["metadata"].get("num_samples", 0) + len(y)

    meta = split_data.get("metadata", {})
    print(
        f"Loaded FOV splits from {split_file} "
        f"(created {meta.get('created', 'unknown')}): "
        f"{len(out['X_train'])} train, {len(out['X_val'])} val"
    )

    # Save split-specific cache if requested (legacy behavior)
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {}
        for key, val in out.items():
            if key == "metadata":
                continue
            if isinstance(val, list):
                save_dict[key] = np.array(val, dtype=object)
            else:
                save_dict[key] = val
        save_dict["cache_metadata"] = np.array(
            json.dumps(expected_cache_meta),
            dtype=object,
        )
        _atomic_np_savez(cache_file, **save_dict)
        print(f"Cached features to {cache_path}")

    return out


def _get_cell_data_from_ds(ds, dataset_key, preproc):
    """Extract cell types and indices from a dataset.

    Uses annotations as primary path (covers ~2155 datasets), with
    cell_type_info arrays as fallback (only 3 datasets have labels there).
    Annotation centroid values are in original image coordinates;
    scale_factor is applied before KDTree matching against preprocessed centroids.

    Returns:
        (cell_types, cell_indices) or None
    """
    from deepcell_types.training.annotations import extract_cell_annotations

    return extract_cell_annotations(ds, dataset_key, preproc, include_centroids=False)
