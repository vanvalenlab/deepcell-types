"""Cell-type / marker-positivity metric trackers.

Split out of ``training/utils.py`` so callers can import metric helpers without
pulling in the baseline-feature / IO surface. ``utils.py`` keeps re-exports at
the bottom for backward compatibility with external callers.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


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

        # Symmetric vacuous-marker exclusion across precision/recall/F1/accuracy
        # — otherwise the triangle identity macro_f1 ≈ 2pr/(p+r) breaks because
        # prec/rec would include 0.0 for vacuous markers while F1 used nanmean
        # (PR #55 fix). Accuracy is also excluded so the four macro reductions
        # share the same denominator (mp_num_markers - excluded).
        n_pos_gt = tp + fn
        n_pos_pred = tp + fp
        if n_pos_gt == 0 and n_pos_pred == 0:
            f1s.append(np.nan)
            precisions.append(np.nan)
            recalls.append(np.nan)
            accuracies.append(np.nan)
            excluded += 1
        else:
            f1s.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            accuracies.append(acc)

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
        "mp_macro_accuracy": (
            float(np.nanmean(accuracies))
            if len(accuracies) > 0 and not np.all(np.isnan(accuracies))
            else 0.0
        ),
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
        # macro-F1 is the single cell-type quality metric used throughout this
        # repo (robust to TissueNet's class imbalance, where accuracy is
        # inflated by over-predicting majority classes). Derived from the
        # (optionally hierarchy-adjusted) confusion matrix via the canonical
        # _conf_mat_summary() so the main model, baselines, and the abstention
        # evaluator all report the same quantity. Imported lazily because
        # baseline_features.py imports adjust_conf_mat_hierarchy from this
        # module, so a top-level import would cycle.
        from .baseline_features import _conf_mat_summary

        conf_mat = self.conf_mat_ct_metric.compute().cpu().numpy()
        if self.hierarchy and self.ct2idx:
            conf_mat = adjust_conf_mat_hierarchy(conf_mat, self.hierarchy, self.ct2idx)
        summary = _conf_mat_summary(conf_mat)
        mp = self.mp_metrics.compute()
        return {
            "ct_macro_f1": summary["macro_f1"],
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


