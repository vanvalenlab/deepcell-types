"""FOV-grouped samplers and sample-weight computation.

Extracted from ``deepcell_types.training.dataset`` for modularity. These
symbols are re-exported from ``dataset`` for backward compatibility.

Contains the sqrt-inverse-frequency weight helper (``compute_sample_weights``),
the full-inverse-frequency / equal-proportion helpers
(``compute_sample_weights_equal``, ``subsample_indices_per_class`` — the
faithful CellSighter recipe), and the two FOV-grouped samplers
(``FOVGroupedSampler``, ``SequentialFOVGroupedSampler``) that preserve
per-worker numpy-cache locality by emitting same-FOV cells contiguously.
"""

import random

import numpy as np
import torch
from torch.utils.data import Sampler


def compute_sample_weights_dct(labels, *, floor: int = 1000):
    """DCT sampler weights as a pure label-array helper (one source of truth).

    Sqrt-inverse-frequency with a minimum effective-count ``floor``:
    ``weight(class) = sqrt(total / max(count, floor))``. This is the exact
    balancing the main DeepCell-Types model uses (via ``compute_sample_weights``
    over a dataset) and that CellSighter selects with ``--class_balance sqrt``.
    Exposed as a label-array helper so the MAPS and XGBoost baselines — which
    operate on numpy ``y`` arrays rather than a ``dataset`` — can share the
    identical formula, giving every method the same sampler.

    The ``floor`` treats any class as having at least ``floor`` samples for
    weighting, preventing rare single-FOV classes (e.g. Myofibroblast with 236
    cells) from receiving extreme weights that corrupt common classes'
    representations.

    Args:
        labels: 1-D array-like of per-sample class labels.
        floor: Minimum effective per-class count (default 1000).

    Returns:
        weights: np.ndarray (float64) of per-sample weights, aligned to ``labels``.
    """
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum())
    cls_weight = {
        cls: float(np.sqrt(total / max(int(count), floor)))
        for cls, count in zip(uniq.tolist(), counts.tolist())
    }
    return np.array([cls_weight[c] for c in labels.tolist()], dtype=np.float64)


def compute_sample_weights(dataset, indices):
    """Compute sqrt-inverse-frequency sample weights for WeightedRandomSampler.

    Thin dataset-interface wrapper over :func:`compute_sample_weights_dct` (the
    shared formula); see that helper for the weighting rationale.

    Args:
        dataset: FullImageDataset instance
        indices: List of indices to compute weights for

    Returns:
        weights: torch.Tensor of per-sample weights
    """
    labels = np.array([dataset.indices[i].ct_label_standard for i in indices])
    return torch.from_numpy(compute_sample_weights_dct(labels)).float()


def compute_sample_weights_equal(dataset, indices):
    """Full-inverse-frequency (equal-proportion) per-sample weights.

    Faithful reproduction of the original CellSighter ``define_sampler``
    (KerenLab/CellSighter ``train.py``): per class, ``weight = total / count``
    (no sqrt, no minimum-count floor), so under a ``WeightedRandomSampler``
    every class receives equal *expected* representation per epoch — the paper's
    "upsample rare cells so major lineages are represented in equal
    proportions". This is more aggressive than the DCT-wide
    ``compute_sample_weights`` (sqrt-inverse-frequency with a 1000-count floor).

    Pair with ``subsample_indices_per_class`` to reproduce CellSighter's
    ``size_data`` cap (``subsample_const_size``): cap the per-class training
    pool first, then weight the capped pool here.

    Args:
        dataset: FullImageDataset instance.
        indices: List of indices to compute weights for.

    Returns:
        weights: torch.Tensor of per-sample weights.
    """
    from collections import defaultdict

    ct_counts = defaultdict(int)
    for i in indices:
        ct_counts[dataset.indices[i].ct_label_standard] += 1

    total = sum(ct_counts.values())
    ct_weights = {ct: total / count for ct, count in ct_counts.items()}

    weights = torch.zeros(len(indices))
    for i, idx in enumerate(indices):
        weights[i] = ct_weights[dataset.indices[idx].ct_label_standard]

    return weights


def subsample_indices_per_class(dataset, indices, size_data, *, seed=42):
    """Cap each class's training pool to at most ``size_data`` cells.

    Faithful reproduction of CellSighter's ``subsample_const_size`` (applied to
    the train crops before sampler construction): classes with more than
    ``size_data`` cells are randomly subsampled down to ``size_data``; classes
    at or below the cap keep all their cells. Large, redundant classes thus lose
    per-epoch diversity while rare classes are untouched — combined with
    ``compute_sample_weights_equal`` this is the paper's training recipe.

    The subsample is deterministic for a given ``seed`` (independent of global
    RNG state) and preserves the input ordering so downstream FOV grouping is
    unaffected.

    Args:
        dataset: FullImageDataset instance.
        indices: List of integer indices into ``dataset`` (the train subset).
        size_data: Per-class cell cap (e.g. 1000, matching CellSighter). If
            ``None``, the indices are returned unchanged.
        seed: Base seed for the deterministic per-class subsample.

    Returns:
        List of indices (a subset of ``indices``), order-preserving.
    """
    from collections import defaultdict

    if size_data is None:
        return list(indices)

    by_class = defaultdict(list)
    for idx in indices:
        by_class[dataset.indices[idx].ct_label_standard].append(idx)

    rng = random.Random(int(seed))
    keep = set()
    for ct in sorted(by_class):  # sort for seed-stable iteration order
        idxs = by_class[ct]
        if len(idxs) > size_data:
            keep.update(rng.sample(idxs, size_data))
        else:
            keep.update(idxs)

    # Preserve original ordering so FOV-group cache locality downstream is kept.
    return [idx for idx in indices if idx in keep]


class FOVGroupedSampler(Sampler):
    """Wraps a WeightedRandomSampler to group samples by FOV for cache locality.

    Draws all indices via weighted sampling (preserving class balance), then
    sorts them by FOV so consecutive samples come from the same FOV. This
    makes per-worker numpy array caching effective: each FOV is loaded once
    and reused for all its cells before being evicted.

    FOV groups are shuffled each epoch to avoid always processing FOVs in the
    same order, while cells within each FOV group remain together.
    """

    def __init__(
        self,
        weights,
        num_samples,
        dataset_indices,
        train_indices,
        replacement=True,
        seed=42,
    ):
        """
        Args:
            weights: Per-sample weights tensor of length len(train_indices).
            num_samples: Number of samples to draw per epoch.
            dataset_indices: dataset.indices list (full dataset).
            train_indices: List of integer indices into dataset (the train subset).
                           Must have len(train_indices) == len(weights).
            replacement: Whether to sample with replacement.
            seed: Base seed for per-epoch group shuffle and the multinomial draw.
                Combined with an internal epoch counter so two runs with the same
                seed produce identical FOV ordering across epochs (independent of
                global ``random``/``torch`` state).
        """
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self._base_seed = int(seed)
        self._epoch = 0
        # Map sampler position i -> ds_idx of the i-th train sample (Fix 1).
        # drawn[i] is a position in [0, len(train_indices)), so _ds_idx_map[drawn[i]]
        # gives the correct ds_idx for FOV grouping.
        self._ds_idx_map = torch.tensor(
            [
                dataset_indices[train_indices[i]].ds_idx
                for i in range(len(train_indices))
            ],
            dtype=torch.long,
        )

    def __iter__(self):
        if self.num_samples <= 0:
            return

        # Per-epoch deterministic generators. Independent of any global RNG so
        # two runs with the same --seed produce the same FOV order regardless
        # of intervening calls to random.* / torch.*.
        epoch_seed = (self._base_seed + self._epoch) & 0xFFFFFFFF
        torch_gen = torch.Generator(device="cpu").manual_seed(epoch_seed)
        py_rng = random.Random(epoch_seed)
        self._epoch += 1

        # Draw weighted samples (same as WeightedRandomSampler)
        drawn = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=torch_gen
        )

        # Group by FOV (ds_idx), shuffle groups, yield
        ds_indices = self._ds_idx_map[drawn]
        # Sort by ds_idx to group same-FOV samples together
        sorted_order = ds_indices.argsort(stable=True)
        sorted_drawn = drawn[sorted_order]

        # Shuffle at FOV-group level (find group boundaries, permute groups)
        sorted_ds = ds_indices[sorted_order]
        # Find where ds_idx changes
        changes = torch.where(sorted_ds[1:] != sorted_ds[:-1])[0] + 1
        boundaries = torch.cat(
            [torch.tensor([0]), changes, torch.tensor([len(sorted_drawn)])]
        )

        # Build list of FOV groups and shuffle them with a per-epoch RNG
        groups = []
        for i in range(len(boundaries) - 1):
            groups.append(sorted_drawn[boundaries[i] : boundaries[i + 1]])
        py_rng.shuffle(groups)

        # Yield indices in group order
        result = torch.cat(groups)
        yield from result.tolist()

    def __len__(self):
        return self.num_samples


class SequentialFOVGroupedSampler(Sampler):
    """One-pass sampler that visits every train index in FOV-grouped order.

    Each FOV's cells are emitted contiguously, and the order of FOV groups
    is shuffled per-epoch with a deterministic seed. This is the unweighted
    counterpart to ``FOVGroupedSampler`` — same cache-locality guarantee
    (each worker reads one FOV at a time, so the per-FOV ~1 GB cold zarr
    load is amortised across all of that FOV's cells), but with uniform
    coverage instead of weighted multinomial sampling.

    Used by ``predict.py --learn_mp_thresholds`` (and the standalone
    threshold-learning helper) so that one-shot scans over the training
    split do not trigger the cold-zarr I/O storm that ``shuffle=True``
    produces under spawn workers on a large multi-FOV archive.
    """

    def __init__(
        self, dataset_indices, train_indices, seed: int = 42, max_samples=None
    ):
        """
        Args:
            dataset_indices: ``dataset.indices`` list (full dataset).
            train_indices: List of integer indices into ``dataset`` that
                participate in this pass.
            seed: Base seed for per-epoch group shuffle. Combined with an
                internal epoch counter so successive epochs visit FOVs in
                different orders without colliding across runs.
        """
        # The sampler is paired with ``Subset(dataset, train_indices)``, whose
        # ``__getitem__(idx)`` does ``self.dataset[self.indices[idx]]``. So we
        # MUST yield positions within ``train_indices`` (i.e. values in
        # ``[0, len(train_indices))``), not raw indices into ``dataset.indices``
        # — same contract as ``FOVGroupedSampler.__iter__``.
        train_indices = [int(i) for i in train_indices]
        self._n = len(train_indices)
        self._ds_idx_map = [int(dataset_indices[i].ds_idx) for i in train_indices]
        self._base_seed = int(seed)
        self._epoch = 0
        # Optional per-epoch cap. When set, each epoch emits the first
        # ``max_samples`` positions in the (per-epoch reshuffled) FOV-group
        # order, so successive epochs draw a different cache-local subset of
        # FOVs and full coverage is reached across epochs — uniform-distribution
        # counterpart to ``FOVGroupedSampler``'s ``num_samples`` cap.
        self._max = None if max_samples is None else int(max_samples)

    def _cap(self):
        return self._n if self._max is None else min(self._max, self._n)

    def __iter__(self):
        if self._n == 0:
            return
        groups: dict[int, list[int]] = {}
        for pos, ds_idx in enumerate(self._ds_idx_map):
            groups.setdefault(ds_idx, []).append(pos)

        epoch_seed = (self._base_seed + self._epoch) & 0xFFFFFFFF
        self._epoch += 1
        rng = random.Random(epoch_seed)
        ordered_ds = list(groups.keys())
        rng.shuffle(ordered_ds)

        cap = self._cap()
        emitted = 0
        for ds_idx in ordered_ds:
            for pos in groups[ds_idx]:
                if emitted >= cap:
                    return
                yield pos
                emitted += 1

    def __len__(self):
        return self._cap()
