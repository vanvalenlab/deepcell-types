"""DataLoader construction for training/validation.

Extracted from ``deepcell_types.training.dataset`` for modularity. These
symbols are re-exported from ``dataset`` for backward compatibility.

Contains ``create_dataloader`` (the full keyword API), the ``DataLoaderConfig``
dataclass that bundles its 20+ knobs, and ``create_dataloader_from_config``
(the dataclass-based wrapper). This module sits at the top of the training-data
dependency chain: it imports the dataset core, transforms, samplers, and split
helpers, but nothing imports it back.
"""

from dataclasses import dataclass, fields
from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import FullImageDataset
from .samplers import (
    FOVGroupedSampler,
    SequentialFOVGroupedSampler,
    compute_sample_weights,
)
from .splits import create_fov_splits, load_fov_splits
from .transforms import (
    AugmentedDataset,
    DropOutChannels,
    _Compose,
    _RandomHorizontalFlip,
    _RandomVerticalFlip,
)


def _carve_inner_val_fovs(dataset, train_indices, inner_val_ratio, inner_val_seed):
    """Carve a FOV-grouped inner-validation set out of ``train_indices``.

    Whole FOVs go to inner-train or inner-val (never split), so the inner-val
    set used for baseline checkpoint selection is disjoint — at FOV
    granularity — from inner-train (and therefore never selects on a cell that
    also trained the model). Returns ``(inner_train_indices, inner_val_indices)``;
    ``inner_val_indices`` is ``None`` when the carve is a no-op
    (``inner_val_ratio`` <= 0 or no train cells), leaving ``train_indices``
    unchanged so the main-model path is unaffected.
    """
    if not (inner_val_ratio and inner_val_ratio > 0.0 and len(train_indices) > 0):
        return list(train_indices), None
    fov_keys = [
        (dataset.indices[i][6], dataset.indices[i][5]) for i in train_indices
    ]  # (dataset_name, fov_name)
    unique_fovs = sorted(set(fov_keys))
    rng_iv = np.random.default_rng(inner_val_seed)
    perm = rng_iv.permutation(len(unique_fovs))
    n_iv = max(1, int(round(len(unique_fovs) * inner_val_ratio)))
    n_iv = min(n_iv, len(unique_fovs) - 1)  # always keep ≥1 inner-train FOV
    inner_val_fovs = {unique_fovs[p] for p in perm[:n_iv].tolist()}
    inner_train_indices = [
        i for i, k in zip(train_indices, fov_keys) if k not in inner_val_fovs
    ]
    inner_val_indices = [
        i for i, k in zip(train_indices, fov_keys) if k in inner_val_fovs
    ]
    return inner_train_indices, inner_val_indices


def create_dataloader(
    zarr_dir,
    dct_config,
    skip_datasets=None,
    keep_datasets=None,
    batch_size=256,
    num_dropout_channels=8,
    num_workers=16,
    only_test=False,
    keep_fovs=None,
    lengths=None,
    use_fov_splits=True,
    train_ratio=0.8,
    seed=42,
    use_weighted_sampler=True,
    split_file=None,
    skip_distance_transform=False,
    persistent_workers=False,
    max_samples_per_epoch=None,
    max_val_samples=None,
    multiprocessing_context=None,
    pin_memory=False,
    numpy_cache_max_bytes=None,
    fov_grouped_train: bool = False,
    inner_val_ratio: float = 0.0,
    inner_val_seed: int = 42,
):
    """Create dataloaders with factored representation.

    Args:
        zarr_dir: Path to tissuenet zarr archive
        dct_config: TissueNetConfig instance
        skip_datasets: Dataset keys to skip
        keep_datasets: Dataset keys to keep
        batch_size: Batch size
        num_dropout_channels: Channels to drop during training
        num_workers: DataLoader workers
        only_test: If True, return only test loader
        keep_fovs: FOV names to keep (for prediction on specific FOVs)
        lengths: Deprecated - use use_fov_splits instead
        use_fov_splits: Use FOV-level splits (default True, no leakage)
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed
        use_weighted_sampler: Use sqrt-frequency WeightedRandomSampler (default True)
        split_file: Path to pre-computed FOV split JSON (overrides use_fov_splits/seed)
        skip_distance_transform: Skip distance transform in patch extraction
        persistent_workers: Keep DataLoader workers alive between epochs
        max_samples_per_epoch: Cap the number of samples drawn per epoch by the
            WeightedRandomSampler. Useful for large datasets where iterating
            over all samples per epoch is impractical (e.g. 7M samples).
            If None (default), draws len(train_indices) samples per epoch.
        max_val_samples: Cap the validation set to this many samples (fixed random subset,
            seeded for reproducibility). Useful to keep validation fast. If None (default),
            evaluates all val cells.
        pin_memory: Pin DataLoader memory for faster CPU→GPU transfers (default False)
        numpy_cache_max_bytes: Optional per-worker numpy cache budget. If None,
            defaults to a 2 GiB total budget divided across workers.
        inner_val_ratio: If > 0 (and FOV splits are used), carve a FOV-grouped
            inner-validation set out of the TRAIN indices for leakage-free model
            selection. ``train_loader`` then iterates only the inner-train cells,
            and ``metadata["inner_val_loader"]`` holds a DataLoader over the
            held-out inner-val cells. The reported ``val_loader`` is unchanged.
            Default 0.0 is a no-op (main-model training path is unaffected). Used
            by the MAPS / CellSighter baselines so the reported set never drives
            checkpoint selection.
        inner_val_seed: Seed for the FOV-grouped inner-val carve (default 42).

    Returns:
        train_loader, val_loader, metadata dict
        (train_loader is None if only_test=True). When inner_val_ratio > 0,
        metadata["inner_val_loader"] is the held-out inner-val DataLoader.
    """
    train_transform = _Compose(
        [
            _RandomHorizontalFlip(),
            _RandomVerticalFlip(),
        ]
    )

    dropout_transform = DropOutChannels(num_dropout_channels)

    if numpy_cache_max_bytes is None:
        total_cache_budget = 2 * 1024**3
        if num_workers > 0:
            numpy_cache_max_bytes = max(
                128 * 1024**2,
                total_cache_budget // num_workers,
            )
        else:
            numpy_cache_max_bytes = total_cache_budget

    # Only use persistent_workers when num_workers > 0
    pw = persistent_workers and num_workers > 0

    dataset = FullImageDataset(
        zarr_dir,
        dct_config=dct_config,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        transform=None,
        keep_fovs=keep_fovs,
        skip_distance_transform=skip_distance_transform,
        numpy_cache_max_bytes=numpy_cache_max_bytes,
    )

    metadata = dataset.metadata

    if only_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=pw,
            pin_memory=pin_memory,
        )
        return None, test_loader, metadata

    if split_file is not None:
        use_fov_splits = True  # split_file implies FOV splits

    if use_fov_splits:
        if split_file is not None:
            train_indices, val_indices = load_fov_splits(dataset, split_file)
        else:
            train_indices, val_indices = create_fov_splits(
                dataset, train_ratio=train_ratio, seed=seed
            )

        # Optionally carve a FOV-grouped inner-validation set out of the TRAIN
        # indices, for callers that must not select on the reported (val) set
        # — e.g. the MAPS / CellSighter baselines. The carve is at FOV
        # granularity (whole FOVs go to inner-train or inner-val, never split),
        # mirroring the XGBoost baseline's GroupShuffleSplit early-stopping set.
        # Default inner_val_ratio=0.0 is a no-op, so the main-model training
        # path is unchanged.
        # Train only on inner-train; the sampler/loaders below all key off
        # ``train_indices``, so reassigning here is sufficient. Default
        # inner_val_ratio=0.0 is a no-op (returns train_indices unchanged, None).
        train_indices, inner_val_indices = _carve_inner_val_fovs(
            dataset, train_indices, inner_val_ratio, inner_val_seed
        )

        train_subset = torch.utils.data.Subset(dataset, train_indices)

        if max_val_samples is not None and max_val_samples < len(val_indices):
            rng = np.random.default_rng(seed)
            val_indices = rng.choice(
                val_indices, size=max_val_samples, replace=False
            ).tolist()
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        # Wrap train with augmentation
        train_dataset = AugmentedDataset(
            train_subset, train_transform, dropout_transform
        )

        # Weighted sampler for class balance
        sampler = None
        shuffle = True
        if use_weighted_sampler and len(train_indices) > 0:
            weights = compute_sample_weights(dataset, train_indices)
            num_samples = len(weights)
            if max_samples_per_epoch is not None:
                num_samples = min(num_samples, max_samples_per_epoch)
            sampler = FOVGroupedSampler(
                weights,
                num_samples,
                dataset.indices,
                train_indices,
                replacement=True,
                seed=seed,
            )
            shuffle = False
        elif fov_grouped_train and len(train_indices) > 0:
            # One-pass uniform sampler that preserves FOV cache locality.
            # `shuffle=True` over a multi-thousand-FOV archive forces every
            # worker to cold-load a fresh ~1 GB FOV per cell, which on spawn
            # workers manifests as the documented `--learn_mp_thresholds`
            # deadlock. Same locality guarantee as `FOVGroupedSampler`, but
            # with uniform coverage instead of weighted draws.
            sampler = SequentialFOVGroupedSampler(
                dataset.indices,
                train_indices,
                seed=seed,
                max_samples=max_samples_per_epoch,
            )
            shuffle = False

        mp_ctx = multiprocessing_context if num_workers > 0 else None
        # Wire per-worker RNG seeding so augmentation (`_RandomHorizontalFlip`,
        # `DropOutChannels`) is reproducible across runs with the same --seed.
        # Without this, two runs with --seed 42 differ by ~0.1-0.3pp macro
        # because PyTorch's default per-worker seed varies per process.
        from deepcell_types.training.utils import make_generator, worker_init_fn

        train_gen = make_generator(seed)
        val_gen = make_generator(seed + 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=4
            if num_workers > 0
            else None,  # 4 vs 2: deeper queue reduces GPU starvation
            drop_last=True,
            persistent_workers=pw,
            multiprocessing_context=mp_ctx,
            pin_memory=pin_memory,
            generator=train_gen,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=pw,
            multiprocessing_context=mp_ctx,
            pin_memory=pin_memory,
            generator=val_gen,
            worker_init_fn=worker_init_fn,
        )
        if inner_val_indices is not None:
            inner_val_gen = make_generator(seed + 2)
            inner_val_loader = DataLoader(
                torch.utils.data.Subset(dataset, inner_val_indices),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=pw,
                multiprocessing_context=mp_ctx,
                pin_memory=pin_memory,
                generator=inner_val_gen,
                worker_init_fn=worker_init_fn,
            )
            metadata["inner_val_loader"] = inner_val_loader
            metadata["num_inner_val"] = len(inner_val_indices)
    else:
        # Legacy: cell-level random split
        if lengths is None:
            lengths = [0.8, 0.2]
        random_generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            dataset, lengths, generator=random_generator
        )

        if max_val_samples is not None and max_val_samples < len(val_subset):
            rng = np.random.default_rng(seed)
            sub_indices = rng.choice(
                len(val_subset), size=max_val_samples, replace=False
            ).tolist()
            val_subset = torch.utils.data.Subset(val_subset, sub_indices)

        train_dataset = AugmentedDataset(
            train_subset, train_transform, dropout_transform
        )

        from deepcell_types.training.utils import make_generator, worker_init_fn

        train_gen = make_generator(seed)
        val_gen = make_generator(seed + 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            persistent_workers=pw,
            pin_memory=pin_memory,
            generator=train_gen,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=pw,
            generator=val_gen,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
        )

    metadata["num_train"] = len(train_subset) if hasattr(train_subset, "__len__") else 0
    metadata["num_val"] = len(val_subset) if hasattr(val_subset, "__len__") else 0

    return train_loader, val_loader, metadata


@dataclass
class DataLoaderConfig:
    """Bundle the 20+ knobs ``create_dataloader`` accepts into a single object.

    Use ``create_dataloader_from_config(zarr_dir, dct_config, cfg)`` when a
    caller has many parameters to set — it's more readable than 20+ keyword
    arguments at the call site, and it gives the IDE / type checker a
    discoverable home for new options.

    Field defaults exactly mirror ``create_dataloader``'s defaults; passing a
    bare ``DataLoaderConfig()`` is equivalent to calling ``create_dataloader``
    with no overrides.
    """

    skip_datasets: Optional[List[str]] = None
    keep_datasets: Optional[List[str]] = None
    batch_size: int = 256
    num_dropout_channels: int = 8
    num_workers: int = 16
    only_test: bool = False
    keep_fovs: Optional[List[str]] = None
    lengths: Optional[List[float]] = None
    use_fov_splits: bool = True
    train_ratio: float = 0.8
    seed: int = 42
    use_weighted_sampler: bool = True
    split_file: Optional[str] = None
    skip_distance_transform: bool = False
    persistent_workers: bool = False
    max_samples_per_epoch: Optional[int] = None
    max_val_samples: Optional[int] = None
    multiprocessing_context: Optional[Any] = None
    pin_memory: bool = False
    numpy_cache_max_bytes: Optional[int] = None


def create_dataloader_from_config(zarr_dir, dct_config, config: DataLoaderConfig):
    """Dataclass-based wrapper around :func:`create_dataloader`.

    Identical behaviour; the keyword forms exist side-by-side so existing
    callers do not need to be touched. New code is encouraged to use this
    entry point — the dataclass makes the 20+ knobs greppable and
    refactor-safe.
    """
    return create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        **{f.name: getattr(config, f.name) for f in fields(config)},
    )
