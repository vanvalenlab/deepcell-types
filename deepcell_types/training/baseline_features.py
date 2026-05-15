"""Baseline classifier feature extraction and metric helpers.

Split out of ``training/utils.py`` so baseline pipelines can import these
helpers without pulling in the training-time IO/RNG surface. ``utils.py``
keeps re-exports at the bottom for backward compatibility with external
callers.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .metrics import adjust_conf_mat_hierarchy
from .utils import (
    _atomic_np_savez,
    _atomic_pickle_dump,
    _cache_metadata_mismatches,
    _feature_cache_metadata,
    _format_examples,
)

logger = logging.getLogger(__name__)


def _apply_missing_value(out: dict, missing_value: float) -> None:
    """Replace absent-marker slots in ``X_train`` / ``X_val`` with ``missing_value``.

    The feature matrices stored in ``out`` are 0-filled at absent-marker
    columns by ``_extract_all_dataset_features``. This helper substitutes
    the caller-supplied sentinel post-extraction (and post-cache-load) so
    that the cached matrices stay parameter-agnostic — the same cache
    serves a MAPS run (``missing_value=0.0``) and an XGBoost run
    (``missing_value=np.nan``).

    Substitution is a no-op when ``missing_value == 0.0`` (the matrix is
    already 0-filled at absent slots) or when block metadata is absent
    (legacy cache prior to ``cache_version=6``).
    """
    if missing_value == 0.0:
        return
    for split in ("train", "val"):
        X = out.get(f"X_{split}")
        block_sizes = out.get(f"{split}_block_sizes")
        block_absent = out.get(f"{split}_block_absent")
        if X is None or block_sizes is None or block_absent is None:
            continue
        if len(X) == 0 or len(block_sizes) == 0:
            continue
        block_sizes = np.asarray(block_sizes)
        block_absent = np.asarray(block_absent, dtype=bool)
        if block_sizes.sum() != len(X):
            logger.warning(
                "%s block sizes (%d) do not sum to feature row count (%d); "
                "skipping missing-value substitution",
                split, int(block_sizes.sum()), len(X),
            )
            continue
        # Ensure we own the array before in-place writes (cached arrays
        # may be read-only memory-mapped views).
        if not X.flags.writeable or X.dtype != np.float32:
            X = np.array(X, dtype=np.float32)
            out[f"X_{split}"] = X
        row = 0
        for n_cells, absent_markers in zip(block_sizes.tolist(), block_absent):
            if absent_markers.any():
                X[row:row + int(n_cells), absent_markers] = missing_value
            row += int(n_cells)


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
        present_markers = np.zeros(num_markers, dtype=bool)
        cell_counts = counts[cell_idxs]
        cell_counts_safe = np.where(cell_counts > 0, cell_counts, 1)
        for c, ch_name in enumerate(channel_names):
            global_idx = resolve_marker_idx(ch_name)
            if global_idx is None:
                continue
            present_markers[global_idx] = True
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
            # Bool mask over the global marker vocabulary indicating which
            # marker slots were populated by a real channel in this dataset.
            # Lets ``extract_features_from_zarr`` distinguish "absent marker"
            # from "marker present, mean intensity is 0.0" so that XGBoost
            # and other baselines can substitute their preferred
            # missing-value sentinel (e.g. NaN).
            "present_markers": present_markers,
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
    missing_value: float = 0.0,
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
        missing_value: Sentinel written into feature slots whose marker is
            **absent** from the source dataset (no real channel for that
            marker). Default ``0.0`` preserves the legacy 0-fill behavior
            relied on by MAPS / CellSighter. XGBoost should pass
            ``numpy.nan`` so absent markers route through XGBoost's
            ``missing=NaN`` default direction at every split, rather than
            being conflated with real channels whose mean intensity
            happens to be 0.0. Substitution is applied per-row using the
            present-marker mask recorded for each row's source dataset.

    Returns:
        dict with keys:
            X_train, y_train, train_dataset_names, train_fov_names, train_cell_indices
            X_val, y_val, val_dataset_names, val_fov_names, val_cell_indices
            metadata: dict with active_datasets, num_samples
    """
    import zarr

    # Determine which datasets to process. This happens before cache loading so
    # cache provenance includes the exact requested dataset set.
    zf = zarr.open_group(zarr_dir, mode="r")
    # Look up via the ``config`` module so that monkey-patches of
    # ``config._discover_fov_keys`` (used by tests/test_baseline_feature_splits.py)
    # take effect after the split-out of this function from utils.py.
    from deepcell_types.training import config as _config_module

    all_dataset_keys = _config_module._discover_fov_keys(zf)

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
                        _apply_missing_value(out, missing_value)
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

    # Extract features — use global cache if available.
    # Look up the helper via the ``utils`` module so that monkey-patches of
    # ``utils._extract_all_dataset_features`` (used by
    # tests/test_baseline_feature_splits.py) take effect after the split-out
    # of this function from utils.py.
    from deepcell_types.training import utils as _utils_module

    per_dataset = _utils_module._extract_all_dataset_features(
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
            # Per-block (row_count, absent_markers) lists for the
            # post-concat missing-value substitution. Tracked per block
            # rather than per row so we don't materialise an (N, M) bool
            # array for big splits. Persisted alongside ``X_*`` so that
            # the per-split cache load path can re-apply substitution
            # without re-extracting features from zarr.
            "block_sizes": [],
            "block_absent": [],
        },
        "val": {
            "features": [],
            "labels": [],
            "ds_names": [],
            "fov_names": [],
            "cell_indices": [],
            "cell_sizes": [],
            "block_sizes": [],
            "block_absent": [],
        },
    }

    num_markers = len(dct_config.marker2idx)
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
        # Pre-v6 entries lack ``present_markers``; treat every column as
        # present so the missing-value substitution is a no-op (matches
        # legacy 0-fill behavior).
        present_markers = data.get("present_markers")
        if present_markers is None:
            absent_markers = np.zeros(num_markers, dtype=bool)
        else:
            absent_markers = ~np.asarray(present_markers, dtype=bool)
        acc["block_sizes"].append(n_cells)
        acc["block_absent"].append(absent_markers)

    # Concatenate
    out = {"metadata": {"active_datasets": dataset_keys, "num_samples": 0}}
    for split in ("train", "val"):
        acc = results[split]
        if acc["features"]:
            X = np.concatenate(acc["features"], axis=0)
            y = np.concatenate(acc["labels"], axis=0)
            cell_sizes = np.concatenate(acc["cell_sizes"], axis=0)
            block_sizes = np.asarray(acc["block_sizes"], dtype=np.int64)
            block_absent = np.stack(acc["block_absent"], axis=0).astype(bool)
        else:
            X = np.zeros((0, num_markers), dtype=np.float32)
            y = np.zeros(0, dtype=np.int64)
            cell_sizes = np.zeros(0, dtype=np.float32)
            block_sizes = np.zeros(0, dtype=np.int64)
            block_absent = np.zeros((0, num_markers), dtype=bool)

        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        out[f"{split}_dataset_names"] = acc["ds_names"]
        out[f"{split}_fov_names"] = acc["fov_names"]
        out[f"{split}_cell_indices"] = acc["cell_indices"]
        out[f"{split}_cell_sizes"] = cell_sizes
        out[f"{split}_block_sizes"] = block_sizes
        out[f"{split}_block_absent"] = block_absent
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

    _apply_missing_value(out, missing_value)
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
