import json
import logging
import os
import pickle
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader, Sampler, random_split

from .archive import cached_archive_metadata_fingerprint
from .patch import extract_patch
from .annotations import (
    build_centroid_tree,
    centroid_to_cell_idx_fast,
    extract_cell_annotations,
    group_filesystem_path,
    lookup_centroid,
    read_v3_1d_array,
)

logger = logging.getLogger(__name__)

_ADVISORY_SPLIT_METADATA_KEYS = {
    "zarr_path",
    # min_channels filter is a no-op on the labeled v10 corpus; tolerate
    # mismatch between split-file metadata and runtime config so that
    # legacy splits generated with --min_channels=3 load cleanly under the
    # new default of --min_channels=0 (and vice versa).
    "min_channels",
}


class CellIndexRecord(NamedTuple):
    """One per-cell entry in ``FullImageDataset.indices``.

    Replaces a positional 8-tuple. Pickles compactly (NamedTuple is
    serialized as a regular tuple), so existing cell-data caches that
    stored raw 8-tuples still deserialize correctly — and code that
    treats this as a tuple (indexing by integer, unpacking by position)
    continues to work too. The named accessors prevent the
    "positional magic number" footgun called out by complexity H8.

    Field 5 (``fov_name``) and field 6 (``dataset_name``) both currently
    hold ``dataset_key`` because the v8 archive layout encodes the FOV
    path in the dataset key itself; the two attributes are kept distinct
    so that downstream code can later differentiate them without another
    rename.
    """

    ds_idx: int
    ct_label: str
    ct_label_standard: str
    domain: str
    cell_idx: int
    fov_name: str
    dataset_name: str
    centroid: Tuple[float, ...]


class _Compose:
    """Minimal tensor transform composition to avoid a hard torchvision import."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class _RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-1,))
        return x


class _RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-2,))
        return x


def _archive_fingerprint(zf):
    """Compute a short fingerprint of zarr archive metadata.

    The cell-data cache depends on nested annotations, centroids, scale factors,
    and channel metadata, so root attrs alone are insufficient after in-place
    archive repairs. Delegate to the shared metadata fingerprint helper.
    """
    return cached_archive_metadata_fingerprint(zf)


class FullImageDataset(Dataset):
    """Dataset with factored representation for CellTypeAnnotator.

    Per-item layout:
    - Returns (C, 1, H, W) raw*self_mask + (3, H, W) spatial context (self_mask, neighbor_mask, distance_transform)
    - Handles "?" marker positivity by returning a validity mask
    - No hardcoded per-dataset sample caps (use sqrt-frequency sampling instead)
    - Supports FOV-level splits via create_fov_splits()
    """

    def __init__(
        self,
        zarr_dir,
        dct_config,
        skip_datasets=None,
        keep_datasets=None,
        transform=None,
        skip_fovs=None,
        keep_fovs=None,
        skip_distance_transform=False,
        min_channels=0,
        numpy_cache_max_bytes=None,
        **kwargs,
    ):
        """
        Args:
            zarr_dir: Path to tissuenet zarr archive or directory of zarr files
            dct_config: TissueNetConfig instance
            skip_datasets: List of dataset names to skip
            keep_datasets: List of dataset names to keep
            transform: Optional spatial transform for data augmentation
            skip_fovs: Set of FOV names to skip (for FOV splits)
            keep_fovs: Set of FOV names to keep (for FOV splits)
            skip_distance_transform: If True, skip distance transform computation
                (zeros instead). Saves CPU time for models that don't use it.
            min_channels: Minimum number of model-visible marker channels required per dataset.
                Datasets with fewer are excluded. Default 0 (no filtering).
            numpy_cache_max_bytes: Per-worker full-FOV numpy cache budget.
        """
        super().__init__(**kwargs)
        self.skip_distance_transform = skip_distance_transform
        self.min_channels = min_channels
        self._zarr_path = None  # Set by _load_tissuenet_archive (string, picklable)
        self._zarr_root = None  # Lazily opened per-worker
        self.archive_fingerprint = None
        self._array_cache = {}  # Per-worker cache: ds_idx -> (raw_zarr, mask_zarr)
        self._numpy_cache = OrderedDict()  # LRU cache: ds_idx -> (raw_np, mask_np)
        self._numpy_cache_bytes = 0  # Current cache size in bytes
        self._numpy_cache_max_bytes = (
            2 * 1024**3 if numpy_cache_max_bytes is None else int(numpy_cache_max_bytes)
        )
        # Per-FOV mask of channels that are listed in channel_names but are
        # all-zero in raw across the entire FOV (acquisition-empty). Populated
        # lazily from the numpy cache so the model can mask these out instead
        # of feeding constant-zero patches with a misleading marker embedding.
        self._zero_channel_cache: dict = {}

        self.max_channels = dct_config.MAX_NUM_CHANNELS
        self.crop_size = dct_config.CROP_SIZE
        self.output_size = dct_config.OUTPUT_SIZE
        self.transform = transform
        self.ct_mapping = dct_config.celltype_mapping
        self.domain_mapping = dct_config.domain_mapping
        self.marker2idx = dct_config.marker2idx
        self._idx2marker = {v: k for k, v in self.marker2idx.items()}
        self.ct2idx = dct_config.ct2idx
        self.domain2idx = dct_config.domain2idx
        self.marker_positivity_labels = dct_config.marker_positivity_labels
        self.ct_counts = {}
        # Strict tissue lookup: every archive entry must declare a tissue
        # attr. Pan-M FOVs go through gold_metadata.resolve_gold_metadata
        # at inference time, not this loader.
        self._tissue2idx = getattr(dct_config, "tissue2idx", {})

        self.zarr_files = []
        self.indices = []

        if skip_datasets and keep_datasets:
            raise ValueError("Cannot specify both skip_datasets and keep_datasets")
        if skip_datasets is None:
            skip_datasets = []

        if skip_fovs is None:
            skip_fovs = set()
        else:
            skip_fovs = set(skip_fovs)
        if keep_fovs is None:
            keep_fovs = set()
        else:
            keep_fovs = set(keep_fovs)

        if skip_fovs and keep_fovs:
            raise ValueError("Cannot specify both skip_fovs and keep_fovs")

        zarr_dir = Path(zarr_dir)
        self._load_tissuenet_archive(
            zarr_dir, skip_datasets, keep_datasets, skip_fovs, keep_fovs
        )

        # Count cell types
        self.ct_counts = {}
        for idx in self.indices:
            self.ct_counts[idx.ct_label_standard] = (
                self.ct_counts.get(idx.ct_label_standard, 0) + 1
            )

        # Store metadata
        self.metadata = {
            "active_datasets": [ds["name"] for ds in self.zarr_files],
            "num_samples": len(self.indices),
            "num_datasets": len(self.zarr_files),
            "keep_fovs": list(keep_fovs) if keep_fovs else None,
            "skip_fovs": list(skip_fovs) if skip_fovs else None,
        }

    def _get_cache_path(self, zarr_path):
        """Get path for the cell data cache file."""
        return Path(zarr_path).parent / f".{Path(zarr_path).name}.celldata_cache.pkl"

    def _load_tissuenet_archive(
        self, zarr_path, skip_datasets, keep_datasets, skip_fovs, keep_fovs
    ):
        """Load from tissuenet-style single zarr archive.

        Uses a disk cache for cell data (centroids + cell types) to avoid
        re-parsing ~750MB of zarr.json attrs on every run. First run builds
        the cache (~50min), subsequent runs load from cache (~10s).
        """
        self._zarr_path = str(zarr_path)
        zf = zarr.open_group(zarr_path, mode="r")
        from deepcell_types.training.config import _discover_fov_keys

        all_dataset_keys = _discover_fov_keys(zf)

        if keep_datasets:
            dataset_keys = [k for k in all_dataset_keys if k in keep_datasets]
        else:
            dataset_keys = [k for k in all_dataset_keys if k not in skip_datasets]

        # Apply FOV filters early
        if keep_fovs:
            dataset_keys = [k for k in dataset_keys if k in keep_fovs]
        elif skip_fovs:
            dataset_keys = [k for k in dataset_keys if k not in skip_fovs]

        print(f"Found {len(dataset_keys)} datasets in tissuenet archive")

        using_filtered_cache_keys = bool(
            skip_datasets or keep_datasets or skip_fovs or keep_fovs
        )
        cache_dataset_keys = (
            dataset_keys if using_filtered_cache_keys else all_dataset_keys
        )

        # Try loading from cache (validated by archive fingerprint)
        cache_path = self._get_cache_path(zarr_path)
        fingerprint = _archive_fingerprint(zf)
        self.archive_fingerprint = fingerprint
        all_cell_data = self._load_cell_data_cache(
            cache_path, cache_dataset_keys, fingerprint
        )

        if all_cell_data is None:
            # Build cache from zarr (slow, ~50min for full archive)
            logger.info("Building cell data cache (first run only)...")
            all_cell_data = {}
            failed_keys: list[str] = []
            for i, key in enumerate(cache_dataset_keys):
                try:
                    ds = zf[key]
                    if "preprocessed" not in ds:
                        all_cell_data[key] = {"channel_names": [], "cell_data": None}
                        continue
                    preproc = ds["preprocessed"]
                    channel_names = list(preproc.attrs.get("channel_names", []))
                    if not channel_names:
                        all_cell_data[key] = {"channel_names": [], "cell_data": None}
                        continue
                    cell_data = self._get_cell_data_tissuenet(ds, key)
                    all_cell_data[key] = {
                        "channel_names": channel_names,
                        "cell_data": cell_data,
                    }
                except (
                    KeyError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                    json.JSONDecodeError,
                    zarr.errors.GroupNotFoundError,
                ):
                    # Narrowed from a bare ``except Exception``: schema/logic
                    # bugs (anything outside this set) should crash loudly.
                    # Routes through logging (was bare ``print("WARNING:...")``)
                    # so the failure shows up in CI/wandb log capture with a
                    # real traceback.
                    logger.warning(
                        "Failed to load dataset %s; recording as empty entry",
                        key,
                        exc_info=True,
                    )
                    failed_keys.append(key)
                    all_cell_data[key] = {"channel_names": [], "cell_data": None}
                    continue
                if (i + 1) % 500 == 0:
                    logger.info(
                        "  Processed %d/%d datasets...",
                        i + 1,
                        len(cache_dataset_keys),
                    )
            if failed_keys:
                logger.warning(
                    "cell-data cache build completed with %d failed dataset(s) "
                    "out of %d; failed entries are stored as empty and will "
                    "contribute no training cells (first 10 keys: %s)",
                    len(failed_keys),
                    len(cache_dataset_keys),
                    failed_keys[:10],
                )
                fail_rate = len(failed_keys) / max(1, len(cache_dataset_keys))
                if fail_rate > 0.01:
                    raise RuntimeError(
                        f"cell-data cache build dropped {len(failed_keys)} of "
                        f"{len(cache_dataset_keys)} datasets ({fail_rate:.1%}, "
                        "above the 1% safety threshold). This usually means a "
                        "schema regression. Inspect the warning tracebacks "
                        "above and either fix the archive or relax the "
                        "threshold deliberately."
                    )

            if using_filtered_cache_keys:
                logger.info(
                    "not saving filtered cell-data cache (%d/%d datasets) to shared path",
                    len(cache_dataset_keys),
                    len(all_dataset_keys),
                )
            else:
                self._save_cell_data_cache(cache_path, all_cell_data, fingerprint)
                print(f"Saved cache to {cache_path}")

        # Pre-fetch references
        domain_mapping = self.domain_mapping
        ct_mapping = self.ct_mapping
        ct2idx = self.ct2idx

        # Aggregate results for requested datasets
        for dataset_key in dataset_keys:
            if dataset_key not in all_cell_data:
                continue

            entry = all_cell_data[dataset_key]
            if not entry.get("cell_data"):
                continue
            channel_names = entry["channel_names"]

            if len(channel_names) > self.max_channels:
                raise ValueError(
                    f"{dataset_key}: {len(channel_names)} channels exceeds "
                    f"MAX_NUM_CHANNELS={self.max_channels}. Increase the model "
                    "cap or define an explicit channel truncation policy."
                )

            # Filter datasets with too few model-visible marker channels
            if self.min_channels > 0:
                num_real = sum(
                    1 for c in channel_names if self._resolve_channel_index(c)[0] != -1
                )
                if num_real < self.min_channels:
                    continue

            domain = domain_mapping.get(dataset_key)
            if domain is None:
                # Fallback: read modality directly from the zarr group attrs.
                # This lets FullImageDataset load FOVs from auxiliary archives
                # (e.g. the Pan-M Gold-Standard ingest produced by
                # scripts/ingest_gold_to_zarr.py) whose keys don't appear in
                # the config's training-archive ``domain_mapping``.
                try:
                    modality_attr = str(
                        zf[dataset_key].attrs.get("modality", "")
                    ).upper().strip()
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    zarr.errors.GroupNotFoundError,
                ):
                    logger.warning(
                        "Could not read modality attr for %s",
                        dataset_key,
                        exc_info=True,
                    )
                    modality_attr = ""
                if not modality_attr:
                    continue
                domain = modality_attr

            cell_types, cell_indices, centroids = entry["cell_data"]

            tissue_attr = ""
            try:
                tissue_attr = str(zf[dataset_key].attrs.get("tissue", "")).lower().strip()
            except (
                AttributeError,
                KeyError,
                TypeError,
                zarr.errors.GroupNotFoundError,
            ):
                logger.warning(
                    "Could not read tissue attr for %s",
                    dataset_key,
                    exc_info=True,
                )
                tissue_attr = ""
            self.zarr_files.append(
                {
                    "name": dataset_key,
                    "channel_names": channel_names,
                    "dataset_key": dataset_key,
                    "tissue": tissue_attr,
                    "modality": domain,
                }
            )
            ds_idx = len(self.zarr_files) - 1

            for ct_label, cell_idx, centroid in zip(
                cell_types, cell_indices, centroids
            ):
                ct_label = str(ct_label)
                cell_idx = int(cell_idx)

                if dataset_key in ct_mapping and ct_label in ct_mapping[dataset_key]:
                    ct_label_standard = ct_mapping[dataset_key][ct_label]
                else:
                    ct_label_standard = ct_label

                if ct_label_standard not in ct2idx:
                    continue

                self.indices.append(
                    CellIndexRecord(
                        ds_idx=ds_idx,
                        ct_label=ct_label,
                        ct_label_standard=ct_label_standard,
                        domain=domain,
                        cell_idx=cell_idx,
                        fov_name=dataset_key,
                        dataset_name=dataset_key,
                        centroid=tuple(centroid),
                    )
                )

    @staticmethod
    def _load_cell_data_cache(cache_path, expected_keys, fingerprint):
        """Load cell data cache from disk if valid.

        The cache is wrapped in a dict of shape
        ``{"fingerprint": <hex>, "data": {...}}``. A mismatching fingerprint
        (archive mutated) or an unexpected schema triggers a rebuild. A bare
        legacy dict has no archive fingerprint, so it is rejected.
        """
        if not cache_path.exists():
            return None
        # Reject any cache not owned by the current user before deserializing.
        # The path is predictable (`{zarr_parent}/.{zarr_name}.celldata_cache.pkl`),
        # so on a shared filesystem another user could replace it with a crafted
        # pickle that would execute arbitrary code at `pickle.load` time. Owner
        # check + world-writable rejection blocks that vector while leaving the
        # single-user workflow unchanged.
        try:
            st = cache_path.stat()
        except OSError as e:
            logger.warning("cell-data cache stat failed (%s), rebuilding...", e)
            return None
        if st.st_uid != os.getuid():
            logger.warning(
                "cell-data cache at %s not owned by current user (uid=%d, "
                "expected %d) — rejecting and rebuilding",
                cache_path,
                st.st_uid,
                os.getuid(),
            )
            return None
        if st.st_mode & 0o002:  # world-writable
            logger.warning(
                "cell-data cache at %s is world-writable — rejecting and rebuilding",
                cache_path,
            )
            return None
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            logger.warning("cell-data cache load failed (%s), rebuilding...", e)
            return None

        # New format: {"fingerprint": ..., "data": {...}}
        if isinstance(cached, dict) and "fingerprint" in cached and "data" in cached:
            if cached["fingerprint"] != fingerprint:
                logger.warning(
                    "cell-data cache invalidated: archive fingerprint changed, rebuilding"
                )
                return None
            data = cached["data"]
            if not isinstance(data, dict):
                return None
            missing = set(expected_keys) - set(data.keys())
            if missing:
                logger.warning(
                    "cell-data cache stale: %d new datasets not in cache, rebuilding",
                    len(missing),
                )
                return None
            logger.info(
                "Loaded cell data cache (%d datasets) from %s", len(data), cache_path
            )
            return data

        # Legacy format: bare dict of dataset_key -> entry. It has no archive
        # fingerprint, so accepting it after in-place repairs can silently reuse
        # stale labels/centroids.
        if isinstance(cached, dict):
            logger.warning(
                "cell-data cache lacks archive fingerprint metadata, rebuilding"
            )
            return None

        return None

    @staticmethod
    def _save_cell_data_cache(cache_path, cell_data, fingerprint):
        """Save cell data cache to disk with an archive fingerprint.

        Uses an atomic write (tempfile + fsync + rename) so a SIGTERM mid-write
        during the ~50min cache build cannot leave a half-written pickle that
        would later be misread as valid.
        """
        from deepcell_types.training.utils import _atomic_pickle_dump

        payload = {"fingerprint": fingerprint, "data": cell_data}
        try:
            _atomic_pickle_dump(
                payload, Path(cache_path), protocol=pickle.HIGHEST_PROTOCOL
            )
            # Tighten perms so a foreign user can't replace the cache between
            # writes and have the next reader pickle.load their pickle.
            try:
                os.chmod(cache_path, 0o600)
            except OSError:
                pass
        except (OSError, PermissionError, pickle.PicklingError) as e:
            logger.warning("failed to save cell-data cache to %s: %s", cache_path, e)

    @staticmethod
    def _lookup_centroid(centroids_raw, idx):
        """Look up centroid by cell index, handling both int and string keys.

        Zarr v3 serializes all attribute keys as strings (JSON), so centroid
        dicts have string keys like {"1": [r, c]} even though cell indices
        are integers. Try string key first (v3 format), then int key (v2).
        """
        return lookup_centroid(centroids_raw, idx)

    def _get_cell_data_tissuenet(self, ds, dataset_key):
        """Extract cell types, indices, and centroids from a tissuenet dataset.

        Uses annotation sources (standardized_source or caitlinb) as the primary
        path. Annotation values are either cell indices (ints) or centroids
        ([row, col] pairs in original image coordinates). For centroid values,
        applies scale_factor to convert to preprocessed coordinate space and
        uses KDTree nearest-neighbor matching.

        Falls back to cell_type_info arrays for the rare datasets (3) that have
        labels there but no annotations group.
        """
        preproc = ds["preprocessed"]
        return extract_cell_annotations(
            ds, dataset_key, preproc, include_centroids=True
        )

    @staticmethod
    def _build_centroid_tree(centroids_raw):
        """Build a KDTree from preprocessed centroids for fast lookup."""
        return build_centroid_tree(centroids_raw)

    @staticmethod
    def _array_nbytes(arr) -> int:
        """Return array byte size across numpy and zarr 3 alpha arrays."""
        nbytes = getattr(arr, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        return int(np.prod(arr.shape) * np.dtype(arr.dtype).itemsize)

    def _resolve_channel_index(self, ch_name):
        """Direct marker2idx lookup. Strict canonical contract — no
        runtime alias resolution and no case-insensitive fallback.
        Source-data variants must be canonicalized at ingestion (by
        hubmap-to-zarr/apply_canonicalization.py)."""
        return self.marker2idx.get(ch_name, -1), ch_name

    def _calculate_marker_positivity(self, dataset_name, ct_label, ch_names):
        """Calculate marker positivity with proper "?" handling.

        Returns:
            marker_positivity: (C,) float32 array of 0/1 values
            validity_mask: (C,) bool array, True = valid (compute loss), False = skip
        """
        n_channels = len(ch_names)

        if dataset_name not in self.marker_positivity_labels:
            # No labels for this dataset: skip marker positivity entirely
            return np.zeros(n_channels, dtype=np.float32), np.zeros(
                n_channels, dtype=bool
            )

        df = self.marker_positivity_labels[dataset_name]
        row_lookup = {str(k).lower(): k for k in df.index}
        col_lookup = {str(k).lower(): k for k in df.columns}
        marker_positivity = np.zeros(n_channels, dtype=np.float32)
        validity_mask = np.ones(n_channels, dtype=bool)

        for i, channel in enumerate(ch_names):
            try:
                row_key = (
                    ct_label
                    if ct_label in df.index
                    else row_lookup.get(str(ct_label).lower(), ct_label)
                )
                col_key = (
                    channel
                    if channel in df.columns
                    else col_lookup.get(str(channel).lower(), channel)
                )
                val = df.loc[row_key, col_key]
                if val == "?" or val == 0.5 or val == "0.5":
                    # Uncertain: mask out from loss
                    marker_positivity[i] = 0.0
                    validity_mask[i] = False
                else:
                    float_val = float(val)
                    marker_positivity[i] = 1.0 if float_val >= 0.5 else 0.0
                    validity_mask[i] = True
            except (KeyError, ValueError):
                marker_positivity[i] = 0.0
                validity_mask[i] = False

        return marker_positivity, validity_mask

    def _get_zarr_arrays(self, ds_idx, fov_name=None):
        """Get arrays for patch extraction, with per-worker LRU numpy caching.

        When FOV-grouped sampling is used, consecutive samples come from the
        same FOV. Caching the full FOV as numpy arrays converts 3-4ms zarr
        reads into ~0.01ms numpy slices (300x speedup). An LRU cache of
        _numpy_cache_maxsize FOVs keeps memory bounded.

        Falls back to zarr Array references when the cache is disabled
        (maxsize=0).
        """
        # Check numpy LRU cache first (fastest path)
        if ds_idx in self._numpy_cache:
            self._numpy_cache.move_to_end(ds_idx)
            return self._numpy_cache[ds_idx]

        # Get zarr Array reference (cached separately for metadata access)
        if ds_idx not in self._array_cache:
            ds_info = self.zarr_files[ds_idx]
            if self._zarr_root is None:
                self._zarr_root = zarr.open_group(self._zarr_path, mode="r")
            dataset_key = ds_info["dataset_key"]
            preproc = self._zarr_root[dataset_key]["preprocessed"]
            raw_zarr = preproc["raw"]
            mask_zarr = preproc["mask"]

            self._array_cache[ds_idx] = (raw_zarr, mask_zarr)

        raw_zarr, mask_zarr = self._array_cache[ds_idx]

        # Load full FOV into numpy and cache (if within memory budget)
        if self._numpy_cache_max_bytes > 0:
            entry_bytes = self._array_nbytes(raw_zarr) + self._array_nbytes(mask_zarr)
            # Skip numpy caching if a single FOV exceeds the budget — but
            # still load once to populate the zero-channel mask, since
            # otherwise large CODEX FOVs (~13 GB) silently lose zero-channel
            # masking and the model attends to constant-zero tokens with a
            # marker-embedding prior.
            if entry_bytes > self._numpy_cache_max_bytes:
                if ds_idx not in self._zero_channel_cache:
                    raw_np = raw_zarr[:]
                    self._zero_channel_cache[ds_idx] = (
                        raw_np.reshape(raw_np.shape[0], -1).max(axis=1) == 0
                    )
                return raw_zarr, mask_zarr
            # Evict oldest entries until there's room. Drop the matching
            # zero-channel cache entry too so the two caches stay aligned.
            while (
                self._numpy_cache_bytes + entry_bytes > self._numpy_cache_max_bytes
                and self._numpy_cache
            ):
                evicted_idx, (old_raw, old_mask) = self._numpy_cache.popitem(last=False)
                self._numpy_cache_bytes -= self._array_nbytes(
                    old_raw
                ) + self._array_nbytes(old_mask)
                self._zero_channel_cache.pop(evicted_idx, None)
            raw_np = raw_zarr[:]
            mask_np = mask_zarr[:]
            self._numpy_cache[ds_idx] = (raw_np, mask_np)
            self._numpy_cache_bytes += raw_np.nbytes + mask_np.nbytes
            # Compute FOV-wide zero-channel mask once per cache populate. Cheap
            # vs. the FOV load itself (one max-reduce across H,W per channel).
            self._zero_channel_cache[ds_idx] = (
                raw_np.reshape(raw_np.shape[0], -1).max(axis=1) == 0
            )
            return raw_np, mask_np

        return raw_zarr, mask_zarr

    def __getstate__(self):
        """Clear zarr/numpy caches before pickling (for spawn workers)."""
        state = self.__dict__.copy()
        state["_array_cache"] = {}
        state["_numpy_cache"] = OrderedDict()
        state["_numpy_cache_bytes"] = 0
        state["_zero_channel_cache"] = {}
        state["_zarr_root"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.indices[idx]
        ds_idx = record.ds_idx
        ct_label = record.ct_label
        ct_label_standard = record.ct_label_standard
        domain = record.domain
        cell_idx = record.cell_idx
        fov_name = record.fov_name
        dataset_name = record.dataset_name
        centroid = record.centroid

        ds_info = self.zarr_files[ds_idx]
        ch_names = ds_info["channel_names"]

        raw_zarr, mask_zarr = self._get_zarr_arrays(ds_idx, fov_name)

        # Factored representation
        raw_masked, spatial_context = extract_patch(
            raw_zarr,
            mask_zarr,
            centroid,
            cell_idx,
            self.crop_size,
            self.output_size,
            skip_distance_transform=self.skip_distance_transform,
        )
        # raw_masked: (C, H, W), spatial_context: (3, H, W)

        n_real_channels = len(ch_names)
        if n_real_channels > self.max_channels:
            raise ValueError(
                f"{dataset_name}: {n_real_channels} channels exceeds "
                f"MAX_NUM_CHANNELS={self.max_channels}. Increase the model "
                "cap or define an explicit channel truncation policy."
            )

        ch_idx = np.full(self.max_channels, -1, dtype=np.int64)
        resolved_ch_names = []
        for i, ch_name in enumerate(ch_names):
            idx, resolved_name = self._resolve_channel_index(ch_name)
            ch_idx[i] = idx
            resolved_ch_names.append(resolved_name)

        ct_idx = self.ct2idx[ct_label_standard]
        domain_idx = self.domain2idx[domain]

        # Pad raw_masked to (C_max, 1, H, W) — needed before T-cell subtype check
        H, W = raw_masked.shape[1], raw_masked.shape[2]
        sample = np.full((self.max_channels, 1, H, W), -1.0, dtype=np.float32)
        sample[:n_real_channels, 0, :, :] = raw_masked

        # Marker positivity with "?" handling — read directly from the
        # per-dataset zarr `marker_positivity` group (curated labels). Cells
        # in datasets without a curated group get all-False validity_mask,
        # so they are dropped from the MP loss in `train.py`.
        ds_meta = self.zarr_files[ds_idx]
        tissue_str = (ds_meta.get("tissue") or "").lower().strip() or None
        marker_positivity, validity_mask = self._calculate_marker_positivity(
            dataset_name, ct_label_standard, resolved_ch_names
        )
        if not tissue_str:
            raise ValueError(
                f"Dataset {dataset_name!r} has no tissue attr; "
                f"every archive entry must declare a tissue. Pan-M "
                f"gold-standard FOVs should be routed through "
                f"deepcell_types.training.gold_metadata.resolve_gold_metadata, "
                f"not loaded via FullImageDataset."
            )
        if tissue_str not in self._tissue2idx:
            raise ValueError(
                f"Dataset {dataset_name!r} has tissue={tissue_str!r} "
                f"which is not in the canonical tissue vocab "
                f"{sorted(self._tissue2idx)}."
            )
        tissue_idx = self._tissue2idx[tissue_str]

        # Pad marker positivity and validity mask to max_channels
        mp_padded = np.zeros(self.max_channels, dtype=np.float32)
        mp_padded[:n_real_channels] = marker_positivity
        vm_padded = np.zeros(self.max_channels, dtype=bool)
        vm_padded[:n_real_channels] = validity_mask

        # Attention mask (True = padding, unknown channel, or all-zero channel)
        attn_mask = np.ones(self.max_channels, dtype=bool)
        attn_mask[:n_real_channels] = ch_idx[:n_real_channels] == -1

        # Mask out channels that are all-zero across the entire FOV. Acquisition
        # may list a marker in channel_names but never collect signal for it on
        # this FOV (~3.4% of valid channels per MIBI/IMC FOV; up to 16-20% on
        # outliers). Without this, the transformer attends to constant-zero
        # tokens with a marker embedding, injecting misleading prior.
        fov_zero_mask = self._zero_channel_cache.get(ds_idx)
        if fov_zero_mask is not None:
            attn_mask[:n_real_channels] |= fov_zero_mask[:n_real_channels]

        # Zero out unknown channels (ch_idx=-1) and FOV-zero channels.
        # Unified mask covers both so sample/mp/vm get cleared consistently.
        clear_mask = ch_idx[:n_real_channels] == -1
        if fov_zero_mask is not None:
            clear_mask = clear_mask | fov_zero_mask[:n_real_channels]
        if clear_mask.any():
            clear_idx = np.where(clear_mask)[0]
            sample[clear_idx] = -1.0
            mp_padded[clear_idx] = 0
            vm_padded[clear_idx] = False

        # Convert to tensors
        sample = torch.as_tensor(sample)
        spatial_context = torch.as_tensor(spatial_context)
        ch_idx = torch.as_tensor(ch_idx)
        attn_mask = torch.as_tensor(attn_mask)
        mp_padded = torch.as_tensor(mp_padded)
        vm_padded = torch.as_tensor(vm_padded)

        if self.transform:
            # Apply spatial transforms to both sample and spatial_context
            # Stack for consistent transform
            combined = torch.cat(
                [
                    sample.view(self.max_channels, H, W),  # (C_max, H, W)
                    spatial_context,  # (3, H, W)
                ],
                dim=0,
            )  # (C_max+3, H, W)
            combined = self.transform(combined)
            sample = combined[: self.max_channels].unsqueeze(1)  # (C_max, 1, H, W)
            spatial_context = combined[self.max_channels :]  # (3, H, W)

        return (
            sample,
            spatial_context,
            ch_idx,
            attn_mask,
            ct_idx,
            domain_idx,
            mp_padded,
            vm_padded,
            cell_idx,
            dataset_name,
            str(fov_name),
            torch.as_tensor(int(tissue_idx), dtype=torch.long),
        )


class AugmentedDataset(Dataset):
    """Wraps a dataset (or Subset) with augmentation transforms."""

    def __init__(self, dataset, transform=None, dropout_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.dropout_transform = dropout_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item) != 12:
            raise ValueError(
                f"AugmentedDataset expects a 12-tuple from the wrapped "
                f"dataset (sample, spatial_context, ch_idx, attn_mask, "
                f"ct_idx, domain_idx, mp, mp_mask, cell_index, "
                f"dataset_name, fov_name, tissue_idx), "
                f"got {len(item)}-tuple."
            )
        (
            sample,
            spatial_context,
            ch_idx,
            attn_mask,
            ct_idx,
            domain_idx,
            mp,
            mp_mask,
            cell_index,
            dataset_name,
            fov_name,
            tissue_idx,
        ) = item

        if self.transform:
            # Apply spatial transform consistently
            C_max = sample.shape[0]
            H, W = sample.shape[2], sample.shape[3]
            combined = torch.cat(
                [
                    sample.view(C_max, H, W),
                    spatial_context,
                ],
                dim=0,
            )
            combined = self.transform(combined)
            sample = combined[:C_max].unsqueeze(1)
            spatial_context = combined[C_max:]

        if self.dropout_transform:
            sample, ch_idx, attn_mask, mp, mp_mask = self.dropout_transform(
                sample, ch_idx, attn_mask, mp, mp_mask
            )

        return (
            sample,
            spatial_context,
            ch_idx,
            attn_mask,
            ct_idx,
            domain_idx,
            mp,
            mp_mask,
            cell_index,
            dataset_name,
            fov_name,
            tissue_idx,
        )


class DropOutChannels:
    """Drop random VALID channels (not padding) for regularization."""

    def __init__(self, n=3):
        self.n = n

    def __call__(self, sample, ch_idx, mask, marker_positivity, mp_mask):
        # Find valid (non-padded) channel indices
        valid_indices = torch.where(~mask)[0]
        n_valid = len(valid_indices)

        # Proportional dropout: drop at most 30% of valid channels, capped at
        # self.n. Very small panels are already information-poor; do not drop
        # from them.
        if n_valid <= 3:
            return sample, ch_idx, mask, marker_positivity, mp_mask

        n_drop = min(self.n, int(n_valid * 0.3))

        if n_drop <= 0 or n_valid <= n_drop:
            return sample, ch_idx, mask, marker_positivity, mp_mask

        # Sample from valid channels only
        drop_positions = valid_indices[torch.randperm(n_valid)[:n_drop]]

        sample[drop_positions] = -1.0
        ch_idx[drop_positions] = -1
        mask[drop_positions] = True
        marker_positivity[drop_positions] = 0
        mp_mask[drop_positions] = False

        return sample, ch_idx, mask, marker_positivity, mp_mask


def _find_sole_source_fovs(dataset, fov_to_indices):
    """Find FOVs that are the sole source of a cell type class.

    These FOVs must go to train so the model can learn rare classes.
    Without this, single-FOV classes randomly land in val, creating
    classes with 0 train support (guaranteed 0% accuracy on those cells).

    Returns:
        set of fov_keys that must be in train
    """
    # Map: class -> set of FOV keys containing it
    class_to_fovs = defaultdict(set)
    for fov_key, indices in fov_to_indices.items():
        fov_classes = set()
        for idx in indices:
            ct_label = dataset.indices[idx].ct_label_standard
            fov_classes.add(ct_label)
        for ct in fov_classes:
            class_to_fovs[ct].add(fov_key)

    # FOVs that are the only source of a class
    forced_train = set()
    for ct, fovs in class_to_fovs.items():
        if len(fovs) == 1:
            forced_train.update(fovs)

    if forced_train:
        forced_classes = [ct for ct, fovs in class_to_fovs.items() if len(fovs) == 1]
        print(
            f"Rare-class stratification: {len(forced_train)} FOVs forced to train "
            f"(sole source of {len(forced_classes)} classes: {sorted(forced_classes)})"
        )

    return forced_train


def _build_fov_strata(dataset, fov_to_indices, stratify_by):
    """For each fov_key, compute its stratum tuple from `stratify_by` keys.

    A stratum is the (modality, tissue) bucket a FOV belongs to. Callers
    force single-FOV strata to train (cannot evaluate a held-out FOV from a
    bucket that has only one) and split multi-FOV strata at train_ratio.

    Returns:
        dict mapping fov_key -> stratum tuple (e.g. ("mibi", "lymphnode"))
    """
    fov_to_stratum = {}
    for fov_key, idxs in fov_to_indices.items():
        sample_i = idxs[0]
        record = dataset.indices[sample_i]
        ds_idx = record.ds_idx
        modality = record.domain
        zf_entry = dataset.zarr_files[ds_idx]
        tissue = zf_entry.get("tissue", "")
        parts = []
        for key in stratify_by:
            if key == "modality":
                parts.append(modality)
            elif key == "tissue":
                parts.append(tissue)
            else:
                raise ValueError(f"Unsupported stratify_by key: {key}")
        fov_to_stratum[fov_key] = tuple(parts)
    return fov_to_stratum


def create_fov_splits(dataset, train_ratio=0.8, seed=42, stratify_by=()):
    """Split dataset by FOV (no spatial leakage).

    Groups cells by FOV, then assigns entire FOVs to train or val.
    FOVs that are the sole source of a cell type class are forced into
    train to prevent classes with 0 train support.

    When ``stratify_by`` is non-empty (e.g. ``("modality", "tissue")``), the
    remaining FOVs are bucketed by that key tuple and the train/val split is
    applied within each bucket. Single-FOV buckets are forced to train, since
    a single FOV cannot support both train and val.

    Args:
        dataset: FullImageDataset instance
        train_ratio: Fraction of FOVs for training
        seed: Random seed
        stratify_by: Tuple of stratification keys, e.g. ("modality", "tissue").
            Empty tuple disables stratification (legacy global shuffle).

    Returns:
        train_indices: List of integer indices into dataset
        val_indices: List of integer indices into dataset
    """
    rng = random.Random(seed)

    # Group indices by (dataset_name, fov_name)
    fov_to_indices = defaultdict(list)
    for i, idx_tuple in enumerate(dataset.indices):
        fov_key = (idx_tuple[6], idx_tuple[5])  # (dataset_name, fov_name)
        fov_to_indices[fov_key].append(i)

    # Force sole-source FOVs into train
    forced_train_fovs = _find_sole_source_fovs(dataset, fov_to_indices)

    train_indices = []
    val_indices = []
    for fov_key in forced_train_fovs:
        train_indices.extend(fov_to_indices[fov_key])

    remaining_fov_keys = sorted(
        k for k in fov_to_indices.keys() if k not in forced_train_fovs
    )

    if stratify_by:
        # Per-stratum split: ensures (modality, tissue) buckets with ≥2 FOVs
        # have both train and val coverage. Single-FOV buckets go to train.
        fov_to_stratum = _build_fov_strata(dataset, fov_to_indices, stratify_by)
        by_stratum = defaultdict(list)
        for fov_key in remaining_fov_keys:
            by_stratum[fov_to_stratum[fov_key]].append(fov_key)
        single_fov_strata = []
        for stratum in sorted(by_stratum.keys()):
            keys = list(by_stratum[stratum])
            rng.shuffle(keys)
            if len(keys) == 1:
                single_fov_strata.append(stratum)
                train_indices.extend(fov_to_indices[keys[0]])
                continue
            # Round to nearest, clamp so neither side is empty
            n_train = max(1, min(len(keys) - 1, int(round(len(keys) * train_ratio))))
            for fk in keys[:n_train]:
                train_indices.extend(fov_to_indices[fk])
            for fk in keys[n_train:]:
                val_indices.extend(fov_to_indices[fk])
        if single_fov_strata:
            logger.info(
                "stratified split: %d single-FOV strata forced to train (cannot eval): %s",
                len(single_fov_strata), single_fov_strata,
            )
        return train_indices, val_indices

    # Legacy non-stratified path (global random shuffle)
    rng.shuffle(remaining_fov_keys)
    target_train = int(len(fov_to_indices) * train_ratio)
    requested_remaining = target_train - len(forced_train_fovs)
    n_train_remaining = max(0, requested_remaining)
    if requested_remaining < 0:
        logger.warning(
            "rare-class forcing clamp: %d FOVs forced to train exceeds target %d "
            "(requested %d remaining, satisfied 0); train pool saturated by sole-source FOVs",
            len(forced_train_fovs),
            target_train,
            requested_remaining,
        )

    for fov_key in remaining_fov_keys[:n_train_remaining]:
        train_indices.extend(fov_to_indices[fov_key])
    for fov_key in remaining_fov_keys[n_train_remaining:]:
        val_indices.extend(fov_to_indices[fov_key])

    return train_indices, val_indices


def _split_metadata_for_dataset(dataset):
    """Return provenance fields that make split reuse auditable."""
    marker2idx = getattr(dataset, "marker2idx", {})
    ct2idx = getattr(dataset, "ct2idx", {})
    zarr_path = getattr(dataset, "_zarr_path", None)

    return {
        "min_channels": getattr(dataset, "min_channels", None),
        "max_channels": getattr(dataset, "max_channels", None),
        "num_marker_channels": len(marker2idx) if marker2idx is not None else None,
        "num_cell_types": len(ct2idx) if ct2idx is not None else None,
        # Kept for auditability only. Split files must remain portable across
        # mount points and symlinks, so load_fov_splits never treats this as
        # strict provenance.
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "archive_fingerprint": getattr(dataset, "archive_fingerprint", None),
    }


def _format_fov_examples(fov_keys, limit=5):
    examples = sorted(fov_keys)[:limit]
    suffix = "" if len(fov_keys) <= limit else f", ... (+{len(fov_keys) - limit} more)"
    return ", ".join(f"{ds}/{fov}" for ds, fov in examples) + suffix


def save_fov_splits(dataset, split_file, train_ratio=0.8, seed=42, stratify_by=()):
    """Generate FOV splits and save to a JSON file for reproducibility.

    Delegates to ``create_fov_splits`` for the actual partitioning logic
    (rare-class sole-source forcing, optional stratification, train/val
    bucket assignment) and adds JSON serialization with metadata.

    Args:
        dataset: FullImageDataset instance.
        split_file: Path to write the JSON split file.
        train_ratio: Fraction of FOVs for training.
        seed: Random seed.
        stratify_by: Tuple of stratification keys, e.g. ``("modality", "tissue")``.
            Empty tuple disables stratification (legacy global shuffle, used by
            v9 splits for benchmark continuity).

    Returns:
        train_indices, val_indices (same as ``create_fov_splits``).
    """
    train_indices, val_indices = create_fov_splits(
        dataset, train_ratio=train_ratio, seed=seed, stratify_by=stratify_by,
    )

    # Reconstruct (dataset_name -> [fov_names]) groupings from the per-cell
    # index lists. Each integer in train_indices/val_indices points at a row
    # of dataset.indices, whose fields 6 and 5 are dataset_name and fov_name.
    train_split: dict = {}
    val_split: dict = {}
    train_fov_keys: set = set()
    for i in train_indices:
        record = dataset.indices[i]
        fov_key = (record.dataset_name, record.fov_name)
        if fov_key in train_fov_keys:
            continue
        train_fov_keys.add(fov_key)
        train_split.setdefault(fov_key[0], []).append(fov_key[1])
    val_fov_keys: set = set()
    for i in val_indices:
        record = dataset.indices[i]
        fov_key = (record.dataset_name, record.fov_name)
        if fov_key in val_fov_keys:
            continue
        val_fov_keys.add(fov_key)
        val_split.setdefault(fov_key[0], []).append(fov_key[1])

    # Re-derive single-FOV stratum count for metadata transparency.
    num_single_fov_strata = 0
    if stratify_by:
        fov_to_indices = defaultdict(list)
        for i, record in enumerate(dataset.indices):
            fov_to_indices[(record.dataset_name, record.fov_name)].append(i)
        forced = _find_sole_source_fovs(dataset, fov_to_indices)
        fov_to_stratum = _build_fov_strata(dataset, fov_to_indices, stratify_by)
        by_stratum: dict = defaultdict(list)
        for fov_key in fov_to_indices:
            if fov_key in forced:
                continue
            by_stratum[fov_to_stratum[fov_key]].append(fov_key)
        num_single_fov_strata = sum(1 for v in by_stratum.values() if len(v) == 1)

    split_data = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "stratify_by": list(stratify_by),
            "num_train_fovs": sum(len(v) for v in train_split.values()),
            "num_val_fovs": sum(len(v) for v in val_split.values()),
            "num_datasets": len(set(list(train_split) + list(val_split))),
            "num_single_fov_strata_forced_to_train": num_single_fov_strata,
            "created": datetime.now(timezone.utc).isoformat(),
            **_split_metadata_for_dataset(dataset),
        },
        "train": train_split,
        "val": val_split,
    }

    split_path = Path(split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"FOV splits saved to {split_path}")
    return train_indices, val_indices


def load_fov_splits(dataset, split_file, *, strict=True):
    """Load pre-computed FOV splits from a JSON file.

    Args:
        dataset: FullImageDataset instance.
        split_file: Path to the JSON split file.
        strict: If True (default), raise ValueError on overlap between train/
            val/heldout, FOVs in the JSON missing from the live archive, or
            FOVs in the live archive missing from the JSON. If False, log a
            warning and continue with whatever overlap is resolvable.

    Returns:
        train_indices: List of integer indices into dataset.
        val_indices: List of integer indices into dataset.
    """
    with open(split_file) as f:
        split_data = json.load(f)

    train_fovs_by_ds = split_data["train"]
    val_fovs_by_ds = split_data["val"]

    # Build lookup: (dataset_name, fov_name) -> set membership
    train_fov_set = set()
    for ds_name, fov_names in train_fovs_by_ds.items():
        for fov_name in fov_names:
            train_fov_set.add((ds_name, fov_name))

    val_fov_set = set()
    for ds_name, fov_names in val_fovs_by_ds.items():
        for fov_name in fov_names:
            val_fov_set.add((ds_name, fov_name))

    heldout_fov_set = set()
    for ds_name, fov_names in split_data.get("heldout", {}).items():
        for fov_name in fov_names:
            heldout_fov_set.add((ds_name, fov_name))

    train_indices = []
    val_indices = []
    skipped = 0
    heldout = 0
    dataset_fov_set = set()

    for i, idx_tuple in enumerate(dataset.indices):
        fov_key = (idx_tuple[6], idx_tuple[5])  # (dataset_name, fov_name)
        dataset_fov_set.add(fov_key)
        if fov_key in train_fov_set:
            train_indices.append(i)
        elif fov_key in val_fov_set:
            val_indices.append(i)
        elif fov_key in heldout_fov_set:
            heldout += 1
        else:
            skipped += 1

    if heldout > 0:
        logger.info("%d samples intentionally held out by split file", heldout)

    if skipped > 0:
        msg = f"{skipped} samples not found in split file (dataset/FOV mismatch)"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    train_val_overlap = train_fov_set & val_fov_set
    if train_val_overlap:
        msg = (
            f"{len(train_val_overlap)} FOVs appear in both train and val splits: "
            f"{_format_fov_examples(train_val_overlap)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    heldout_overlap = heldout_fov_set & (train_fov_set | val_fov_set)
    if heldout_overlap:
        msg = (
            f"{len(heldout_overlap)} heldout FOVs also appear in train/val: "
            f"{_format_fov_examples(heldout_overlap)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    missing_split_fovs = (train_fov_set | val_fov_set) - dataset_fov_set
    if missing_split_fovs:
        msg = (
            f"{len(missing_split_fovs)} split FOVs are not present in the current "
            f"dataset after filters: {_format_fov_examples(missing_split_fovs)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    missing_heldout_fovs = heldout_fov_set - dataset_fov_set
    if missing_heldout_fovs:
        msg = (
            f"{len(missing_heldout_fovs)} heldout FOVs are not present in the "
            f"current dataset after filters: {_format_fov_examples(missing_heldout_fovs)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    meta = split_data.get("metadata", {})
    current_meta = _split_metadata_for_dataset(dataset)
    metadata_mismatches = []
    for key, current_value in current_meta.items():
        saved_value = meta.get(key)
        if current_value is None:
            continue
        if saved_value is None:
            msg = f"{key}: file is missing, current dataset has {current_value!r}"
            if key in _ADVISORY_SPLIT_METADATA_KEYS:
                logger.warning("split metadata missing advisory %s", msg)
            else:
                metadata_mismatches.append(msg)
        elif saved_value != current_value:
            msg = f"{key}: file has {saved_value!r}, current dataset has {current_value!r}"
            if key in _ADVISORY_SPLIT_METADATA_KEYS:
                logger.warning("split metadata advisory mismatch: %s", msg)
            else:
                metadata_mismatches.append(msg)

    if metadata_mismatches:
        msg = "split metadata mismatch: " + "; ".join(metadata_mismatches)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    print(
        f"Loaded FOV splits from {split_file} "
        f"(created {meta.get('created', 'unknown')}): "
        f"{len(train_indices)} train, {len(val_indices)} val"
    )

    return train_indices, val_indices


def compute_sample_weights(dataset, indices):
    """Compute sqrt-inverse-frequency sample weights for WeightedRandomSampler.

    Args:
        dataset: FullImageDataset instance
        indices: List of indices to compute weights for

    Returns:
        weights: torch.Tensor of per-sample weights
    """
    # Count cell types in the given indices
    ct_counts = defaultdict(int)
    for i in indices:
        ct_label = dataset.indices[i].ct_label_standard
        ct_counts[ct_label] += 1

    # Compute sqrt-inverse-frequency weights with a minimum effective count cap.
    # Treat any class as if it has at least 1000 samples for weighting purposes,
    # preventing rare single-FOV classes (e.g. Myofibroblast with 236 cells)
    # from receiving extreme weights that corrupt representations for common classes.
    total = sum(ct_counts.values())
    ct_weights = {}
    for ct, count in ct_counts.items():
        effective_count = max(count, 1000)
        ct_weights[ct] = np.sqrt(total / effective_count)

    # Assign per-sample weight
    weights = torch.zeros(len(indices))
    for i, idx in enumerate(indices):
        ct_label = dataset.indices[idx].ct_label_standard
        weights[i] = ct_weights[ct_label]

    return weights


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

    def __init__(self, dataset_indices, train_indices, seed: int = 42):
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

        for ds_idx in ordered_ds:
            yield from groups[ds_idx]

    def __len__(self):
        return self._n


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
    min_channels=0,
    numpy_cache_max_bytes=None,
    fov_grouped_train: bool = False,
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

    Returns:
        train_loader, val_loader, metadata dict
        (train_loader is None if only_test=True)
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
        min_channels=min_channels,
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
                dataset.indices, train_indices, seed=seed,
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
    min_channels: int = 0
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
