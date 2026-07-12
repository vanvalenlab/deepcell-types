"""Core full-image dataset for cell-type training.

This module hosts ``FullImageDataset`` (the per-cell zarr-backed dataset with
factored patch representation and per-worker FOV caching) and its archive
fingerprint helper. The augmentation transforms, FOV samplers, split/
stratification logic, and dataloader construction were factored out into the
sibling modules ``transforms``, ``samplers``, ``splits``, and ``dataloader``.

For backward compatibility, every public symbol that historically lived here is
re-exported below, so existing imports such as
``from deepcell_types.training.dataset import create_dataloader`` (and the
``CellIndexRecord``, ``AugmentedDataset``, ``FOVGroupedSampler``,
``create_fov_splits``, etc. names) continue to resolve unchanged.
"""

import json
import logging
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .annotations import (
    extract_cell_annotations,
    lookup_centroid,
)
from .archive import cached_archive_metadata_fingerprint
from .patch import extract_patch
from .splits import CellIndexRecord

logger = logging.getLogger(__name__)


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
        numpy_cache_max_bytes=None,
        crop_size=None,
        output_size=None,
        mask_intensities=True,
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
            numpy_cache_max_bytes: Per-worker full-FOV numpy cache budget.
            crop_size: Patch extraction size. Defaults to ``dct_config.CROP_SIZE``
                (32). The faithful CellSighter baseline overrides this to 60.
            output_size: Final patch size after resize. Defaults to
                ``dct_config.OUTPUT_SIZE``. When ``crop_size == output_size`` no
                resize is done.
            mask_intensities: If True (default), feed single-cell input
                (``raw * self_mask``) — the canonical DCT/MAPS behavior. If
                False, feed the full crop including neighbor intensities
                (faithful CellSighter). See ``patch.extract_patch``.
        """
        super().__init__(**kwargs)
        self.skip_distance_transform = skip_distance_transform
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
        self.crop_size = crop_size if crop_size is not None else dct_config.CROP_SIZE
        self.output_size = (
            output_size if output_size is not None else dct_config.OUTPUT_SIZE
        )
        self.mask_intensities = mask_intensities
        self.transform = transform
        self.domain_mapping = dct_config.domain_mapping
        self.marker2idx = dct_config.marker2idx
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

        logger.info("Found %d datasets in tissuenet archive", len(dataset_keys))

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
                    # so the failure shows up in CI log capture with a
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
                logger.info("Saved cache to %s", cache_path)

        # Pre-fetch references
        domain_mapping = self.domain_mapping
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

            domain = domain_mapping.get(dataset_key)
            if domain is None:
                # Fallback: read modality directly from the zarr group attrs.
                # This lets FullImageDataset load FOVs from auxiliary archives
                # (e.g. the Pan-M Gold-Standard ingest) whose keys don't
                # appear in the config's training-archive ``domain_mapping``.
                try:
                    modality_attr = (
                        str(zf[dataset_key].attrs.get("modality", "")).upper().strip()
                    )
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
            lengths = (len(cell_types), len(cell_indices), len(centroids))
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"{dataset_key}: cell_type, cell_index, and centroid arrays "
                    f"must have equal lengths; got {lengths}."
                )

            tissue_attr = ""
            try:
                tissue_attr = (
                    str(zf[dataset_key].attrs.get("tissue", "")).lower().strip()
                )
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

                # Cell types are canonical in the archive, so the stored label
                # is already its standardized form; keep only types the model
                # has a class for.
                if ct_label not in ct2idx:
                    continue
                ct_label_standard = ct_label

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
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(cache_path, flags)
            with os.fdopen(fd, "rb") as f:
                st = os.fstat(f.fileno())
                if st.st_uid != os.getuid():
                    logger.warning(
                        "cell-data cache at %s not owned by current user "
                        "(uid=%d, expected %d) — rejecting and rebuilding",
                        cache_path,
                        st.st_uid,
                        os.getuid(),
                    )
                    return None
                if st.st_mode & 0o022:
                    logger.warning(
                        "cell-data cache at %s is group- or world-writable — "
                        "rejecting and rebuilding",
                        cache_path,
                    )
                    return None
                cached = pickle.load(f)
        except (
            OSError,
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
            TypeError,
            ValueError,
        ) as e:
            logger.warning(
                "cell-data cache at %s could not be loaded (%s), rebuilding...",
                cache_path,
                e,
            )
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
        the archive ingestion pipeline)."""
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
            mask_intensities=self.mask_intensities,
        )
        # raw_masked: (C, H, W), spatial_context: (3, H, W) = (self_mask,
        # neighbor_mask, distance_transform)

        # Tripwire: a labeled cell must have a non-empty self-mask. An all-zero
        # self-mask means `raw_masked` is a blank patch carrying a real cell-type
        # label — caused by a mask/centroid desync or a non-integer mask defeating
        # the `== cell_idx` test in extract_patch. The canonical archive has zero
        # such cells; fail loudly rather than silently train on garbage if a
        # future archive regresses.
        if float(spatial_context[0].sum()) == 0:
            raise ValueError(
                f"{dataset_name}/{fov_name} cell {cell_idx}: empty self-mask "
                "(no pixels match cell_idx in the crop) — the patch would be "
                "all-zero with a valid label. Check the mask dtype and the "
                "mask-vs-centroid alignment for this FOV."
            )

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


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports.
#
# These symbols were factored out of this module into focused siblings. They
# are re-imported here so that every name historically importable from
# ``deepcell_types.training.dataset`` keeps resolving from this module
# unchanged (``from deepcell_types.training.dataset import X`` and ``import *``
# both continue to work).
# ---------------------------------------------------------------------------
from .dataloader import (  # noqa: E402,F401  (re-exported for back-compat)
    create_dataloader,
)
from .samplers import (  # noqa: E402,F401  (re-exported for back-compat)
    FOVGroupedSampler,
    SequentialFOVGroupedSampler,
    compute_sample_weights,
    compute_sample_weights_equal,
    subsample_indices_per_class,
)
from .splits import (  # noqa: E402,F401  (re-exported for back-compat)
    _ADVISORY_SPLIT_METADATA_KEYS,
    _build_fov_strata,
    _find_sole_source_fovs,
    _format_fov_examples,
    _split_metadata_for_dataset,
    create_fov_splits,
    load_fov_splits,
    save_fov_splits,
)
from .transforms import (  # noqa: E402,F401  (re-exported for back-compat)
    AugmentedDataset,
    DropOutChannels,
    _Compose,
    _RandomHorizontalFlip,
    _RandomVerticalFlip,
)

__all__ = [
    # Core (defined here)
    "FullImageDataset",
    "CellIndexRecord",
    "_archive_fingerprint",
    # transforms
    "AugmentedDataset",
    "DropOutChannels",
    "_Compose",
    "_RandomHorizontalFlip",
    "_RandomVerticalFlip",
    # samplers
    "FOVGroupedSampler",
    "SequentialFOVGroupedSampler",
    "compute_sample_weights",
    "compute_sample_weights_equal",
    "subsample_indices_per_class",
    # splits
    "CellIndexRecord",
    "create_fov_splits",
    "save_fov_splits",
    "load_fov_splits",
    "_ADVISORY_SPLIT_METADATA_KEYS",
    "_find_sole_source_fovs",
    "_build_fov_strata",
    "_split_metadata_for_dataset",
    "_format_fov_examples",
    # dataloader
    "create_dataloader",
]
