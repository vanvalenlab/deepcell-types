"""Shared annotation extraction for zarr-backed datasets."""

import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def lookup_centroid(centroids_raw, idx):
    """Look up centroid by cell index, handling zarr-v3 string keys."""
    str_key = str(idx)
    if str_key in centroids_raw:
        return centroids_raw[str_key]
    return centroids_raw.get(idx)


def build_centroid_tree(centroids_raw):
    """Build a KDTree from preprocessed centroids for fast lookup."""
    from scipy.spatial import cKDTree

    keys = list(centroids_raw.keys())
    coords = np.array([[centroids_raw[k][0], centroids_raw[k][1]] for k in keys])
    return cKDTree(coords), keys


def centroid_to_cell_idx_fast(tree, keys, target_centroid, tol=1.5):
    """Reverse-lookup cell index using KDTree nearest-neighbor search.

    The default tolerance is 1.5 px to absorb sub-pixel centroid drift between
    `standardized_source` (raw image coords) and preprocessed centroids; some
    ingestion pipelines (notably mcmicro_TMA11) produce drift of 1.03–1.4 px
    after `scale_factor` rescaling, which would be lost at tol=1.0.
    """
    target = np.array([float(target_centroid[0]), float(target_centroid[1])])
    dist, idx = tree.query(target)
    if dist < tol:
        return int(keys[idx])
    return None


def group_filesystem_path(group):
    """Return the local filesystem path for a zarr group, when available."""
    store_path = getattr(group, "store_path", None)
    store = getattr(store_path, "store", None)
    root = getattr(store, "root", None)
    if root is None:
        return None
    path = getattr(store_path, "path", "")
    return Path(root) / path if path else Path(root)


def read_v3_1d_array(array_dir: Path):
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

    # Read the zstd level from codec config (NOT hardcoded 0). Mirrors the
    # inference-side read in deepcell_types/config.py — keeping these in sync
    # prevents a latent correctness bug if archives are ever written at a
    # different compression level.
    zstd = None
    for codec in meta.get("codecs", []):
        if codec.get("name") == "zstd":
            from numcodecs import Zstd

            config = codec.get("configuration", {})
            zstd = Zstd(level=config.get("level", 0))
            break

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


def _add_label(records, idx, ct_name, centroid=None):
    if idx not in records:
        records[idx] = {"labels": [], "centroid": centroid}
    records[idx]["labels"].append(str(ct_name))
    if records[idx]["centroid"] is None and centroid is not None:
        records[idx]["centroid"] = centroid


def _finalize_records(records, dataset_key, include_centroids):
    cell_types, cell_indices, centroids = [], [], []
    duplicate_same_label = 0
    conflicts = []

    for idx, record in records.items():
        labels = record["labels"]
        unique = set(labels)
        if len(unique) > 1:
            conflicts.append((idx, labels))
            continue
        if len(labels) > 1:
            duplicate_same_label += len(labels) - 1
        cell_types.append(labels[0])
        cell_indices.append(int(idx))
        if include_centroids:
            centroids.append(record["centroid"])

    if duplicate_same_label:
        logger.info(
            "%s: collapsed %d duplicate agreeing annotation entries",
            dataset_key,
            duplicate_same_label,
        )
    if conflicts:
        logger.warning(
            "%s: dropped %d cells with conflicting duplicate labels; examples=%s",
            dataset_key,
            len(conflicts),
            conflicts[:3],
        )

    if not cell_types:
        return None
    if include_centroids:
        return cell_types, cell_indices, centroids
    return cell_types, cell_indices


def extract_cell_annotations(ds, dataset_key, preproc, include_centroids=False):
    """Extract canonical cell labels/indices from a zarr FOV.

    Duplicate annotations for the same resolved cell are collapsed when all
    labels agree. If labels conflict, the cell is dropped as ambiguous.
    """
    records = OrderedDict()
    centroids_raw = dict(preproc.attrs.get("centroids", {}))

    if "cell_types" in ds and "annotations" in ds["cell_types"]:
        annotations_attrs = dict(ds["cell_types/annotations"].attrs)
        source = annotations_attrs.get("standardized_source")
        if source is None:
            source = annotations_attrs.get("caitlinb")

        scale_factor = preproc.attrs.get("scale_factor", 1.0)
        if source:
            centroid_tree, centroid_keys = None, None
            centroid_attempts = 0
            centroid_drops = 0

            for ct_name, values in source.items():
                if ct_name is None or ct_name == "null":
                    continue
                for val in values:
                    if isinstance(val, (int, float)) and not isinstance(val, list):
                        idx = int(val)
                        cent = lookup_centroid(centroids_raw, idx)
                        if cent is not None:
                            _add_label(records, idx, ct_name, cent)
                    elif isinstance(val, (list, tuple)) and len(val) == 2:
                        if centroid_tree is None:
                            if not centroids_raw:
                                continue
                            centroid_tree, centroid_keys = build_centroid_tree(
                                centroids_raw
                            )
                        centroid_attempts += 1
                        scaled = [
                            float(val[0]) * scale_factor,
                            float(val[1]) * scale_factor,
                        ]
                        idx = centroid_to_cell_idx_fast(
                            centroid_tree, centroid_keys, scaled
                        )
                        if idx is None:
                            centroid_drops += 1
                            continue
                        cent = lookup_centroid(centroids_raw, idx)
                        if cent is not None:
                            _add_label(records, idx, ct_name, cent)

            if centroid_attempts > 0:
                drop_pct = 100.0 * centroid_drops / centroid_attempts
                if drop_pct > 5.0:
                    logger.warning(
                        "centroid match drop for %s: %d/%d (%.1f%%) annotations dropped (dist >= 1.5)",
                        dataset_key,
                        centroid_drops,
                        centroid_attempts,
                        drop_pct,
                    )

    if not records:
        preproc_dir = group_filesystem_path(preproc)
        if preproc_dir is not None:
            info_dir = preproc_dir / "cell_type_info"
            ct_arr = read_v3_1d_array(info_dir / "cell_type")
            idx_arr = read_v3_1d_array(info_dir / "cell_index")
            if ct_arr is not None and idx_arr is not None:
                for ct, idx in zip(ct_arr, idx_arr):
                    idx = int(idx)
                    cent = lookup_centroid(centroids_raw, idx)
                    if cent is not None:
                        _add_label(records, idx, ct, cent)

    if not records:
        return None
    return _finalize_records(records, dataset_key, include_centroids)
