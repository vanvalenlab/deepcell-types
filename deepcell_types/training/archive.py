"""Zarr archive fingerprinting and FOV discovery helpers.

Split out of ``training/config.py`` so that loader code can depend on these
without pulling in the full ``TissueNetConfig`` class. ``config.py`` keeps a
re-export at the bottom for backward compatibility with external callers.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# NOTE: a ``_patch_zarr_v3_alpha_metadata`` monkeypatch previously lived here to
# let a zarr 3.0.0a* alpha parse the ``consolidated_metadata`` /
# ``storage_transformers`` metadata keys emitted by newer writers. The pinned
# ``zarr>=3.1`` (see pyproject) reads these natively, so the shim was removed.


def _local_zarr_root_path(zarr_obj_or_path) -> Optional[Path]:
    """Return a local filesystem root for a zarr object/path when available."""
    if isinstance(zarr_obj_or_path, (str, os.PathLike, Path)):
        return Path(zarr_obj_or_path)

    store_path = getattr(zarr_obj_or_path, "store_path", None)
    store = getattr(store_path, "store", None)
    root = getattr(store, "root", None)
    if root is None:
        return None
    path = getattr(store_path, "path", "")
    root_path = Path(root)
    return root_path / path if path else root_path


_FOV_KEYS_CACHE: Dict[str, List[str]] = {}
_FINGERPRINT_CACHE: Dict[str, str] = {}


def cached_archive_metadata_fingerprint(zarr_obj_or_path) -> str:
    """Per-process memoized wrapper around ``archive_metadata_fingerprint``.

    Production archives are immutable for the lifetime of a single process,
    and full-archive fingerprinting reads all chunk contents under
    ``cell_type_info`` (~8s on a 2.7k-FOV archive). Loaders that construct
    multiple datasets/predictors against the same archive should call this
    wrapper. Unit tests that exercise the mutate-and-rehash contract should
    keep calling ``archive_metadata_fingerprint`` directly.
    """
    root_path = _local_zarr_root_path(zarr_obj_or_path)
    if root_path is None:
        return archive_metadata_fingerprint(zarr_obj_or_path)
    key = str(root_path.resolve()) if root_path.exists() else str(root_path)
    cached = _FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return cached
    fp = archive_metadata_fingerprint(zarr_obj_or_path)
    _FINGERPRINT_CACHE[key] = fp
    return fp


def archive_metadata_fingerprint(zarr_obj_or_path) -> str:
    """Fingerprint archive metadata files for cache/split provenance.

    The cell-data and baseline-feature caches depend mostly on nested zarr
    metadata attrs (annotations, centroids, scale factors, channel names), not
    just root attrs. For local stores, hash every v3 ``zarr.json`` path plus
    size and mtime. This is intentionally cheaper than reading the large
    annotation JSON payloads on every startup, while still invalidating normal
    in-place repairs that touch nested metadata files.
    """
    import json

    h = hashlib.sha256()
    h.update(b"archive-metadata-fingerprint-v2")

    root_path = _local_zarr_root_path(zarr_obj_or_path)
    if root_path is not None and root_path.exists():
        # Single os.walk collects both lists; two separate ``**/`` globs each
        # cost ~10s on a 45k-file archive and dominate the fingerprint cost.
        meta_files: list[Path] = []
        cti_dirs: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames.sort()
            if "zarr.json" in filenames:
                meta_files.append(Path(dirpath) / "zarr.json")
            parts = dirpath.split(os.sep)
            if len(parts) >= 2 and parts[-1] == "cell_type_info" and parts[-2] == "preprocessed":
                cti_dirs.append(Path(dirpath))
        meta_files.sort()
        cti_dirs.sort()
        if meta_files:
            for path in meta_files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rel = path.relative_to(root_path).as_posix()
                h.update(rel.encode())
                h.update(str(stat.st_size).encode())
                h.update(str(stat.st_mtime_ns).encode())
            for info_dir in cti_dirs:
                for array_name in ("cell_type", "cell_index"):
                    _hash_zarr_array_files(
                        h,
                        info_dir / array_name,
                        root_path,
                        include_chunk_contents=True,
                    )
            return h.hexdigest()[:16]

    attrs = dict(getattr(zarr_obj_or_path, "attrs", {}))
    blob = json.dumps(attrs, sort_keys=True, default=str).encode()
    h.update(blob)
    return h.hexdigest()[:16]


def _hash_file_stat(h: "hashlib._Hash", path: Path, root_path: Path) -> None:
    """Add stable file stat metadata to an existing hash."""
    try:
        stat = path.stat()
    except OSError:
        return
    rel = path.relative_to(root_path).as_posix()
    h.update(rel.encode())
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())


def _hash_file_contents(h: "hashlib._Hash", path: Path, root_path: Path) -> None:
    """Add file path and contents to an existing hash."""
    try:
        rel = path.relative_to(root_path).as_posix()
        with open(path, "rb") as f:
            h.update(rel.encode())
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return


def _iter_files_sorted(root_path: Path) -> Iterable[Path]:
    """Yield files below root_path in deterministic order without glob overhead."""
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def _hash_zarr_array_files(
    h: "hashlib._Hash",
    array_path: Path,
    root_path: Path,
    *,
    include_chunk_contents: bool = False,
) -> None:
    """Add zarr array metadata and chunk file stats to an existing hash."""
    _hash_file_stat(h, array_path / "zarr.json", root_path)
    chunk_root = array_path / "c"
    if not chunk_root.exists():
        return
    for chunk_path in _iter_files_sorted(chunk_root):
        if include_chunk_contents:
            _hash_file_contents(h, chunk_path, root_path)
        else:
            _hash_file_stat(h, chunk_path, root_path)


def archive_array_fingerprint(
    zarr_obj_or_path, dataset_keys: Optional[Iterable[str]] = None
) -> str:
    """Fingerprint archive metadata plus preprocessed raw/mask chunk metadata.

    Baseline feature caches depend on raw intensity and mask chunks. Metadata-only
    fingerprints are insufficient after in-place chunk repairs that preserve
    zarr.json files, so this hashes path/size/mtime for the relevant
    ``preprocessed/raw`` and ``preprocessed/mask`` chunks.
    """
    h = hashlib.sha256()
    h.update(b"archive-array-fingerprint-v1")
    h.update(archive_metadata_fingerprint(zarr_obj_or_path).encode())

    root_path = _local_zarr_root_path(zarr_obj_or_path)
    if root_path is None or not root_path.exists():
        return h.hexdigest()[:16]

    if dataset_keys is None:
        preprocessed_paths = sorted(
            path.parent for path in root_path.glob("**/preprocessed/zarr.json")
        )
    else:
        preprocessed_paths = sorted(
            root_path / key / "preprocessed" for key in dataset_keys
        )

    for preprocessed_path in preprocessed_paths:
        if not preprocessed_path.exists():
            continue
        for array_name in ("raw", "mask"):
            _hash_zarr_array_files(h, preprocessed_path / array_name, root_path)

    return h.hexdigest()[:16]


def _discover_fov_keys(zarr_root) -> List[str]:
    """Enumerate leaf FOV keys across flat (v7) and nested (v8+) layouts.

    Archives from v8 onward (v8, v9, v10 — the current canonical) set
    ``schema_version`` at the root and organize datasets as
    ``modality/tissue/cohort/sample/fov``. v7 archives predate this and
    store flat dataset keys directly under the root. Both zarr (via
    ``zf[key]``) and the filesystem resolve ``a/b/c`` the same way, so the
    returned slash-joined keys are drop-in replacements for the old flat
    keys throughout the loader.
    """
    store_path = getattr(zarr_root, "store_path", None)
    store = getattr(store_path, "store", None)
    root = getattr(store, "root", None)
    path = getattr(store_path, "path", "")
    if root is not None:
        root_path = Path(root)
        if path:
            root_path = root_path / path
        # The ``**/preprocessed/zarr.json`` glob costs ~10s on a 45k-file
        # archive; memoize per-process. Archives are immutable for the
        # lifetime of a single loader process. Tests use unique tmp_paths.
        cache_key = str(root_path.resolve()) if root_path.exists() else str(root_path)
        cached = _FOV_KEYS_CACHE.get(cache_key)
        if cached is not None:
            return cached
        keys = sorted(
            preproc_json.parent.parent.relative_to(root_path).as_posix()
            for preproc_json in root_path.glob("**/preprocessed/zarr.json")
        )
        if keys:
            _FOV_KEYS_CACHE[cache_key] = keys
            return keys

    if "schema_version" in zarr_root.attrs:
        raise RuntimeError(
            "Could not discover FOV keys from filesystem for nested zarr archive"
        )
    return list(zarr_root.group_keys())
