"""
Configuration module for deepcelltypes-cell-type-assignment-pytorch.

Loads configuration from TissueNet zarr v3 archive, removing dependency on deepcelltypes_kit.
All mappings and metadata are read directly from the zarr archive attributes.

The zarr v3 archive stores all group/array metadata in zarr.json files and
serializes attribute keys as strings (including centroid indices).
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize

logger = logging.getLogger(__name__)

# Default paths using DATA_DIR environment variable
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))
DEFAULT_ZARR_PATH = DATA_DIR / "tissuenet-caitlin-labels.zarr"

# Config directory (relative to repo root: deepcell_types/training/config.py
# -> deepcell_types/training/ -> deepcell_types/ -> repo root -> config/).
# After migrating into deepcell-types this is one ``.parent`` deeper than
# the original deepcelltypes-cell-type-assignment-pytorch layout, where
# this module sat directly under deepcelltypes/.
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

# Cell type hierarchy for evaluation.
# Predictions of child types count as correct when ground truth is a parent type.
# Training loss still uses exact labels.
CELL_TYPE_HIERARCHY = {
    "Tcell": ["CD4T", "CD8T", "Treg", "NKT"],
    "Stromal": ["Fibroblast", "Pericyte"],
}

# Training constants
WARMUP_PCT = 0.05  # Warmup percentage for OneCycleLR scheduler


def _patch_zarr_v3_alpha_metadata() -> None:
    """Allow the installed zarr 3 alpha to read metadata emitted by newer writers."""
    try:
        from zarr.core.group import GroupMetadata
        from zarr.core.metadata.v3 import ArrayV3Metadata
    except Exception:
        return

    if not getattr(GroupMetadata.from_dict, "_dct_compat", False):
        group_from_dict = GroupMetadata.from_dict.__func__

        def _group_from_dict_compat(cls, data):
            data = data.copy()
            data.pop("consolidated_metadata", None)
            return group_from_dict(cls, data)

        _group_from_dict_compat._dct_compat = True
        GroupMetadata.from_dict = classmethod(_group_from_dict_compat)

    if not getattr(ArrayV3Metadata.from_dict, "_dct_compat", False):
        array_from_dict = ArrayV3Metadata.from_dict.__func__

        def _array_from_dict_compat(cls, data):
            data = data.copy()
            storage_transformers = data.pop("storage_transformers", [])
            if storage_transformers not in (None, []):
                raise ValueError(
                    f"unsupported storage_transformers={storage_transformers!r}"
                )
            return array_from_dict(cls, data)

        _array_from_dict_compat._dct_compat = True
        ArrayV3Metadata.from_dict = classmethod(_array_from_dict_compat)


_patch_zarr_v3_alpha_metadata()


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
    """Enumerate leaf FOV keys across v7 (flat) and v8 (5-level) layouts.

    v8 archives set ``schema_version`` at the root and organize datasets as
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


class LazyMarkerPositivityDict(dict):
    """Dict-like object that lazily loads marker positivity DataFrames on demand.

    Only datasets that actually have marker_positivity groups in the zarr archive
    will be loaded, and only when first accessed. This avoids iterating all ~1,900
    datasets at init time when only ~285 have MP data.

    Supports __contains__, __getitem__, keys(), values(), items(), __iter__, __len__,
    and list() for full dict compatibility (e.g., nimbus.py iterates keys).
    """

    def __init__(self, config: "TissueNetConfig", mp_keys: List[str]):
        """
        Args:
            config: TissueNetConfig instance (for _load_marker_positivity)
            mp_keys: List of dataset keys that have marker_positivity groups
        """
        super().__init__()
        self._config = config
        self._mp_keys = set(mp_keys)
        self._loaded_keys: set = set()  # Track which keys we've attempted to load
        self._fully_loaded = False

    def __reduce_ex__(self, protocol):
        loaded = {k: dict.__getitem__(self, k) for k in dict.keys(self)}
        state = {
            "zarr_path": (
                str(self._config.zarr_path)
                if getattr(self._config, "zarr_path", None) is not None
                else None
            ),
            "mp_keys": list(self._mp_keys),
            "loaded_keys": list(self._loaded_keys),
            "fully_loaded": self._fully_loaded,
        }
        return (self.__class__._from_pickle, (state, loaded))

    @classmethod
    def _from_pickle(cls, state, loaded):
        obj = cls.__new__(cls)
        dict.__init__(obj)
        dict.update(obj, loaded)
        obj.__setstate__(state)
        return obj

    def __setstate__(self, state):
        zarr_path = state.get("zarr_path")
        self._config = TissueNetConfig(zarr_path) if zarr_path is not None else None
        self._mp_keys = set(state.get("mp_keys", []))
        self._loaded_keys = set(state.get("loaded_keys", []))
        self._fully_loaded = bool(state.get("fully_loaded", False))

    def _load_one(self, key: str):
        """Load a single dataset's marker positivity if not already loaded."""
        if key not in self._loaded_keys:
            self._loaded_keys.add(key)
            df = self._config._load_marker_positivity(key)
            if df is not None:
                super().__setitem__(key, df)

    def _load_all(self):
        """Load all MP datasets (for iteration)."""
        if self._fully_loaded:
            return
        for key in self._mp_keys:
            self._load_one(key)
        self._fully_loaded = True

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if key in self._mp_keys and key not in self._loaded_keys:
            self._load_one(key)
            return super().__contains__(key)
        return False

    def __getitem__(self, key):
        if not super().__contains__(key) and key in self._mp_keys:
            self._load_one(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        self._load_all()
        return super().keys()

    def values(self):
        self._load_all()
        return super().values()

    def items(self):
        self._load_all()
        return super().items()

    def __iter__(self):
        self._load_all()
        return super().__iter__()

    def __len__(self):
        self._load_all()
        return super().__len__()


class TissueNetConfig:
    """
    Configuration loaded from TissueNet zarr archive.

    This class provides the same interface as the old config from deepcelltypes_kit,
    but loads all configuration from the zarr archive instead of separate YAML files.

    The zarr path defaults to $DATA_DIR/tissuenet-caitlin-labels.zarr where DATA_DIR
    defaults to /data2. Override DATA_DIR via environment variable or .envrc (direnv).
    The archive uses zarr v3 format (zarr.json metadata files).

    Usage:
        # Use default path ($DATA_DIR/tissuenet-caitlin-labels.zarr)
        config = TissueNetConfig()

        # Or specify path explicitly
        config = TissueNetConfig("/path/to/tissuenet.zarr")

        # Access mappings
        ct_idx = config.ct2idx["Bcell"]
        marker_idx = config.marker2idx["CD45"]
    """

    # Constants
    SEED = 0
    MAX_NUM_CHANNELS = 80
    BATCH_SIZE = 400
    CROP_SIZE = 32  # Extraction size (direct 32x32 to match PatchDataset)
    OUTPUT_SIZE = 32  # Final patch size (no resize when equal to CROP_SIZE)
    STANDARD_MPP_RESOLUTION = 0.5

    def __init__(self, zarr_path: Path = DEFAULT_ZARR_PATH):
        """
        Initialize config from zarr archive.

        Args:
            zarr_path: Path to TissueNet zarr archive
        """
        import zarr

        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr archive not found: {self.zarr_path}")

        self._zf = zarr.open_group(self.zarr_path, mode="r")

        # Load root attributes
        self._cell_type_mapping = dict(self._zf.attrs.get("cell_type_mapping", {}))
        self._all_channels = list(self._zf.attrs.get("all_standardized_channels", []))
        self._all_cell_types = list(
            self._zf.attrs.get("all_standardized_cell_types", [])
        )

        # Build ct2idx from cell_type_mapping (maps cell type name -> integer ID)
        # This is used for model output labels
        self._ct2idx = {ct: idx for ct, idx in self._cell_type_mapping.items()}

        # Build marker2idx from all standardized channels
        self._marker2idx = {ch: idx for idx, ch in enumerate(self._all_channels)}

        # Lazy-loaded caches
        self._all_mappings_computed = False
        self._domain_mapping_cache: Optional[Dict[str, str]] = None
        self._celltype_mapping_cache: Optional[Dict[str, Dict[str, str]]] = None
        self._tissue_celltype_mapping_cache: Optional[Dict[str, List[str]]] = None
        self._marker_positivity_cache: Dict[str, pd.DataFrame] = {}
        self._mp_keys: Optional[List[str]] = None  # Keys with marker_positivity groups
        self._dataset_keys: Optional[List[str]] = None

        # Compute domain2idx (after loading domain mapping)
        self._domain2idx: Optional[Dict[str, int]] = None
        # Tissue name → idx (index 0 reserved for null token; built lazily via
        # tissue2idx property after tissue_celltype_mapping is computed).
        self._tissue2idx_cache: Optional[Dict[str, int]] = None

        # Number of classes and markers
        self.NUM_CELLTYPES = len(self._ct2idx)
        self.NUM_MARKERS = len(self._all_channels)

        # Tumor dataset flags (for binary tumor prediction head)
        self._tumor_datasets = set(self._zf.attrs.get("tumor_datasets", []))

        logger.info(f"Loaded TissueNetConfig from {self.zarr_path}")
        logger.info(f"  Cell types: {self.NUM_CELLTYPES}")
        logger.info(f"  Channels: {len(self._all_channels)}")

    @property
    def ct2idx(self) -> Dict[str, int]:
        """Cell type name to integer index mapping."""
        return self._ct2idx

    @property
    def marker2idx(self) -> Dict[str, int]:
        """Marker/channel name to integer index mapping."""
        return self._marker2idx

    @property
    def tumor_datasets(self) -> set:
        """Set of dataset keys that contain tumor cells."""
        return self._tumor_datasets

    @property
    def domain2idx(self) -> Dict[str, int]:
        """Domain (modality) to integer index mapping."""
        if self._domain2idx is None:
            # Get unique domains from domain_mapping
            domains = sorted(set(self.domain_mapping.values()))
            self._domain2idx = {d: idx for idx, d in enumerate(domains)}
        return self._domain2idx

    @property
    def NUM_DOMAINS(self) -> int:
        """Number of unique domains."""
        return len(self.domain2idx)

    @property
    def tissue2idx(self) -> Dict[str, int]:
        """Tissue name → integer index mapping.

        Built from ``tissue_celltype_mapping`` (sorted alphabetically). All
        archive datasets must declare a non-empty ``tissue`` attr; for
        Pan-M Gold-Standard FOVs the lookup goes through
        ``deepcell_types.training.gold_metadata.resolve_gold_metadata``. There is
        no reserved null index — the MP head raises if tissue_idx is None.
        """
        if not hasattr(self, "_tissue2idx_cache") or self._tissue2idx_cache is None:
            tissues = sorted(t for t in set(self.tissue_celltype_mapping.keys()) if t)
            self._tissue2idx_cache = {t: i for i, t in enumerate(tissues)}
        return self._tissue2idx_cache

    @property
    def NUM_TISSUES(self) -> int:
        """Number of unique tissues."""
        return len(self.tissue2idx)

    @property
    def dataset_keys(self) -> List[str]:
        """List of all dataset keys in the archive.

        Detects archive layout from root attrs: v8 archives expose
        ``schema_version`` and use a 5-level ``modality/tissue/cohort/sample/fov``
        hierarchy; older v7 archives store flat keys at root. For v8 this
        walks the hierarchy and returns slash-joined FOV paths that
        zarr/filesystem both resolve the same way.
        """
        if self._dataset_keys is None:
            self._dataset_keys = _discover_fov_keys(self._zf)
        return self._dataset_keys

    @staticmethod
    def _read_dataset_metadata(args: tuple) -> Dict:
        """Read all metadata for a single dataset directly from zarr.json files.

        Bypasses the zarr Python API entirely to avoid per-dataset overhead.
        Called in a ProcessPoolExecutor to parallelize JSON parsing across cores
        (annotation files total ~1GB, CPU-bound json.load is GIL-limited).

        Args:
            args: (zarr_dir_str, key) tuple

        Returns:
            Dict with key, domain, tissue, ct_names (set or None), has_mp (bool)
        """
        zarr_dir_str, key = args
        result = {
            "key": key,
            "domain": "UNKNOWN",
            "tissue": None,
            "ct_names": None,
            "has_mp": False,
        }
        # Outer except is narrowed to IO / JSON-decode errors so that
        # schema drift (KeyError/AttributeError/TypeError) is NOT silently
        # swallowed. ProcessPool workers print warnings to their stderr;
        # that's acceptable — the point is to keep logic bugs loud.
        try:
            # Dataset-level attrs (small file, ~1KB)
            ds_json_path = f"{zarr_dir_str}/{key}/zarr.json"
            try:
                with open(ds_json_path) as f:
                    ds_data = json.load(f)
                ds_attrs = ds_data.get("attributes", {})
                result["domain"] = ds_attrs.get("modality", "unknown").upper()
                result["tissue"] = ds_attrs.get("tissue", "unknown").lower().strip()
            except (FileNotFoundError, OSError):
                pass

            # Annotation attrs (large files, avg 504KB each)
            ann_json_path = f"{zarr_dir_str}/{key}/cell_types/annotations/zarr.json"
            try:
                with open(ann_json_path) as f:
                    ann_data = json.load(f)
                ann_attrs = ann_data.get("attributes", {})
                ct_names = set()
                for source_key in ("standardized_source", "caitlinb"):
                    source = ann_attrs.get(source_key, {})
                    ct_names.update(
                        ct for ct in source.keys() if ct is not None and ct != "null"
                    )
                if ct_names:
                    result["ct_names"] = ct_names
            except (FileNotFoundError, OSError):
                pass

            # Marker positivity presence check — a plain os.path.exists
            # cannot fail in a way we need to guard against here, so no
            # try/except is needed around it.
            mp_json_path = f"{zarr_dir_str}/{key}/marker_positivity/zarr.json"
            result["has_mp"] = os.path.exists(mp_json_path)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            logger.warning("_read_dataset_metadata failed for %s: %s", key, e)
        return result

    def _compute_all_mappings(self):
        """Compute domain_mapping, celltype_mapping, tissue_celltype_mapping in a single pass.

        Reads zarr.json files directly (bypassing zarr API) and uses
        ProcessPoolExecutor to parallelize the heavy annotation JSON parsing
        (~1GB total) across CPU cores, bypassing the GIL.
        """
        if self._all_mappings_computed:
            return

        from concurrent.futures import ProcessPoolExecutor

        keys = self.dataset_keys
        zarr_dir_str = str(self.zarr_path)

        # Parallel reads of all dataset metadata via ProcessPoolExecutor.
        # Cap at 8 workers (zarr v3 metadata parsing is JSON-bound, not CPU-bound,
        # so more workers don't help) and clamp to available cores for small/CI hosts.
        # Bypasses both zarr API overhead and GIL for json.load()
        args_list = [(zarr_dir_str, key) for key in keys]
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            results = list(
                executor.map(self._read_dataset_metadata, args_list, chunksize=50)
            )

        # Aggregate results (single-threaded, fast)
        domain_mapping: Dict[str, str] = {}
        celltype_mapping: Dict[str, Dict[str, str]] = {}
        tissue_ct_mapping: Dict[str, set] = {}
        mp_keys: List[str] = []
        ct2idx = self._ct2idx

        for r in results:
            key = r["key"]
            domain_mapping[key] = r["domain"]

            tissue = r["tissue"]
            if tissue is not None:
                if tissue not in tissue_ct_mapping:
                    tissue_ct_mapping[tissue] = set()

            ct_names = r["ct_names"]
            if ct_names is not None:
                celltype_mapping[key] = {ct: ct for ct in ct_names}
                if tissue is not None:
                    for ct in ct_names:
                        if ct in ct2idx:
                            tissue_ct_mapping[tissue].add(ct)

            if r["has_mp"]:
                mp_keys.append(key)

        self._domain_mapping_cache = domain_mapping
        self._celltype_mapping_cache = celltype_mapping
        self._tissue_celltype_mapping_cache = {
            k: sorted(v) for k, v in tissue_ct_mapping.items()
        }
        self._mp_keys = mp_keys
        self._all_mappings_computed = True

        logger.info(
            f"Computed all mappings: {len(domain_mapping)} domains, "
            f"{len(celltype_mapping)} celltype maps, "
            f"{len(self._tissue_celltype_mapping_cache)} tissues, "
            f"{len(mp_keys)} MP datasets"
        )

    @property
    def domain_mapping(self) -> Dict[str, str]:
        """
        Dataset key to domain (modality) mapping.

        Built dynamically from dataset attrs.modality.
        Returns uppercase modality names (e.g., "MIBI", "IMC", "CODEX").
        """
        if self._domain_mapping_cache is None:
            self._compute_all_mappings()
        return self._domain_mapping_cache

    @property
    def celltype_mapping(self) -> Dict[str, Dict[str, str]]:
        """Per-dataset cell type mapping.

        After archive migration, cell types are already canonical in the zarr,
        so this is an identity mapping.
        """
        if self._celltype_mapping_cache is None:
            self._compute_all_mappings()
        return self._celltype_mapping_cache

    @property
    def marker_positivity_labels(self) -> "LazyMarkerPositivityDict":
        """
        Marker positivity labels as DataFrames (lazy-loaded on demand).

        Returns a dict-like object mapping dataset_key to DataFrame with:
        - Index: cell type names
        - Columns: marker names
        - Values: 0, 0.5, or 1 (or NaN)

        Only datasets with marker_positivity group are included.
        Individual datasets are loaded on first access (__contains__ / __getitem__),
        not all at once. Iteration (keys/values/items) triggers full load.
        """
        if not isinstance(self._marker_positivity_cache, LazyMarkerPositivityDict):
            # Ensure _compute_all_mappings has run to discover MP keys
            if self._mp_keys is None:
                self._compute_all_mappings()
            self._marker_positivity_cache = LazyMarkerPositivityDict(
                self, self._mp_keys
            )
        return self._marker_positivity_cache

    @property
    def tissue_celltype_mapping(self) -> Dict[str, List[str]]:
        """Tissue to valid cell type list, computed from zarr annotations.

        Only includes post-standardization cell types that are in ct2idx,
        so the model can only predict types it has training data for.
        """
        if self._tissue_celltype_mapping_cache is None:
            self._compute_all_mappings()
        return self._tissue_celltype_mapping_cache

    def build_tissue_mapping_from_split(self, split_file: str) -> Dict[str, List[str]]:
        """Build per-tissue allowed cell types from a training split's ground truth.

        Only cell types that actually appear in training annotations for datasets
        sharing the same tissue are allowed. This is stricter than the archive-based
        tissue_celltype_mapping, which includes ALL cell types from ALL annotations.

        Args:
            split_file: Path to FOV split JSON (must have 'train' key)

        Returns:
            dict mapping tissue name -> sorted list of allowed cell type names
        """
        import json
        from collections import defaultdict

        with open(split_file) as f:
            split = json.load(f)

        train_datasets = list(split["train"].keys())
        ct_mapping = self.celltype_mapping
        ct2idx = self.ct2idx

        tissue_types: Dict[str, set] = defaultdict(set)
        for ds_key in train_datasets:
            tissue = self.get_tissue_for_dataset(ds_key)
            if tissue is None:
                continue
            ds_ct_map = ct_mapping.get(ds_key, {})
            for ct_name in ds_ct_map.keys():
                ct_standard = ds_ct_map.get(ct_name, ct_name)
                if ct_standard in ct2idx:
                    tissue_types[tissue].add(ct_standard)

        return {k: sorted(v) for k, v in tissue_types.items()}

    @property
    def combined_celltype_mapping(self) -> Dict[str, List[str]]:
        """
        Combined cell type grouping mapping.

        Maps group names (e.g., "Tcell", "Epithelial") to lists of individual
        cell types that belong to that group. Loaded from combined_celltypes.yaml.
        """
        if not hasattr(self, "_combined_celltype_mapping_cache"):
            yaml_path = CONFIG_DIR / "combined_celltypes.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                # The YAML maps individual cell types to group names;
                # invert to map group names to lists of cell types
                groups: Dict[str, List[str]] = {}
                for ct, group in raw.items():
                    groups.setdefault(group, []).append(ct)
                self._combined_celltype_mapping_cache = groups
                logger.info(f"Loaded combined_celltype_mapping from {yaml_path}")
            else:
                logger.warning(f"combined_celltypes.yaml not found at {yaml_path}")
                self._combined_celltype_mapping_cache = {}
        return self._combined_celltype_mapping_cache

    @property
    def color_mapping(self) -> Dict[str, str]:
        """Cell type to hex color mapping for visualization."""
        return dict(self._zf.attrs.get("color_mapping", {}))

    @property
    def core_tree(self) -> Dict:
        """Hierarchical cell type taxonomy."""
        return dict(self._zf.attrs.get("core_tree", {}))

    @property
    def lineage_mapping(self) -> Dict[str, str]:
        """Cell type to broad biological lineage mapping."""
        return dict(self._zf.attrs.get("lineage_mapping", {}))

    def get_channel_embedding(
        self, embedding_model_name: str = "deepseek-r1-70b"
    ) -> Dict[str, List[float]]:
        """Get marker/channel embeddings from zarr or fallback to JSON file."""
        embeddings = self._zf.attrs.get(f"marker_embeddings_{embedding_model_name}")
        if embeddings is not None:
            return dict(embeddings)
        # Fallback: search known locations. Narrow the except so that bugs
        # in the JSON structure (e.g. schema drift causing KeyError/TypeError
        # later in the caller) are NOT hidden here.
        search_paths = [
            Path(__file__).parent.parent / "config",
            Path(__file__).parent.parent
            / "deepcelltypes-kit"
            / "deepcelltypes_kit"
            / "config",
        ]
        for config_path in search_paths:
            json_path = config_path / f"marker_embeddings-{embedding_model_name}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        return json.load(f)
                except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        "get_channel_embedding: failed to read %s: %s",
                        json_path,
                        e,
                    )
        logger.warning(f"Could not load marker embeddings for {embedding_model_name}")
        return {}

    def get_celltype_embedding(
        self, embedding_model_name: str = "deepseek-r1-70b-llama-distill-q4_K_M_full"
    ) -> Dict[str, List[float]]:
        """Get cell type embeddings from zarr or fallback to JSON file."""
        embeddings = self._zf.attrs.get(f"celltype_embeddings_{embedding_model_name}")
        if embeddings is not None:
            return dict(embeddings)
        # Fallback to JSON file. Narrow except to only IO / JSON-decode errors.
        config_path = (
            Path(__file__).parent.parent
            / "deepcelltypes-kit"
            / "deepcelltypes_kit"
            / "config"
        )
        json_path = config_path / f"celltype_embeddings-{embedding_model_name}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    return json.load(f)
            except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "get_celltype_embedding: failed to read %s: %s",
                    json_path,
                    e,
                )
        logger.warning(
            f"Could not load cell type embeddings for {embedding_model_name}"
        )
        return {}

    def load_marker_embeddings_array(
        self,
        embedding_model_name: str = "deepseek-r1-70b",
        svd_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load marker embeddings as a numpy array aligned with marker2idx.

        Args:
            embedding_model_name: Name of the embedding model
            svd_path: Optional path to pre-computed SVD-reduced embeddings

        Returns:
            marker_embeddings: (NUM_CHANNELS, embedding_dim) array
        """
        if svd_path is not None:
            if not Path(svd_path).exists():
                raise FileNotFoundError(
                    f"SVD embeddings not found at {svd_path}. "
                    "Generate with: python -m scripts.generate_openai_embeddings "
                    "--svd_output_path embeddings/svd_512_v5.npz"
                )
            data = np.load(svd_path, allow_pickle=True)
            svd_embeds = data["marker_embeddings"]

            # If the npz includes a saved marker2idx, use it to align embeddings
            # with the *current* archive's marker2idx (no silent positional
            # assumption). Legacy files without marker2idx fall through to the
            # positional path with a loud WARNING.
            saved_m2i: Optional[Dict[str, int]] = None
            if "marker2idx" in data.files:
                raw = data["marker2idx"]
                try:
                    if hasattr(raw, "item"):
                        candidate = raw.item()
                    else:
                        candidate = raw
                    if isinstance(candidate, dict):
                        saved_m2i = {str(k): int(v) for k, v in candidate.items()}
                    elif isinstance(candidate, (bytes, str)):
                        saved_m2i = {
                            str(k): int(v) for k, v in json.loads(candidate).items()
                        }
                    else:
                        # Unknown encoding — fall through to positional with warning
                        logger.warning(
                            "load_marker_embeddings_array: saved marker2idx has "
                            "unexpected type %s; falling back to positional load",
                            type(candidate).__name__,
                        )
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(
                        "load_marker_embeddings_array: failed to parse saved "
                        "marker2idx in %s: %s; falling back to positional load",
                        svd_path,
                        e,
                    )
                    saved_m2i = None

            if saved_m2i is not None:
                # If saved and current mappings agree, no reindex needed.
                if saved_m2i == self.marker2idx:
                    logger.info(
                        f"Loaded SVD-reduced marker embeddings from {svd_path}: "
                        f"{svd_embeds.shape} (marker2idx matches archive)"
                    )
                    return svd_embeds

                # Reindex: align saved rows to current marker2idx order.
                embed_dim = svd_embeds.shape[1]
                aligned = np.zeros(
                    (self.NUM_MARKERS, embed_dim), dtype=svd_embeds.dtype
                )
                missing: List[str] = []
                for marker_name, new_idx in self.marker2idx.items():
                    old_idx = saved_m2i.get(marker_name)
                    if old_idx is None:
                        missing.append(marker_name)
                        continue
                    if old_idx < 0 or old_idx >= svd_embeds.shape[0]:
                        missing.append(marker_name)
                        continue
                    aligned[new_idx] = svd_embeds[old_idx]
                if missing:
                    logger.warning(
                        "load_marker_embeddings_array: %d markers missing from "
                        "saved marker2idx and zero-filled; first 5: %s",
                        len(missing),
                        missing[:5],
                    )
                logger.info(
                    f"Loaded SVD-reduced marker embeddings from {svd_path}: "
                    f"{aligned.shape} (reindexed from saved marker2idx; "
                    f"{len(missing)} missing)"
                )
                return aligned

            # Legacy path: no saved marker2idx — positional load with loud warning.
            if svd_embeds.shape[0] < self.NUM_MARKERS:
                padded = np.zeros(
                    (self.NUM_MARKERS, svd_embeds.shape[1]), dtype=np.float32
                )
                padded[: svd_embeds.shape[0]] = svd_embeds
                n_padded = self.NUM_MARKERS - svd_embeds.shape[0]
                padded_names = list(self.marker2idx.keys())[
                    svd_embeds.shape[0] : svd_embeds.shape[0] + 5
                ]
                logger.warning(
                    "load_marker_embeddings_array: %s lacks marker2idx; falling "
                    "back to POSITIONAL load and zero-padding %d rows "
                    "(shape %d -> %d). First zero-filled markers: %s",
                    svd_path,
                    n_padded,
                    svd_embeds.shape[0],
                    self.NUM_MARKERS,
                    padded_names,
                )
                svd_embeds = padded
            else:
                logger.warning(
                    "load_marker_embeddings_array: %s lacks marker2idx; using "
                    "POSITIONAL load — verify order matches archive marker2idx",
                    svd_path,
                )
            logger.info(
                f"Loaded SVD-reduced marker embeddings from {svd_path}: {svd_embeds.shape}"
            )
            return svd_embeds

        raise ValueError(
            "svd_path is required for load_marker_embeddings_array. "
            "Pass --svd_embeddings_path embeddings/svd_512_v5.npz"
        )

    def get_marker_positivity(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """
        Get marker positivity DataFrame for a specific dataset.

        Args:
            dataset_key: Dataset key in the zarr archive

        Returns:
            DataFrame with cell types as index, markers as columns,
            or None if not available.
        """
        if dataset_key in self._marker_positivity_cache:
            return self._marker_positivity_cache[dataset_key]

        df = self._load_marker_positivity(dataset_key)
        if df is not None:
            self._marker_positivity_cache[dataset_key] = df
        return df

    def _load_marker_positivity(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """Load marker positivity from zarr for a dataset.

        Emits a one-time warning aggregated across all datasets if any MP row
        label is not in ``ct2idx``. Such rows are dead code (no cell carries
        that label, so ``df.loc[ct, ...]`` never reaches them) but their
        presence indicates the archive's standardization passes did not
        propagate to MP rows — usually a hubmap-to-zarr migrate_archive_v2
        regression. Aggregating prevents flooding multi-worker DataLoader
        startup logs (4 workers × ~285 MP datasets = 1140 lines otherwise).

        Recovery: rerun ``python migrate_archive_v2.py --zarr-path <archive>``
        in the hubmap-to-zarr repo.
        """
        try:
            ds = self._zf[dataset_key]
        except KeyError:
            return None
        if "marker_positivity" not in ds:
            return None
        try:
            mp = ds["marker_positivity"]
            markers = list(mp.attrs.get("markers", []))
            cell_types = list(mp.attrs.get("cell_types", []))
            matrix = list(mp.attrs.get("positivity_matrix", []))
        except (KeyError, AttributeError, ValueError) as e:
            logger.warning(
                "Failed to read marker_positivity attrs for %s (%s: %s); "
                "MP signal will be unavailable for this dataset. Likely an "
                "archive schema drift — rerun hubmap-to-zarr migrate_archive_v2.py.",
                dataset_key, type(e).__name__, e,
                exc_info=True,
            )
            return None
        if not markers or not cell_types or not matrix:
            return None

        unknown_rows = [ct for ct in cell_types if ct not in self.ct2idx]
        if unknown_rows:
            if not hasattr(self, "_mp_unknown_seen"):
                self._mp_unknown_seen: dict = {}
            seen_set = self._mp_unknown_seen.setdefault("rows", set())
            new_rows = [ct for ct in unknown_rows if ct not in seen_set]
            if new_rows:
                seen_set.update(new_rows)
                logger.warning(
                    "marker_positivity has %d row label(s) not in ct2idx (dead-code rows; "
                    "first seen in dataset %s): %s. Rerun hubmap-to-zarr "
                    "migrate_archive_v2.py --zarr-path <archive> to fix.",
                    len(new_rows), dataset_key, new_rows,
                )

        try:
            return pd.DataFrame(matrix, index=cell_types, columns=markers)
        except ValueError as e:
            logger.warning(
                "marker_positivity matrix for %s is malformed (%s); "
                "MP signal disabled for this dataset.",
                dataset_key, e,
                exc_info=True,
            )
            return None

    @staticmethod
    def _normalize_tissue_name(tissue: str) -> str:
        """Normalize tissue name to canonical form."""
        return tissue.lower().strip()

    def get_tissue_for_dataset(self, dataset_key: str) -> Optional[str]:
        """Get normalized tissue type for a dataset key.

        Narrow exceptions: a missing dataset (``KeyError``) is the only
        expected failure. Anything else (``AttributeError`` on a malformed
        zarr attr, ``TypeError`` from a non-string tissue value) is logged
        and rethrown — silently disabling tissue masking is worse than
        crashing because the symptom would be confusing per-cell errors in
        the DataLoader.
        """
        try:
            ds = self._zf[dataset_key]
        except KeyError:
            return None
        try:
            raw = ds.attrs.get("tissue", None)
        except (AttributeError, ValueError) as e:
            logger.warning(
                "tissue attr read failed for %s (%s: %s) — tissue-aware "
                "masking disabled for this dataset",
                dataset_key, type(e).__name__, e,
                exc_info=True,
            )
            return None
        if raw is None:
            return None
        if not isinstance(raw, str):
            logger.warning(
                "tissue attr for %s has unexpected type %s (value=%r) — "
                "tissue-aware masking disabled",
                dataset_key, type(raw).__name__, raw,
            )
            return None
        return self._normalize_tissue_name(raw)

    def get_excluded_ct_indices(self, dataset_key: str) -> List[int]:
        """Get cell type indices that should be excluded for a dataset's tissue.

        Returns list of ct2idx indices NOT valid for this dataset's tissue.
        """
        tissue = self.get_tissue_for_dataset(dataset_key)
        if tissue is None or tissue not in self.tissue_celltype_mapping:
            return []

        valid_cts = set(self.tissue_celltype_mapping[tissue])
        excluded = []
        for ct, idx in self.ct2idx.items():
            if ct not in valid_cts:
                excluded.append(idx)
        return excluded

    def validate(self) -> bool:
        """
        Validate configuration consistency.

        Checks:
        - Cell type mapping has expected number of types
        - All channels in marker2idx are valid

        Returns:
            True if valid, raises ValueError otherwise
        """
        if len(self._ct2idx) == 0:
            raise ValueError("No cell types found in zarr archive")

        if len(self._all_channels) == 0:
            raise ValueError("No channels found in zarr archive")

        logger.info(
            f"Configuration validated: {self.NUM_CELLTYPES} cell types, "
            f"{len(self._all_channels)} channels, {self.NUM_DOMAINS} domains"
        )
        return True


def compute_distance_transform(self_mask: np.ndarray) -> np.ndarray:
    """Compute normalized distance transform from cell boundary.

    Args:
        self_mask: (H, W) binary mask of the cell

    Returns:
        dist_transform: (H, W) normalized distance transform (float32, 0-1)
    """
    if self_mask.sum() == 0:
        return np.zeros_like(self_mask, dtype=np.float32)
    dt = distance_transform_edt(self_mask).astype(np.float32)
    max_val = dt.max()
    if max_val > 0:
        dt /= max_val
    return dt


def extract_patch_from_zarr(
    raw_zarr,
    mask_zarr,
    centroid: Tuple[float, float],
    cell_idx: int,
    crop_size: int,
    output_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch directly from zarr arrays without loading the full image.

    Efficiently reads only the needed region from disk. Extracts crop_size x crop_size patch and resizes
    to output_size x output_size.

    Args:
        raw_zarr: zarr array (C, H, W) - NOT loaded, just the zarr reference
        mask_zarr: zarr array (H, W) - NOT loaded, just the zarr reference
        centroid: tuple (row, col) - cell centroid coordinates
        cell_idx: int - cell index for mask extraction
        crop_size: int - extraction patch size (e.g., 64)
        output_size: int - final output patch size after resizing (default 32)

    Returns:
        raw_patch: np.ndarray (C, output_size, output_size) - extracted patch (float32)
        mask_patch: np.ndarray (output_size, output_size, 2) - [self_mask, neighbor_mask] (float32)
    """
    before = crop_size // 2
    after = crop_size - before
    C, H, W = raw_zarr.shape

    # Compute crop box center
    row, col = int(round(centroid[0])), int(round(centroid[1]))

    # Calculate the required padding for edge cases
    pad_top = max(0, before - row)
    pad_bottom = max(0, (row + after) - H)
    pad_left = max(0, before - col)
    pad_right = max(0, (col + after) - W)

    # Adjust coordinates for valid region extraction
    r_start = max(0, row - before)
    r_end = min(H, row + after)
    c_start = max(0, col - before)
    c_end = min(W, col + after)

    # Read only the needed region from zarr (this is the key optimization)
    raw_crop = raw_zarr[:, r_start:r_end, c_start:c_end]  # (C, h, w)
    mask_crop = mask_zarr[r_start:r_end, c_start:c_end]  # (h, w)

    # Apply padding if needed (for cells near image boundaries)
    if pad_top or pad_bottom or pad_left or pad_right:
        raw_crop = np.pad(
            raw_crop,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        mask_crop = np.pad(
            mask_crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    # Generate self and neighbor masks (before resizing to preserve integer labels)
    self_mask = (mask_crop == cell_idx).astype(np.float32)
    neighbor_mask = ((mask_crop != cell_idx) & (mask_crop != 0)).astype(np.float32)

    # Resize if output_size differs from crop_size
    if output_size != crop_size:
        # Resize raw: (C, H, W) -> need to transpose for skimage
        raw_crop = np.transpose(raw_crop, (1, 2, 0))  # (H, W, C)
        raw_crop = resize(
            raw_crop,
            (output_size, output_size),
            preserve_range=True,
            anti_aliasing=True,
        )
        raw_crop = np.transpose(raw_crop, (2, 0, 1))  # (C, H, W)

        # Resize masks using nearest neighbor to preserve binary values
        self_mask = resize(
            self_mask,
            (output_size, output_size),
            order=0,  # nearest neighbor
            preserve_range=True,
            anti_aliasing=False,
        )
        neighbor_mask = resize(
            neighbor_mask,
            (output_size, output_size),
            order=0,  # nearest neighbor
            preserve_range=True,
            anti_aliasing=False,
        )

    mask_patch = np.stack([self_mask, neighbor_mask], axis=-1)

    return raw_crop.astype(np.float32), mask_patch.astype(np.float32)


def extract_patch(
    raw_zarr,
    mask_zarr,
    centroid: Tuple[float, float],
    cell_idx: int,
    crop_size: int,
    output_size: int = 32,
    skip_distance_transform: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract patch in factored format.

    Args:
        skip_distance_transform: If True, fill the distance transform channel
            with zeros instead of computing it. Useful for models that don't
            use it (e.g., CellSighter) to avoid the expensive scipy EDT call.

    Returns:
        raw_masked: (C, output_size, output_size) - raw * self_mask per channel
        spatial_context: (3, output_size, output_size) - [self_mask, neighbor_mask, distance_transform]
    """
    raw_crop, mask_patch = extract_patch_from_zarr(
        raw_zarr, mask_zarr, centroid, cell_idx, crop_size, output_size
    )
    # mask_patch: (H, W, 2) -> self_mask, neighbor_mask
    self_mask = mask_patch[:, :, 0]  # (H, W)
    neighbor_mask = mask_patch[:, :, 1]  # (H, W)

    # Compute distance transform (or skip it)
    if skip_distance_transform:
        dist_transform = np.zeros_like(self_mask, dtype=np.float32)
    else:
        dist_transform = compute_distance_transform(self_mask)

    # Build spatial context: (3, H, W)
    spatial_context = np.stack(
        [self_mask, neighbor_mask, dist_transform], axis=0
    ).astype(np.float32)

    # raw * self_mask for each channel → (C, H, W)
    raw_masked = raw_crop * self_mask[np.newaxis, :, :]

    return raw_masked.astype(np.float32), spatial_context
