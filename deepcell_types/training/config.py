"""Training-side configuration for ``deepcell_types.training``.

Reads all mappings and metadata directly from a TissueNet zarr v3 archive
(group/array metadata in ``zarr.json``; attribute keys serialized as strings,
including centroid indices).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .archive import (  # noqa: F401  -- re-exported for backward compat
    _FINGERPRINT_CACHE,
    _FOV_KEYS_CACHE,
    _patch_zarr_v3_alpha_metadata,
    _local_zarr_root_path,
    cached_archive_metadata_fingerprint,
    archive_metadata_fingerprint,
    archive_array_fingerprint,
    _discover_fov_keys,
)
from .embeddings import LazyMarkerPositivityDict
from .hierarchy import CELL_TYPE_HIERARCHY  # noqa: F401  -- re-export for back-compat

logger = logging.getLogger(__name__)

# Archive resolution: explicit zarr_path > DEEPCELL_TYPES_ZARR_PATH env var.
# No hardwired filesystem default — a lab-internal NFS path here would only
# work on one host and produce confusing FileNotFoundError elsewhere.
ARCHIVE_ENV_VAR = "DEEPCELL_TYPES_ZARR_PATH"

# Config directory — sibling of this file inside the package:
# deepcell_types/training/config.py -> deepcell_types/training/config/.
# Lives inside the package so `pip install` ships the YAMLs as
# package-data (see [tool.setuptools.package-data] in pyproject.toml).
CONFIG_DIR = Path(__file__).parent / "config"

# Training constants
WARMUP_PCT = 0.05  # Warmup percentage for OneCycleLR scheduler



class TissueNetConfig:
    """
    Configuration loaded from TissueNet zarr archive.

    The class loads all configuration (cell-type registry, channel registry,
    cell-type / channel embeddings, tissue and modality lists, etc.) from the
    zarr archive itself rather than from out-of-band YAML files. The archive
    uses zarr v3 format (``zarr.json`` metadata).

    The zarr path can be supplied directly via ``zarr_path=...`` or via the
    ``DEEPCELL_TYPES_ZARR_PATH`` environment variable. There is no hard-coded
    filesystem default; one of these two must be provided.

    Usage:
        # Resolve from the DEEPCELL_TYPES_ZARR_PATH env var
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

    def __init__(self, zarr_path: Optional[Path] = None):
        """
        Initialize config from zarr archive.

        Args:
            zarr_path: Path to TissueNet zarr archive. If None, falls back to
                the ``DEEPCELL_TYPES_ZARR_PATH`` environment variable.
        """
        import zarr

        if zarr_path is None:
            env_path = os.environ.get(ARCHIVE_ENV_VAR)
            if not env_path:
                raise FileNotFoundError(
                    f"No zarr_path provided and {ARCHIVE_ENV_VAR} is unset. "
                    "Pass zarr_path=... explicitly or set the env var."
                )
            zarr_path = env_path
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
        self._dataset_celltypes_cache: Optional[Dict[str, List[str]]] = None
        self._tissue_celltype_mapping_cache: Optional[Dict[str, List[str]]] = None
        # Single source of truth for per-dataset MP DataFrames. Initialized
        # to None and replaced by LazyMarkerPositivityDict on first access of
        # the marker_positivity_labels property (or get_marker_positivity).
        self._marker_positivity_cache: Optional["LazyMarkerPositivityDict"] = None
        self._mp_keys: Optional[List[str]] = None  # Keys with marker_positivity groups
        self._dataset_keys: Optional[List[str]] = None

        # Compute domain2idx (after loading domain mapping)
        self._domain2idx: Optional[Dict[str, int]] = None
        # Tissue name → idx (no reserved null index; built lazily via
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
            except (FileNotFoundError, OSError, UnicodeDecodeError):
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
            except (FileNotFoundError, OSError, UnicodeDecodeError):
                pass

            # Marker positivity presence check — a plain os.path.exists
            # cannot fail in a way we need to guard against here, so no
            # try/except is needed around it.
            mp_json_path = f"{zarr_dir_str}/{key}/marker_positivity/zarr.json"
            result["has_mp"] = os.path.exists(mp_json_path)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
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
        (
            domain_mapping,
            dataset_celltypes,
            tissue_celltype_mapping,
            mp_keys,
        ) = self._aggregate_metadata(results, self._ct2idx)

        self._domain_mapping_cache = domain_mapping
        self._dataset_celltypes_cache = dataset_celltypes
        self._tissue_celltype_mapping_cache = tissue_celltype_mapping
        self._mp_keys = mp_keys
        self._all_mappings_computed = True

        logger.info(
            f"Computed all mappings: {len(domain_mapping)} domains, "
            f"{len(dataset_celltypes)} datasets with annotations, "
            f"{len(tissue_celltype_mapping)} tissues, "
            f"{len(mp_keys)} MP datasets"
        )

    @staticmethod
    def _aggregate_metadata(results, ct2idx):
        """Aggregate per-dataset metadata dicts into the cached mappings.

        Pure (no I/O) so it is unit-testable without a zarr archive. Takes the
        list of ``_read_dataset_metadata`` results and returns:

        - ``domain_mapping``: dataset key -> modality, for every dataset.
        - ``dataset_celltypes``: dataset key -> sorted list of *all* annotated
          cell-type names (including names absent from ``ct2idx``); only
          datasets that have annotations appear.
        - ``tissue_celltype_mapping``: tissue -> sorted list of annotated cell
          types that are in ``ct2idx``, merged across datasets. Tissues whose
          allowed-CT set is empty are dropped so ``--apply_tissue_mask`` cannot
          produce an all-Inf logit mask (NaN softmax) on a FOV whose tissue
          lacks any labeled annotation in the archive.
        - ``mp_keys``: datasets that expose a marker_positivity group.
        """
        domain_mapping: Dict[str, str] = {}
        dataset_celltypes: Dict[str, List[str]] = {}
        tissue_ct_mapping: Dict[str, set] = {}
        mp_keys: List[str] = []

        for r in results:
            key = r["key"]
            domain_mapping[key] = r["domain"]

            tissue = r["tissue"]
            ct_names = r["ct_names"]
            if ct_names is not None:
                dataset_celltypes[key] = sorted(ct_names)
                if tissue is not None:
                    valid = tissue_ct_mapping.setdefault(tissue, set())
                    valid.update(ct for ct in ct_names if ct in ct2idx)

            if r["has_mp"]:
                mp_keys.append(key)

        tissue_celltype_mapping = {
            k: sorted(v) for k, v in tissue_ct_mapping.items() if v
        }
        return domain_mapping, dataset_celltypes, tissue_celltype_mapping, mp_keys

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
    def dataset_celltypes(self) -> Dict[str, List[str]]:
        """Per-dataset annotated cell-type names (post-standardization).

        Maps each dataset key that has annotations to the sorted list of
        cell-type names present in its archive annotations. Cell types are
        already canonical in the zarr, so these are the names verbatim -- no
        remapping. Names absent from ``ct2idx`` are retained here; callers
        apply the ``ct2idx`` filter themselves.
        """
        if self._dataset_celltypes_cache is None:
            self._compute_all_mappings()
        return self._dataset_celltypes_cache

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
        if self._marker_positivity_cache is None:
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
        dataset_celltypes = self.dataset_celltypes
        ct2idx = self.ct2idx

        tissue_types: Dict[str, set] = defaultdict(set)
        for ds_key in train_datasets:
            tissue = self.get_tissue_for_dataset(ds_key)
            if tissue is None:
                continue
            for ct_name in dataset_celltypes.get(ds_key, ()):
                if ct_name in ct2idx:
                    tissue_types[tissue].add(ct_name)

        return {k: sorted(v) for k, v in tissue_types.items()}

    def load_marker_embeddings_array(
        self,
        embedding_model_name: str = "text-embedding-3-large",
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
                    "--svd_output_path embeddings/svd_512.npz"
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
            "Pass --svd_embeddings_path embeddings/svd_512.npz"
        )

    def get_marker_positivity(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """
        Get marker positivity DataFrame for a specific dataset.

        Args:
            dataset_key: Dataset key in the zarr archive

        Returns:
            DataFrame with cell types as index, markers as columns,
            or None if not available.

        Both this method and ``marker_positivity_labels[key]`` share a single
        lazy cache — calling either first does not silently lose entries
        populated by the other.
        """
        labels = self.marker_positivity_labels
        if dataset_key not in labels:
            return None
        return labels[dataset_key]

    def _load_marker_positivity(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """Load marker positivity from zarr for a dataset.

        Emits a one-time warning aggregated across all datasets if any MP row
        label is not in ``ct2idx``. Such rows are dead code (no cell carries
        that label, so ``df.loc[ct, ...]`` never reaches them) but their
        presence indicates the archive's standardization passes did not
        propagate to MP rows. Aggregating prevents flooding multi-worker
        DataLoader startup logs (4 workers × ~285 MP datasets = 1140 lines
        otherwise).

        Recovery: rerun the archive-ingestion pipeline used to build this
        archive so the standardized cell-type labels propagate to the MP
        sub-group attrs.
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
                "archive schema drift — rerun the archive-ingestion pipeline.",
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
                    "first seen in dataset %s): %s. Rerun the archive-ingestion "
                    "pipeline to repopulate the standardized labels.",
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
        # Matches hubmap-to-zarr's migrate_archive_v2.py step4_tissue_normalization
        # (lowercase + strip + remove internal spaces) so a future multi-word raw
        # tissue value ingested without pre-normalization ("Lymph Node") doesn't
        # silently split into a category distinct from the archive's normalized
        # form ("lymphnode").
        return tissue.lower().strip().replace(" ", "")

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


# Backward-compat re-exports: these were defined here pre-split.
# Canonical home is now patch.py. (archive.py re-exports happen at the top.)
from .patch import (  # noqa: F401, E402
    compute_distance_transform,
    extract_patch,
    extract_patch_from_zarr,
)

