import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from numcodecs import Zstd


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _archive_candidate_paths(explicit_path):
    if explicit_path is not None:
        yield Path(explicit_path).expanduser()

    env_path = os.environ.get(DCTConfig.ARCHIVE_ENV_VAR)
    if env_path:
        yield Path(env_path).expanduser()

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        data_dir = Path(data_dir).expanduser()
        for name in DCTConfig.ARCHIVE_CANDIDATE_NAMES:
            yield data_dir / name


def _resolve_archive_path(explicit_path):
    for candidate in _archive_candidate_paths(explicit_path):
        if (candidate / "zarr.json").exists():
            return candidate
    raise FileNotFoundError(
        "DCTConfig reads metadata from a TissueNet zarr archive. "
        f"Pass zarr_path=... or set {DCTConfig.ARCHIVE_ENV_VAR}."
    )


@lru_cache(maxsize=None)
def _archive_root_attrs(zarr_path_str):
    zarr_path = Path(zarr_path_str)
    root_meta = _read_json(zarr_path / "zarr.json")
    return root_meta.get("attributes", {})


def _iter_fov_metadata_paths(zarr_path):
    return sorted(zarr_path.glob("*/*/*/*/*/zarr.json"))


@lru_cache(maxsize=None)
def _archive_domains(zarr_path_str):
    zarr_path = Path(zarr_path_str)
    domains = set()
    for meta_path in _iter_fov_metadata_paths(zarr_path):
        attrs = _read_json(meta_path).get("attributes", {})
        modality = attrs.get("modality")
        if modality:
            domains.add(str(modality).upper())
    return tuple(sorted(domains))


def _dtype_from_zarr_v3(data_type):
    if isinstance(data_type, dict) and data_type.get("name") == "fixed_length_utf32":
        length_bytes = int(data_type["configuration"]["length_bytes"])
        return np.dtype(f"<U{length_bytes // 4}")
    return np.dtype(data_type)


def _read_v3_1d_array(array_dir):
    meta_path = array_dir / "zarr.json"
    if not meta_path.exists():
        return None

    meta = _read_json(meta_path)
    n = int(meta["shape"][0])
    dtype = _dtype_from_zarr_v3(meta["data_type"])
    fill_value = meta.get("fill_value", 0)
    out = np.full((n,), fill_value, dtype=dtype)
    if n == 0:
        return out

    chunk_len = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    zstd = None
    for codec in meta.get("codecs", []):
        if codec.get("name") == "zstd":
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


@lru_cache(maxsize=None)
def _archive_tissue_celltype_mapping(zarr_path_str):
    zarr_path = Path(zarr_path_str)
    mapping = {}
    for cell_type_meta in sorted(
        zarr_path.glob("*/*/*/*/*/preprocessed/cell_type_info/cell_type/zarr.json")
    ):
        rel_parts = cell_type_meta.relative_to(zarr_path).parts
        tissue = rel_parts[1]
        values = _read_v3_1d_array(cell_type_meta.parent)
        if values is None:
            continue
        tissue_types = mapping.setdefault(tissue, set())
        tissue_types.update(str(value) for value in values if str(value))
    return {
        tissue: sorted(cell_types)
        for tissue, cell_types in mapping.items()
        if cell_types
    }


class DCTConfig:
    ARCHIVE_ENV_VAR = "DEEPCELL_TYPES_ZARR_PATH"
    ARCHIVE_CANDIDATE_NAMES = (
        "tissuenet-v9.zarr",
        "tissuenet-v8.zarr",
        "tissuenet-caitlin-labels.zarr",
    )
    CHANNEL_ALIASES = {
        "CgA": "CHGA",
        "DC-SIGN": "CD209",
        "DCSIGN": "CD209",
        "Galectin-9": "Galectin9",
        "HO-1": "HO1",
        "Pan-Cytokeratin": "PanCK",
        "PANCK": "PanCK",
    }

    def __init__(self, zarr_path=None):
        self.SEED = 0
        self.MAX_NUM_CHANNELS = 80
        self.BATCH_SIZE = 400
        self.MAX_CHUNK_PER_CT_PER_DATASET = 25
        self.PERCENTILE_THRESHOLD = 99.0
        self.HIST_NORM_KERNEL_SIZE = 128
        self.CROP_SIZE = 32
        self.OUTPUT_SIZE = self.CROP_SIZE
        self.STANDARD_MPP_RESOLUTION = 0.5

        self.data_folder = Path(os.path.dirname(__file__)) / "config"
        self._tct_mapping = None

        self.zarr_path = _resolve_archive_path(zarr_path)
        root_attrs = _archive_root_attrs(str(self.zarr_path))

        cell_type_mapping = root_attrs.get("cell_type_mapping")
        if not cell_type_mapping:
            raise ValueError(
                f"{self.zarr_path} is missing root attrs.cell_type_mapping."
            )
        all_channels = root_attrs.get("all_standardized_channels")
        if not all_channels:
            raise ValueError(
                f"{self.zarr_path} is missing root attrs.all_standardized_channels."
            )

        self._ct2idx = {str(ct): int(idx) for ct, idx in cell_type_mapping.items()}
        self._core_celltypes = list(self._ct2idx)
        self._master_channels = [str(ch) for ch in all_channels]
        self._marker2idx = {ch: idx for idx, ch in enumerate(self.master_channels)}

        domains = _archive_domains(str(self.zarr_path))
        self._domain2idx = {domain: idx for idx, domain in enumerate(domains)}
        self.NUM_DOMAINS = len(self._domain2idx)
        self.NUM_CELLTYPES = len(self.ct2idx)

        with open(self.data_folder / "channel_mapping.yaml") as fh:
            channel_mapping = yaml.safe_load(fh) or {}

        canonical_mapping = {ch: ch for ch in self.master_channels}
        canonical_mapping.update(
            {
                alias: target
                for alias, target in self.CHANNEL_ALIASES.items()
                if target in self._marker2idx
            }
        )
        canonical_mapping.update(
            {
                alias: target
                for alias, target in channel_mapping.items()
                if target in self._marker2idx
            }
        )
        self.channel_mapping = canonical_mapping
        self._marker_lookup_lower = {
            marker.lower(): marker for marker in self._marker2idx
        }

    def resolve_channel_name(self, ch_name):
        if ch_name in self.marker2idx:
            return ch_name

        mapped = self.channel_mapping.get(ch_name)
        if mapped in self.marker2idx:
            return mapped

        if not ch_name:
            return None

        alias = self.CHANNEL_ALIASES.get(ch_name)
        if alias in self.marker2idx:
            return alias

        lower_name = str(ch_name).lower()
        lower_mapped = self._marker_lookup_lower.get(lower_name)
        if lower_mapped is not None:
            return lower_mapped

        for alias_name, target in self.channel_mapping.items():
            if str(alias_name).lower() == lower_name and target in self.marker2idx:
                return target
        return None

    @property
    def ct2idx(self):
        return self._ct2idx

    @property
    def domain2idx(self):
        return self._domain2idx

    @property
    def marker2idx(self):
        return self._marker2idx

    @property
    def core_celltypes(self):
        return self._core_celltypes

    @property
    def master_channels(self):
        return self._master_channels

    def get_tct_mapping(self):
        if self._tct_mapping is None:
            root_attrs = _archive_root_attrs(str(self.zarr_path))
            root_mapping = root_attrs.get("tissue_celltype_mapping")
            if root_mapping:
                self._tct_mapping = {
                    tissue: sorted(ct for ct in cell_types if ct in self.ct2idx)
                    for tissue, cell_types in root_mapping.items()
                }
            else:
                archive_mapping = _archive_tissue_celltype_mapping(
                    str(self.zarr_path)
                )
                self._tct_mapping = {
                    tissue: sorted(ct for ct in cell_types if ct in self.ct2idx)
                    for tissue, cell_types in archive_mapping.items()
                }
        return self._tct_mapping
