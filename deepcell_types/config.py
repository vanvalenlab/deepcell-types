import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from numcodecs import Zstd

__all__ = ["DCTConfig"]


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _archive_candidate_paths(explicit_path):
    # Resolution precedence: an explicit zarr_path wins, then the
    # DEEPCELL_TYPES_ZARR_PATH env var. If neither yields an archive, the caller
    # falls back to the packaged vocab.json (archive-free inference).
    if explicit_path is not None:
        yield Path(explicit_path).expanduser()

    env_path = os.environ.get(DCTConfig.ARCHIVE_ENV_VAR)
    if env_path:
        yield Path(env_path).expanduser()


def _resolve_archive_path(explicit_path):
    """Resolve a TissueNet zarr archive, or return None to use the packaged vocab.

    Resolution consults an explicit ``zarr_path`` then the
    ``DEEPCELL_TYPES_ZARR_PATH`` environment variable. An explicitly-passed
    ``zarr_path`` that does not contain an archive is an error (raises). When no
    archive is given/found and no explicit path was requested, returns ``None``
    so the caller falls back to the packaged vocabulary snapshot (archive-free
    inference).
    """
    env_path = os.environ.get(DCTConfig.ARCHIVE_ENV_VAR)
    for candidate in _archive_candidate_paths(explicit_path):
        if (candidate / "zarr.json").exists():
            return candidate
    if explicit_path is not None:
        raise FileNotFoundError(
            f"No TissueNet zarr archive found at {explicit_path} "
            "(expected a 'zarr.json' there)."
        )
    if env_path:
        raise FileNotFoundError(
            f"{DCTConfig.ARCHIVE_ENV_VAR} points to {env_path}, but no TissueNet "
            "zarr archive was found there (expected a 'zarr.json' file)."
        )
    return None


def _load_packaged_vocab():
    """Load the marker / cell-type vocabulary snapshot shipped with the package.

    This lets ``predict()`` run without the (large) TissueNet zarr archive: the
    inference path only needs the marker and cell-type registries, which are a
    few KB. The snapshot must match the released checkpoint's frozen channel
    map; the checkpoint's own ``canonical_channels``/``ct2idx`` (when present)
    are asserted against it at load time.
    """
    vocab_path = Path(__file__).parent / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            "No TissueNet zarr archive found and the packaged vocabulary "
            f"snapshot is missing ({vocab_path}). Pass zarr_path=... or set "
            f"{DCTConfig.ARCHIVE_ENV_VAR}."
        )
    return _read_json(vocab_path)


# These metadata readers are cached unbounded by path string. They assume the
# archive is immutable for the lifetime of the process — a second DCTConfig over
# the same path after the archive changed on disk returns the cached (stale)
# result. Inference processes are short-lived and archives are published
# read-only, so this holds in practice.
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
        # Read whatever the (decoded) chunk actually holds rather than forcing
        # ``count=chunk_len``: a conformant final chunk is padded to chunk_len,
        # but some writers store it unpadded. Slicing handles both; a genuinely
        # corrupt/truncated buffer raises with the offending path for context.
        try:
            chunk = np.frombuffer(data, dtype=dtype)
        except ValueError as e:
            raise ValueError(
                f"Failed to decode zarr chunk {chunk_path} as {dtype}: {e}"
            ) from e
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
    """Inference-time configuration for ``deepcell_types.predict``.

    A ``DCTConfig`` snapshots the marker / cell-type registry and a small
    set of preprocessing constants. The registry is loaded from a TissueNet
    zarr v3 archive when one is available; otherwise it falls back to the
    vocabulary snapshot (``vocab.json``) shipped with the package, so
    ``predict()`` works without the (large) archive.

    Note: ``DCTConfig`` is unrelated to
    :data:`deepcell_types.DEFAULT_CONFIG` (a preprocessing op list for the
    ``preprocess`` hook), despite the similar name.

    Parameters
    ----------
    zarr_path : str or Path, optional
        Path to a TissueNet zarr v3 archive. If ``None``, looks for the
        ``DEEPCELL_TYPES_ZARR_PATH`` environment variable, and otherwise falls
        back to the packaged ``vocab.json``. An explicitly-passed ``zarr_path``
        that does not contain an archive raises ``FileNotFoundError``. (The
        tissue->cell-type mapping from :meth:`get_tct_mapping` is only available
        in archive mode.)

    Attributes
    ----------
    ARCHIVE_ENV_VAR : str
        Environment-variable name consulted when ``zarr_path`` is ``None``.
    CHANNEL_ALIASES : dict[str, str]
        Canonical channel-name aliases applied during preprocessing.

    Examples
    --------
    >>> config = DCTConfig()  # reads DEEPCELL_TYPES_ZARR_PATH
    >>> config = DCTConfig(zarr_path="/path/to/tissuenet.zarr")
    """

    ARCHIVE_ENV_VAR = "DEEPCELL_TYPES_ZARR_PATH"
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
        self.MAX_NUM_CHANNELS = 80
        # Per-channel bright-spot clip percentile for the inference patch
        # generator. Set to 99.9 to match the recipe the training archive's
        # ``preprocessed/raw`` was built with (``preprocess_fov`` /
        # ``DEFAULT_PERCENTILE``), so inference preprocessing tracks what the
        # checkpoint was trained on. (The prior value of 99.0 was a carryover
        # from the original library packaging; on a 6-FOV / 3.3k-cell test-split
        # sample, 99.9 reproduced the canonical predictions slightly better —
        # 92.5% vs 91.9% argmax agreement.)
        self.PERCENTILE_THRESHOLD = 99.9
        self.CROP_SIZE = 32
        self.OUTPUT_SIZE = self.CROP_SIZE
        self.STANDARD_MPP_RESOLUTION = 0.5

        self._tct_mapping = None

        # Source the marker / cell-type vocabulary from a TissueNet zarr archive
        # when one is available, else from the packaged vocab.json snapshot
        # (archive-free inference). Training always passes an archive via
        # TissueNetConfig; only the inference DCTConfig uses the fallback.
        self.zarr_path = _resolve_archive_path(zarr_path)
        if self.zarr_path is not None:
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
            domains = list(_archive_domains(str(self.zarr_path)))
        else:
            vocab = _load_packaged_vocab()
            cell_type_mapping = vocab["cell_type_mapping"]
            all_channels = vocab["all_standardized_channels"]
            domains = list(vocab.get("domains", []))

        self._ct2idx = {str(ct): int(idx) for ct, idx in cell_type_mapping.items()}
        self._master_channels = [str(ch) for ch in all_channels]
        self._marker2idx = {ch: idx for idx, ch in enumerate(self.master_channels)}

        self._domain2idx = {domain: idx for idx, domain in enumerate(domains)}
        self.NUM_DOMAINS = len(self._domain2idx)
        self.NUM_CELLTYPES = len(self.ct2idx)

        with open(Path(__file__).parent / "channel_mapping.yaml") as fh:
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
    def master_channels(self):
        return self._master_channels

    def get_tct_mapping(self):
        if self.zarr_path is None:
            raise RuntimeError(
                "The tissue->cell-type mapping requires a TissueNet zarr "
                "archive; construct DCTConfig(zarr_path=...). It is not part "
                "of the packaged vocabulary snapshot."
            )
        if self._tct_mapping is None:
            root_attrs = _archive_root_attrs(str(self.zarr_path))
            root_mapping = root_attrs.get("tissue_celltype_mapping")
            if root_mapping:
                self._tct_mapping = {
                    tissue: sorted(ct for ct in cell_types if ct in self.ct2idx)
                    for tissue, cell_types in root_mapping.items()
                }
            else:
                archive_mapping = _archive_tissue_celltype_mapping(str(self.zarr_path))
                self._tct_mapping = {
                    tissue: sorted(ct for ct in cell_types if ct in self.ct2idx)
                    for tissue, cell_types in archive_mapping.items()
                }
        return self._tct_mapping
