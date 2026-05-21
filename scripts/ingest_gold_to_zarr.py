"""Ingest the Pan-Multiplex Gold-Standard FOVs into a model-readable zarr v3 archive.

The Nimbus paper's gold-standard set ships per-channel OME-TIFFs +
per-FOV segmentation masks under
``data/gold_standard/gold_standard_labelled/{dataset}/{fovs,masks}/``.
``predict.py``/``FullImageDataset`` consume zarr v3 archives with the
hierarchical layout

    {key}/
      preprocessed/
        raw   (C, H, W) float32      — raw intensity, masked by self in extractor
        mask  (H, W)    uint32       — cell-id labels
        cell_type_info/
          cell_index  (N,) int64
          cell_type   (N,) <U64
        attrs:
          channel_names   list[str]   canonical marker names
          centroids       dict[str_id, [row, col]]
          mpp             0.5         (we resample on ingest)
          scale_factor    1.0         (already in preprocessed coords)
          unique_cell_types list[str]
      attrs:
        tissue, modality, nuclear_channel, membrane_channel

This script writes ONE such top-level key per (dataset, fov) into a
**separate** zarr archive (``--output_zarr`` is required), not the
production training archive. Run as::

    uv run python -m scripts.ingest_gold_to_zarr \
        --gold_dir data/gold_standard/gold_standard_labelled \
        --output_zarr path/to/gold_standard.zarr

Why not write into the production archive: gold has no curated CT labels
(stubbed with a placeholder), and the cell-data pickle cache
(``.{archive}.celldata_cache.pkl``) is fingerprinted on the archive — any
write would invalidate the production cache.

Normalization: there is **no** intensity normalization at the data layer.
``deepcell_types.dct_kit.image_funcs.extract_patch`` only multiplies raw by self_mask
and resizes; the model's PerChannelResNet uses ``InstanceNorm2d`` to
handle cross-modality intensity scale. Gold OME-TIFFs are stored as raw
float32, casting through from uint8/uint16 with no rescaling.

Caution re: hierarchical zarr: the production archive uses a **flat**
top-level key per FOV (not nested ``{dataset}/{fov}/...``). We mirror
that here with keys formatted as ``{dataset}__{fov}`` (double underscore
to disambiguate from the in-vocabulary single-underscore convention used
inside dataset names like ``codex_colon``).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tff
import zarr
from skimage.measure import regionprops
from skimage.transform import resize as sk_resize

from deepcell_types.training.gold_metadata import (
    GOLD_DATASET_METADATA,
    resolve_gold_metadata,
)
from deepcell_types.preprocessing import (
    PreprocessedFov,
    TARGET_MPP,
    preprocess_fov,
)


logger = logging.getLogger(__name__)

# Per-dataset native MPP (µm / pixel). Sources:
#   codex_colon  — Hickey et al. 2023 *Nature*, CODEX colon FOV (~0.377)
#   mibi_breast  — TONIC cohort (Liu et al. 2022) MIBI-TOF run at 0.46
#   mibi_decidua — Greenbaum et al. 2023 *Nature* MIBI-TOF (~0.39)
#   vectra_*     — Akoya Vectra/Opal multispectral component data (0.5)
# The TIFF tags don't carry usable PhysicalSizeX so these are baked in;
# can be overridden via --mpp dataset=value at CLI.
DEFAULT_NATIVE_MPP: Dict[str, float] = {
    "codex_colon": 0.377,
    "mibi_breast": 0.46,
    "mibi_decidua": 0.39,
    "vectra_colon": 0.5,
    "vectra_pancreas": 0.5,
}

# Placeholder cell type assigned to every gold cell. Gold has no curated
# CT labels — the model's MP head doesn't need CT (it's marker-conditioned),
# so this only serves to satisfy the dataloader's cell-loading path. We
# pick a class present in the 51-class taxonomy; "Tcell" is common in all
# five tissues.
PLACEHOLDER_CT = "Tcell"


# ---------------------------------------------------------------------------
# Channel canonicalization (mirrors scripts/gold_channel_aliases.py from PR #39)
# ---------------------------------------------------------------------------
# Embedded here so this script is independent of #39's merge timing.
GOLD_TO_CANON: Dict[str, str] = {
    "FOXP3": "FoxP3",
    "Foxp3": "FoxP3",
    "VIM": "Vimentin",
    "ECAD": "E-cadherin",
    "Ecad": "E-cadherin",
    "panCK": "PanCK",
    "PD-1": "PD1",
    "PD-L1": "PDL1",
    "Bcl2": "Bcl-2",
    "BCL2": "Bcl-2",
    "HLAG": "HLA-G",
    "HLA1": "HLA-Class-1",
    "HLADR": "HLA-Class-2",
    "HLA-Class-2": "HLA-Class-2",
    "aSMA": "SMA",
    "ITLN1": "Intelectin-1",
    "Podoplanin": "PDPN",
    "panCK+CK7+CAM5.2": "PanCK",
}

GOLD_SKIP: set = {
    # MIBI elemental / instrument
    "Au", "Ca", "Co", "Cr", "Fe", "Ir", "Na", "Sc", "Si", "Ta",
    # mask helpers / smoothed sidecars
    "background", "Noodle", "DAPI",
    "CD11c_nuc_exclude", "FOXP3_nuc_include",
    "CK17_smoothed", "ECAD_smoothed",
    "DRAQ5",
    # markers absent from current marker2idx
    "CD117", "CD45RB", "CD40-L", "ChyTr", "Cytokeratin", "CollIV",
    "Collagen1", "GrB", "DCSIGN", "H3", "H3K27me3", "H3K9ac",
    "TBET", "aDefensin5",
}


def canonicalize_channel(name: str) -> Optional[str]:
    """Return canonical marker name, or None to skip."""
    if name in GOLD_SKIP:
        return None
    return GOLD_TO_CANON.get(name, name)


# Vectra FOVs are nested under directories whose names embed the panel
# string; the channel TIFFs inside use clean stems (CD3, FoxP1, ...).
_VECTRA_FOV_PATTERN = re.compile(r"^[0-9a-f]+ ", re.IGNORECASE)


def list_fovs(dataset_dir: Path) -> List[Path]:
    """Return ordered FOV directories for a dataset."""
    fovs_dir = dataset_dir / "fovs"
    if not fovs_dir.exists():
        return []
    return sorted([p for p in fovs_dir.iterdir() if p.is_dir()])


def fov_key(dataset: str, fov_dir_name: str) -> str:
    """Stable, flat key for a (dataset, FOV) pair."""
    # Strip vectra's panel-cruft so the key is short. The FOV-dir name and
    # the mask filename share a prefix, and the prefix is the unique ID.
    m = _VECTRA_FOV_PATTERN.match(fov_dir_name)
    if m:
        clean = fov_dir_name.split(" ", 1)[0]
    else:
        clean = fov_dir_name
    return f"{dataset}__{clean}"


def find_mask_for_fov(dataset_dir: Path, fov_dir: Path) -> Optional[Path]:
    """Locate the segmentation mask that goes with this FOV directory."""
    masks_dir = dataset_dir / "masks"
    if not masks_dir.exists():
        return None
    # Mask filename usually equals FOV-dir name with .ome.tif / .tif / .tiff.
    fov_name = fov_dir.name
    candidates = [
        masks_dir / f"{fov_name}.ome.tif",
        masks_dir / f"{fov_name}.tif",
        masks_dir / f"{fov_name}.tiff",
    ]
    # Vectra: the FOV-dir suffix is "_image", the mask drops the "_image".
    if fov_name.endswith("_image"):
        stem = fov_name[: -len("_image")]
        candidates += [
            masks_dir / f"{stem}.ome.tif",
            masks_dir / f"{stem}.tif",
            masks_dir / f"{stem}.tiff",
        ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search by FOV prefix in masks/.
    prefix = fov_name.split(" ", 1)[0]
    for p in masks_dir.iterdir():
        if p.name.startswith(prefix):
            return p
    return None


def load_channels(fov_dir: Path) -> Tuple[List[str], np.ndarray]:
    """Read all channel TIFFs in a FOV dir, drop skips, canonicalize names.

    Returns (canonical_channel_names, raw of shape (C, H, W) float32). Channels
    that canonicalize to the same name (e.g. ``Foxp3`` and ``FOXP3``) are
    deduplicated by keeping the first occurrence.
    """
    channels: List[str] = []
    arrays: List[np.ndarray] = []
    seen: set = set()
    for p in sorted(fov_dir.iterdir()):
        if not p.is_file():
            continue
        # Strip both .ome.tif and .tif/.tiff suffixes.
        stem = p.name
        for suf in (".ome.tif", ".ome.tiff", ".tiff", ".tif"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        canon = canonicalize_channel(stem)
        if canon is None or canon in seen:
            continue
        try:
            arr = tff.imread(str(p))
        except Exception as e:
            logger.warning("Failed reading %s: %s", p, e)
            continue
        # Some FOVs ship multi-page TIFFs; keep page 0.
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr[0]
        if arr.ndim != 2:
            logger.warning("Unexpected shape for %s: %s", p, arr.shape)
            continue
        arrays.append(arr.astype(np.float32))
        channels.append(canon)
        seen.add(canon)

    if not channels:
        raise RuntimeError(f"No usable channels in {fov_dir}")
    raw = np.stack(arrays, axis=0)
    return channels, raw


def load_mask(mask_path: Path) -> np.ndarray:
    """Read a segmentation mask, normalize to (H, W) uint32."""
    arr = tff.imread(str(mask_path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unexpected mask shape {arr.shape} for {mask_path}"
            )
    if arr.ndim != 2:
        raise ValueError(f"Mask {mask_path} is not 2D after squeeze: {arr.shape}")
    # Round float masks (some MIBI masks ship as float32) and cast to uint32.
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    if (arr < 0).any():
        # int16 can carry signed pixels; clamp to 0.
        arr = np.where(arr < 0, 0, arr)
    return arr.astype(np.uint32)


def pick_nuclear_membrane(channel_names: List[str]) -> Tuple[str, str]:
    """Best-effort nuclear/membrane channel selection.

    These attrs are advisory for downstream tooling; the model itself
    doesn't read them. We pick from the present canonical channels.
    """
    nuc_pref = ["DAPI", "Hoechst", "DRAQ5", "DNA1", "HOECHST1", "H3", "Iridium"]
    mem_pref = ["E-cadherin", "PanCK", "CD45", "CD31", "SMA"]
    nuc = next((c for c in nuc_pref if c in channel_names), channel_names[0])
    mem = next((c for c in mem_pref if c in channel_names),
               channel_names[1] if len(channel_names) > 1 else channel_names[0])
    return nuc, mem


def write_fov(
    out_zarr: zarr.Group,
    key: str,
    *,
    raw: np.ndarray,
    mask: np.ndarray,
    channel_names: List[str],
    centroids: Dict[str, List[float]],
    tissue: str,
    modality: str,
    native_mpp: float,
    scale_applied: float,
    overwrite: bool,
) -> None:
    """Write one (dataset, fov) entry into the output zarr archive."""
    if key in out_zarr:
        if not overwrite:
            logger.info("skip existing key %s (use --overwrite to replace)", key)
            return
        del out_zarr[key]

    nuc, mem = pick_nuclear_membrane(channel_names)
    grp = out_zarr.create_group(
        key,
        attributes={
            "tissue": tissue,
            "modality": modality,
            "nuclear_channel": nuc,
            "membrane_channel": mem,
            "gold_native_mpp": native_mpp,
            "gold_resample_scale": scale_applied,
        },
    )
    pp = grp.create_group(
        "preprocessed",
        attributes={
            "channel_names": channel_names,
            "centroids": centroids,
            "mpp": TARGET_MPP,
            # Already in preprocessed coords (we resampled), so no further scale.
            "scale_factor": 1.0,
            "unique_cell_types": [PLACEHOLDER_CT],
        },
    )
    pp.create_array("raw", shape=raw.shape, dtype=raw.dtype)[...] = raw
    pp.create_array("mask", shape=mask.shape, dtype=mask.dtype)[...] = mask

    # cell_type_info: placeholder CT for every cell.
    cell_ids = np.array(sorted(int(k) for k in centroids), dtype=np.int64)
    cti = pp.create_group("cell_type_info")
    cti.create_array("cell_index", shape=cell_ids.shape, dtype=cell_ids.dtype)[
        ...
    ] = cell_ids
    cell_types = np.array([PLACEHOLDER_CT] * len(cell_ids), dtype="<U64")
    cti.create_array(
        "cell_type", shape=cell_types.shape, dtype=cell_types.dtype
    )[...] = cell_types


def ingest_dataset(
    gold_dir: Path,
    out_zarr: zarr.Group,
    dataset: str,
    *,
    native_mpp: float,
    strict: bool,
    overwrite: bool,
    limit: Optional[int],
    fov_filter: Optional[set],
) -> int:
    """Process every FOV in one gold subset. Returns count written."""
    dataset_dir = gold_dir / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Gold dataset directory not found: {dataset_dir}")
    tissue, modality = resolve_gold_metadata(dataset, strict=strict)
    fovs = list_fovs(dataset_dir)
    if limit is not None:
        fovs = fovs[:limit]
    if fov_filter is not None:
        fovs = [f for f in fovs if f.name in fov_filter]
    written = 0
    for fov_dir in fovs:
        mask_path = find_mask_for_fov(dataset_dir, fov_dir)
        if mask_path is None:
            logger.warning("[%s] no mask for %s, skipping", dataset, fov_dir.name)
            continue
        try:
            channel_names, raw_native = load_channels(fov_dir)
        except RuntimeError as e:
            logger.warning("[%s] %s", dataset, e)
            continue
        mask_native = load_mask(mask_path)
        if mask_native.shape[-2:] != raw_native.shape[-2:]:
            logger.warning(
                "[%s/%s] raw %s != mask %s, skipping",
                dataset, fov_dir.name, raw_native.shape, mask_native.shape,
            )
            continue

        # Single canonical preprocessing call: resample to TARGET_MPP +
        # normalize to [0, 1] per channel + compute centroids.
        out = preprocess_fov(
            raw=raw_native,
            mask=mask_native.astype(np.int32),
            native_mpp=native_mpp,
            channel_names=channel_names,
        )
        if not out.centroids:
            logger.warning("[%s/%s] mask has no cells, skipping", dataset, fov_dir.name)
            continue

        key = fov_key(dataset, fov_dir.name)
        write_fov(
            out_zarr,
            key,
            raw=out.raw,
            mask=out.mask,
            channel_names=out.channel_names,
            centroids=out.centroids,
            tissue=tissue,
            modality=modality,
            native_mpp=native_mpp,
            scale_applied=out.scale_factor,
            overwrite=overwrite,
        )
        logger.info(
            "[%s] wrote %s (C=%d, %dx%d, %d cells, scale=%.3f)",
            dataset, key, out.raw.shape[0], out.raw.shape[1], out.raw.shape[2],
            len(out.centroids), out.scale_factor,
        )
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--gold_dir",
        type=Path,
        default=Path("data/gold_standard/gold_standard_labelled"),
        help="Root of the Pan-M Gold-Standard layout (.../{dataset}/{fovs,masks}/).",
    )
    p.add_argument(
        "--output_zarr",
        type=Path,
        required=True,
        help="Path of the zarr v3 archive to create / append to.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(GOLD_DATASET_METADATA),
        choices=sorted(GOLD_DATASET_METADATA),
        help="Gold subsets to ingest (default: all 5).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Pass strict=True to resolve_gold_metadata — refuse non-direct "
        "canonicalizations (mibi_decidua → uterus, vectra_* → cycif).",
    )
    p.add_argument(
        "--mpp",
        action="append",
        default=[],
        metavar="DATASET=MPP",
        help="Override native MPP for a dataset, e.g. --mpp codex_colon=0.4. "
        "Defaults are baked in for the 5 known subsets.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing keys in the output zarr (default: skip).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N FOVs per dataset (for smoke tests).",
    )
    p.add_argument(
        "--fovs",
        nargs="+",
        default=None,
        help="Restrict processing to these FOV directory names (any dataset).",
    )
    p.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides: Dict[str, float] = {}
    for spec in args.mpp:
        try:
            k, v = spec.split("=", 1)
            overrides[k] = float(v)
        except ValueError:
            raise SystemExit(f"--mpp expects DATASET=FLOAT, got {spec!r}")
    mpp_table = {**DEFAULT_NATIVE_MPP, **overrides}

    if not args.gold_dir.exists():
        raise SystemExit(f"--gold_dir not found: {args.gold_dir}")

    out_zarr = zarr.open_group(str(args.output_zarr), mode="a")
    fov_filter = set(args.fovs) if args.fovs else None

    total = 0
    for ds in args.datasets:
        if ds not in mpp_table:
            raise SystemExit(
                f"No native MPP for {ds!r}; pass --mpp {ds}=<value>"
            )
        n = ingest_dataset(
            args.gold_dir,
            out_zarr,
            ds,
            native_mpp=mpp_table[ds],
            strict=args.strict,
            overwrite=args.overwrite,
            limit=args.limit,
            fov_filter=fov_filter,
        )
        logger.info("[%s] %d FOVs written", ds, n)
        total += n
    logger.info("Done: %d FOVs across %d datasets → %s",
                total, len(args.datasets), args.output_zarr)


if __name__ == "__main__":
    main()
