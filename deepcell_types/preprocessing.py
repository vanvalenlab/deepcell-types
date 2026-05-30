"""Canonical preprocessing for raw multiplex imaging data → archive format.

Single source of truth for transforming an ingested raw FOV
(``(C, H, W)`` int/float intensity at a native MPP) into the format the
training pipeline consumes from ``preprocessed/raw`` + ``preprocessed/mask``:

1. Resample to ``TissueNetConfig.STANDARD_MPP_RESOLUTION`` (0.5 µm/pixel).
2. **Per-channel percentile threshold** at p99.9 of nonzero pixels —
   clip values above the per-FOV-per-channel p99.9 to that threshold.
3. **Per-channel min-max normalize** to ``[0, 1]``.
4. Cast mask to ``uint32``; compute centroids in resampled coordinates.

This recipe was recovered from
``hubmap-to-zarr@origin/deepcell-types:preprocess_for_training.py`` —
the script that originally produced the production archive's
``preprocessed/raw`` arrays. A snapshot test
(``tests/test_preprocessing.py::test_snapshot_against_production``)
confirms it reproduces production output within sub-pixel resampling
noise (max per-channel mean drift ~0.05 on
``HBM222_WQKC_382`` MIBI).

The pipeline is **per-FOV self-contained** — no archive-level statistics
are needed; each FOV normalizes against its own per-channel min/max
after the percentile clip. Reproducibility comes from the formula
itself being fixed.

## Public API

- ``preprocess_fov(raw, mask, native_mpp, channel_names) → PreprocessedFov``
- ``DEFAULT_PERCENTILE = 99.9``
- ``TARGET_MPP = 0.5``

The function is deterministic given the input — equal inputs produce
bit-equal outputs (modulo float precision noise from skimage.rescale,
which is the same noise present in the historical pipeline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from skimage.measure import regionprops
from skimage.transform import rescale
from tqdm import tqdm

logger = logging.getLogger(__name__)


TARGET_MPP: float = 0.5
"""Microns-per-pixel that all ingested FOVs are resampled to. Must equal
``TissueNetConfig.STANDARD_MPP_RESOLUTION`` — the dataloader assumes
this everywhere."""


DEFAULT_PERCENTILE: float = 99.9
"""Per-channel percentile (over nonzero pixels) used for the bright-spot
clip step. Matches the production-pipeline value recovered from
``preprocess_for_training.py`` on the deepcell-types branch."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedFov:
    """Result of ``preprocess_fov``.

    All arrays are at ``target_mpp`` (0.5 µm/pixel). The ``raw`` field is
    pre-multiplication-by-self-mask (the dataloader's
    ``extract_patch_from_zarr`` does that during patch extraction).
    """

    raw: np.ndarray            # (C, H', W') float32 in [0, 1]
    mask: np.ndarray           # (H', W') uint32
    centroids: Dict[str, List[float]]  # str(cell_id) -> [row, col]
    channel_names: List[str]
    target_mpp: float = TARGET_MPP
    native_mpp: float = 0.0
    scale_factor: float = 1.0  # native_mpp / target_mpp


# ---------------------------------------------------------------------------
# Internal helpers (each is independently testable)
# ---------------------------------------------------------------------------


def _resample(
    raw: np.ndarray, mask: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample raw (bilinear) and mask (nearest neighbor) by ``scale``.

    Operates on the (H, W, C) layout for raw to match
    ``deepcell-types:preprocess_for_training.py`` exactly. Returns raw
    in (C, H', W') float32 and mask in (H', W') uint32.
    """
    raw_hwc = np.transpose(raw, (1, 2, 0)).astype(np.float32, copy=False)
    if abs(scale - 1.0) > 0.01:
        raw_hwc = rescale(
            raw_hwc, scale, preserve_range=True, channel_axis=-1
        ).astype(np.float32)
        mask = rescale(
            mask, scale, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint32)
    else:
        mask = mask.astype(np.uint32, copy=False)
    return np.transpose(raw_hwc, (2, 0, 1)), mask


def _percentile_threshold(
    image_hwc: np.ndarray, percentile: float = DEFAULT_PERCENTILE
) -> np.ndarray:
    """Clip values above the per-channel p99.9 of nonzero pixels.

    Zeros are excluded from the percentile calculation via NaN masking
    (matches the canonical recipe; without this, a sparse channel's
    threshold would be dominated by background zeros). Channels that
    are entirely zero get threshold=inf (no clipping; min-max
    normalization later returns all zeros for the channel).
    """
    masked = np.where(image_hwc > 0, image_hwc, np.nan)
    thresholds = np.nanpercentile(masked, percentile, axis=(0, 1))  # (C,)
    thresholds = np.nan_to_num(thresholds, nan=np.inf)
    out = image_hwc.copy()
    np.minimum(out, thresholds, out=out)
    return out


def _min_max_normalize(image_hwc: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalize to [0, 1].

    For all-zero channels (range == 0), output is identically zero.
    """
    mn = np.min(image_hwc, axis=(0, 1), keepdims=True)
    mx = np.max(image_hwc, axis=(0, 1), keepdims=True)
    # ``np.ptp`` is being removed as a free function in future NumPy; compute
    # max - min directly. Same numerical behaviour.
    pt = mx - mn
    pt = np.where(pt == 0, 1.0, pt)
    return (image_hwc - mn) / pt


def _compute_centroids(mask: np.ndarray) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for prop in regionprops(mask):
        label = int(prop.label)
        if label == 0:
            continue
        out[str(label)] = [float(prop.centroid[0]), float(prop.centroid[1])]
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preprocess_fov(
    raw: np.ndarray,
    mask: np.ndarray,
    *,
    native_mpp: float,
    channel_names: Sequence[str],
    percentile: float = DEFAULT_PERCENTILE,
    target_mpp: float = TARGET_MPP,
) -> PreprocessedFov:
    """Canonical preprocessing for a single FOV.

    Parameters
    ----------
    raw : np.ndarray
        ``(C, H, W)`` numeric array of native intensity values.
    mask : np.ndarray
        ``(H, W)`` integer cell-id labels (``0`` = background).
    native_mpp : float
        Microns-per-pixel of the input.
    channel_names : Sequence[str]
        Sequence of length ``C`` of canonical marker names.
    percentile : float, optional
        Per-channel percentile used for the bright-spot clip. Default
        ``99.9`` matches the production recipe.
    target_mpp : float, optional
        Output MPP (default ``0.5`` = ``TARGET_MPP``).

    Returns
    -------
    PreprocessedFov
        ``raw``/``mask`` resampled to ``target_mpp``, with ``raw``
        normalized to ``[0, 1]`` per channel.

    Raises
    ------
    ValueError
        On shape / length / sign mismatches.
    """
    if raw.ndim != 3:
        raise ValueError(f"raw must be (C, H, W), got {raw.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H, W), got {mask.shape}")
    if raw.shape[1:] != mask.shape:
        raise ValueError(
            f"raw spatial {raw.shape[1:]} != mask spatial {mask.shape}"
        )
    if len(channel_names) != raw.shape[0]:
        raise ValueError(
            f"len(channel_names)={len(channel_names)} != raw.shape[0]={raw.shape[0]}"
        )
    if native_mpp <= 0.0:
        raise ValueError(f"native_mpp must be positive, got {native_mpp}")

    scale = native_mpp / target_mpp
    raw_chw, mask_r = _resample(raw, mask, scale)
    raw_hwc = np.transpose(raw_chw, (1, 2, 0))
    raw_hwc = _percentile_threshold(raw_hwc, percentile=percentile)
    raw_hwc = _min_max_normalize(raw_hwc)
    raw_norm = np.transpose(raw_hwc, (2, 0, 1)).astype(np.float32)
    centroids = _compute_centroids(mask_r)

    return PreprocessedFov(
        raw=raw_norm,
        mask=mask_r,
        centroids=centroids,
        channel_names=list(channel_names),
        target_mpp=target_mpp,
        native_mpp=native_mpp,
        scale_factor=scale,
    )


# ---------------------------------------------------------------------------
# Legacy patch generator (inference path: ``PatchDataset`` → ``predict``).
#
# Kept separate from ``preprocess_fov`` above because the published model
# was trained against this exact pipeline; bit-equivalence here is what
# the checkpoint expects. The two paths share a percentile-clip + min-max
# normalization structure but differ in detail (nonzero-only percentile
# via ``np.nonzero`` indexing vs. NaN-percentile vectorized). Do not
# unify without retraining.
# ---------------------------------------------------------------------------


def _normalize_per_channel(image):
    min_vals = np.min(image, axis=(0, 1), keepdims=True)
    ptp_vals = np.ptp(image, axis=(0, 1), keepdims=True)
    ptp_vals[ptp_vals == 0] = 1.0
    return (image - min_vals) / ptp_vals


def _percentile_threshold_nonzero(image, percentile=99.9):
    """Per-channel bright-spot clip using nonzero-pixel indexing.

    Mirrors deepcell-toolbox's reference recipe; differs from
    ``_percentile_threshold`` above (which uses NaN-percentile) in that
    it iterates channels and rebuilds the threshold from
    ``np.nonzero(...)``-indexed values. Behavior matches the recipe the
    published checkpoint was trained against — see
    https://github.com/vanvalenlab/deepcell-toolbox/blob/e8c1277/deepcell_toolbox/processing.py#L104
    """
    processed_image = np.zeros_like(image)
    for chan in range(image.shape[-1]):
        current_img = np.copy(image[..., chan])
        non_zero_vals = current_img[np.nonzero(current_img)]
        if len(non_zero_vals) > 0:
            img_max = np.percentile(non_zero_vals, percentile)
            threshold_mask = current_img > img_max
            current_img[threshold_mask] = img_max
            processed_image[..., chan] = current_img
    return processed_image


def _pad_cell(X, y, crop_size):
    delta = crop_size // 2
    X = np.pad(X, ((delta, delta), (delta, delta), (0, 0)))
    y = np.pad(y, ((delta, delta), (delta, delta)))
    return X, y


def _get_crop_box(centroid, delta):
    minr = int(centroid[0]) - delta
    maxr = int(centroid[0]) + delta
    minc = int(centroid[1]) - delta
    maxc = int(centroid[1]) + delta
    return np.array([minr, minc, maxr, maxc])


def _get_neighbor_masks(mask, cbox, cell_idx):
    """Binary masks of a cell and its neighbors within the crop window.

    Assumes the mask has already been padded; an unpadded mask will silently
    wrap at the borders.
    """
    minr, minc, maxr, maxc = cbox
    if not (np.issubdtype(mask.dtype, np.integer) and isinstance(cell_idx, int)):
        raise TypeError(
            f"mask must be an integer array and cell_idx must be int; "
            f"got mask.dtype={mask.dtype!r}, cell_idx={type(cell_idx).__name__}."
        )

    cell_view = mask[minr:maxr, minc:maxc]
    binmask_cell = (cell_view == cell_idx).astype(np.int32)
    binmask_neighbors = (cell_view != cell_idx).astype(np.int32) * (
        cell_view != 0
    ).astype(np.int32)
    return binmask_cell, binmask_neighbors


def patch_generator(raw, mask, mpp, dct_config):
    """Yield (raw_patch, mask_patch, cell_idx, orig_ct) for each cell.

    Output dtypes:
        raw_patch: float32 (C, H, W)
        mask_patch: float32 (H, W, 2) — [self_mask, neighbor_mask]
    """
    raw = np.transpose(raw, (1, 2, 0))  # (H, W, C)

    raw = rescale(
        raw,
        mpp / dct_config.STANDARD_MPP_RESOLUTION,
        preserve_range=True,
        channel_axis=-1,
    )

    mask = rescale(
        mask,
        mpp / dct_config.STANDARD_MPP_RESOLUTION,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.int32)

    raw = _percentile_threshold_nonzero(
        raw, percentile=dct_config.PERCENTILE_THRESHOLD
    )
    raw = _normalize_per_channel(raw)
    raw, mask = _pad_cell(raw, mask, dct_config.CROP_SIZE)

    props = regionprops(mask, cache=False)
    orig_ct = "Unknown"

    for prop in tqdm(props):
        idx = prop.label
        if idx == 0:
            continue

        delta = dct_config.CROP_SIZE // 2
        cbox = _get_crop_box(prop.centroid, delta)
        self_mask, neighbor_mask = _get_neighbor_masks(mask, cbox, prop.label)

        minr, minc, maxr, maxc = cbox
        raw_patch = raw[minr:maxr, minc:maxc, :]  # (H, W, C)
        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # (C, H, W)
        mask_patch = np.stack([self_mask, neighbor_mask], axis=-1)

        yield raw_patch, mask_patch.astype(np.float32), idx, orig_ct
