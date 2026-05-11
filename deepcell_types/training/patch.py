"""Per-cell patch extraction helpers.

Split out of ``training/config.py`` so loaders can depend on these without
pulling in the full ``TissueNetConfig`` class. ``config.py`` keeps a re-export
at the bottom for backward compatibility with external callers.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize

logger = logging.getLogger(__name__)


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
