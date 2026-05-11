"""Tests for ``deepcelltypes/preprocessing.py``.

The unit tests exercise the public API + private steps with synthetic input.
The production-snapshot test reads a known FOV from the production archive,
runs it through ``preprocess_fov``, and asserts the output matches
``preprocessed/raw`` within sub-pixel resampling noise. The recipe is the
canonical one recovered from
``hubmap-to-zarr@origin/deepcell-types:preprocess_for_training.py``, so the
match should be tight (max per-channel mean drift ~0.05).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from deepcell_types.preprocessing import (
    DEFAULT_PERCENTILE,
    PreprocessedFov,
    TARGET_MPP,
    _min_max_normalize,
    _percentile_threshold,
    preprocess_fov,
)


# ---------------------------------------------------------------------------
# Internal step tests — each formula step independently verified
# ---------------------------------------------------------------------------


def test_percentile_threshold_clips_above_p99_9():
    # 10000 pixels uniform in [1, 10], one outlier at 1000 → p99.9 of
    # 10001 nonzero values ≈ 10 (only ~10 pixels above p99.9, so the
    # single outlier doesn't dominate).
    rng = np.random.default_rng(0)
    arr = rng.uniform(1.0, 10.0, size=(100, 100, 1)).astype(np.float32)
    arr[0, 0, 0] = 1000.0
    out = _percentile_threshold(arr, percentile=99.9)
    # Outlier is clipped down to ~p99.9 (≤10)
    assert out[0, 0, 0] < 11.0
    # Non-outliers untouched
    assert out[50, 50, 0] == arr[50, 50, 0]


def test_percentile_threshold_handles_all_zero_channel():
    arr = np.zeros((10, 10, 2), dtype=np.float32)
    arr[..., 0] = 5.0  # channel 0 has signal
    # channel 1 stays all zero
    out = _percentile_threshold(arr, percentile=99.9)
    assert (out[..., 1] == 0.0).all()


def test_percentile_threshold_excludes_zeros_from_calculation():
    # If zeros were included, p99.9 would be near 0 and everything clipped.
    arr = np.zeros((100, 100, 1), dtype=np.float32)
    arr[:5, :5, 0] = 100.0   # 25 nonzero pixels at value 100
    out = _percentile_threshold(arr, percentile=99.9)
    # Threshold from nonzero pixels = 100, so 100s should remain 100
    assert (out[:5, :5, 0] == 100.0).all()
    # And the zeros stay zero
    assert (out[5:, 5:, 0] == 0.0).all()


def test_min_max_normalize_to_unit_range():
    arr = np.array([[[2.0, 5.0]],
                    [[4.0, 5.0]],
                    [[6.0, 5.0]]], dtype=np.float32)  # (3, 1, 2)
    out = _min_max_normalize(arr)
    # Channel 0: min=2, max=6 → normalized to {0, 0.5, 1}
    np.testing.assert_allclose(out[..., 0].ravel(), [0.0, 0.5, 1.0])
    # Channel 1: ptp=0 → output identically zero
    np.testing.assert_allclose(out[..., 1], 0.0)


# ---------------------------------------------------------------------------
# preprocess_fov: contract checks
# ---------------------------------------------------------------------------


def _fixture_raw_mask(C=3, H=20, W=20):
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 100, size=(C, H, W)).astype(np.float32)
    raw[:, :2, :] = 0  # add zero pixels
    mask = np.zeros((H, W), dtype=np.int32)
    mask[5:10, 5:10] = 1
    mask[12:15, 12:15] = 2
    return raw, mask


def test_preprocess_fov_output_in_unit_range_and_correct_dtypes():
    raw, mask = _fixture_raw_mask()
    out = preprocess_fov(
        raw, mask, native_mpp=0.5, channel_names=["A", "B", "C"]
    )
    assert isinstance(out, PreprocessedFov)
    assert out.raw.dtype == np.float32
    assert out.mask.dtype == np.uint32
    assert out.raw.shape == (3, 20, 20)
    assert out.mask.shape == (20, 20)
    assert out.raw.min() >= 0.0
    assert out.raw.max() <= 1.0


def test_preprocess_fov_produces_max_one_per_channel():
    """The recipe normalizes to [0, 1] per channel, so each channel's
    max should be exactly 1.0 (matches production)."""
    raw, mask = _fixture_raw_mask()
    out = preprocess_fov(
        raw, mask, native_mpp=0.5, channel_names=["A", "B", "C"]
    )
    perch_max = out.raw.max(axis=(1, 2))
    np.testing.assert_allclose(perch_max, 1.0, atol=1e-6)


def test_preprocess_fov_resamples_to_target_mpp():
    raw, mask = _fixture_raw_mask()
    out = preprocess_fov(
        raw, mask, native_mpp=1.0, channel_names=["x", "y", "z"]
    )
    # native=1.0, target=0.5 → upsample 2x
    assert out.scale_factor == pytest.approx(2.0)
    assert out.raw.shape[1] == 40
    assert out.mask.shape[0] == 40


def test_preprocess_fov_no_resample_when_mpp_matches():
    raw, mask = _fixture_raw_mask()
    out = preprocess_fov(
        raw, mask, native_mpp=0.5, channel_names=["a", "b", "c"]
    )
    assert out.scale_factor == pytest.approx(1.0)
    assert out.raw.shape == raw.shape
    assert out.mask.shape == mask.shape


def test_preprocess_fov_centroids_match_resampled_mask():
    raw, mask = _fixture_raw_mask()
    out = preprocess_fov(
        raw, mask, native_mpp=0.5, channel_names=["a", "b", "c"]
    )
    assert "1" in out.centroids
    r, c = out.centroids["1"]
    # Cell 1 occupies rows 5..10, cols 5..10
    assert 6.0 <= r <= 8.0
    assert 6.0 <= c <= 8.0


def test_preprocess_fov_handles_all_zero_channel():
    raw = np.zeros((2, 8, 8), dtype=np.float32)
    raw[0] = 5.0  # channel 0 has signal
    # channel 1 is all zero
    mask = np.zeros((8, 8), dtype=np.int32)
    out = preprocess_fov(
        raw, mask, native_mpp=0.5, channel_names=["A", "B"]
    )
    # Channel 0: signal (constant 5) gets clipped + min-max → 0
    # because min == max → ptp=0 → output is 0
    assert (out.raw[1] == 0.0).all()


def test_preprocess_fov_rejects_shape_mismatches():
    raw = np.zeros((3, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.int32)

    with pytest.raises(ValueError, match="raw must be"):
        preprocess_fov(
            raw[0], mask, native_mpp=0.5, channel_names=["x"]
        )
    with pytest.raises(ValueError, match="mask must be"):
        preprocess_fov(
            raw, mask[None], native_mpp=0.5, channel_names=["a", "b", "c"]
        )
    with pytest.raises(ValueError, match="raw spatial"):
        preprocess_fov(
            raw, np.zeros((4, 4), dtype=np.int32), native_mpp=0.5,
            channel_names=["a", "b", "c"]
        )
    with pytest.raises(ValueError, match=r"len\(channel_names\)"):
        preprocess_fov(
            raw, mask, native_mpp=0.5, channel_names=["only-two", "names"]
        )
    with pytest.raises(ValueError, match="native_mpp must be positive"):
        preprocess_fov(
            raw, mask, native_mpp=0.0, channel_names=["a", "b", "c"]
        )


def test_preprocess_fov_deterministic():
    raw, mask = _fixture_raw_mask()
    out1 = preprocess_fov(
        raw, mask, native_mpp=0.7, channel_names=["a", "b", "c"]
    )
    out2 = preprocess_fov(
        raw, mask, native_mpp=0.7, channel_names=["a", "b", "c"]
    )
    np.testing.assert_array_equal(out1.raw, out2.raw)
    np.testing.assert_array_equal(out1.mask, out2.mask)
    assert out1.centroids == out2.centroids


# ---------------------------------------------------------------------------
# Production snapshot — reproduces preprocessed/raw within resampling noise
# ---------------------------------------------------------------------------


PRODUCTION_ARCHIVE = os.environ.get(
    "PRODUCTION_ARCHIVE_PATH",
    "/data/xwang3/tissuenet-caitlin-labels.zarr/tissuenet-caitlin-labels.zarr",
)
SNAPSHOT_DATASET = os.environ.get("PRODUCTION_SNAPSHOT_DATASET", "HBM222_WQKC_382")


@pytest.mark.skipif(
    not Path(PRODUCTION_ARCHIVE).exists(),
    reason=f"Production archive not available at {PRODUCTION_ARCHIVE}",
)
def test_snapshot_against_production():
    """Run preprocess_fov on a real production raw FOV, assert close
    match against ``preprocessed/raw``.

    The recipe in this module is the canonical one used to produce the
    production archive, so the difference should be sub-pixel
    resampling noise (skimage.rescale boundary effects). Empirically:
    max per-channel mean drift on HBM222 ≈ 0.046, so we assert < 0.06.
    """
    import zarr
    z = zarr.open(PRODUCTION_ARCHIVE, mode="r")
    if SNAPSHOT_DATASET not in z:
        pytest.skip(f"{SNAPSHOT_DATASET} not in production archive")
    g = z[SNAPSHOT_DATASET]
    if "image" not in g or "preprocessed" not in g:
        pytest.skip(f"{SNAPSHOT_DATASET} lacks image or preprocessed")

    img = g["image"]
    pp = g["preprocessed/raw"]
    native_mpp = float(img.attrs.get("mpp", 0.5))
    ch_raw = list(
        img.attrs.get("standardized_channels", img.attrs.get("channels", []))
    )
    ch_pp = list(g["preprocessed"].attrs.get("channel_names", []))
    if not ch_raw or not ch_pp:
        pytest.skip(f"{SNAPSHOT_DATASET} missing channel attrs")

    indices = [(ch_raw.index(c), ci) for ci, c in enumerate(ch_pp) if c in ch_raw]
    if not indices:
        pytest.skip("No overlapping channels")
    ri = [r for r, _ in indices]
    pi = [p for _, p in indices]
    selected_raw = np.asarray(img[ri], dtype=np.float32)
    selected_pp = np.asarray(pp[pi], dtype=np.float32)
    selected_channels = [ch_raw[r] for r in ri]

    fake_mask = np.zeros(selected_raw.shape[1:], dtype=np.int32)
    out = preprocess_fov(
        selected_raw, fake_mask,
        native_mpp=native_mpp, channel_names=selected_channels,
    )

    # Bound 1: shape matches within ±1 pixel.
    assert out.raw.shape[0] == selected_pp.shape[0]
    for axis in (1, 2):
        assert abs(out.raw.shape[axis] - selected_pp.shape[axis]) <= 1, (
            f"axis-{axis} shape diff > 1: ours={out.raw.shape}, "
            f"prod={selected_pp.shape}"
        )

    h = min(out.raw.shape[1], selected_pp.shape[1])
    w = min(out.raw.shape[2], selected_pp.shape[2])
    ours = out.raw[:, :h, :w]
    prod = selected_pp[:, :h, :w]

    # Bound 2: per-channel max ≈ 1.0 (recipe enforces this exactly).
    perch_max = ours.max(axis=(1, 2))
    np.testing.assert_allclose(perch_max, 1.0, atol=1e-5)

    # Bound 3: per-channel mean within 0.06 of production (empirical
    # max on HBM222 is 0.046; 0.06 buffer is for cross-FOV drift).
    ours_mean = ours.mean(axis=(1, 2))
    prod_mean = prod.mean(axis=(1, 2))
    diffs = np.abs(ours_mean - prod_mean)
    nonzero_prod = prod_mean > 1e-6
    if nonzero_prod.any():
        max_diff = diffs[nonzero_prod].max()
        assert max_diff < 0.06, (
            f"per-channel mean drift exceeded 0.06: max={max_diff:.4f}; "
            f"channels affected: "
            f"{[selected_channels[i] for i in np.where(diffs > 0.06)[0]][:5]}"
        )

    # Bound 4: ≥85% of pixels match within 1e-2 (resampling boundary).
    pixel_match = (np.abs(ours - prod) < 1e-2).mean()
    assert pixel_match > 0.85, (
        f"only {pixel_match*100:.1f}% of pixels match within 1e-2"
    )
