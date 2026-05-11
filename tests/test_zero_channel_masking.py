"""Unit tests for FOV-zero-channel masking in FullImageDataset.__getitem__.

Round-1 audit found ~3.4% of valid channels per MIBI/IMC FOV are listed in
channel_names but all-zero in raw across the entire FOV. These channels were
fed to the transformer as constant-zero tokens with a misleading marker
embedding prior. The fix masks them out at FOV-time. These tests verify the
mask is applied correctly via the `_zero_channel_cache` mechanism.
"""
import numpy as np


class _MinimalDataset:
    """Just enough of FullImageDataset to test the zero-mask logic in isolation."""

    def __init__(self, n_real_channels, max_channels=80):
        self.max_channels = max_channels
        self.n_real_channels = n_real_channels
        self._zero_channel_cache = {}


def _apply_zero_channel_mask(ds, ds_idx, n_real_channels, ch_idx, sample, mp_padded, vm_padded):
    """Replicates the masking block from dataset.py::__getitem__ for unit testing."""
    attn_mask = np.ones(ds.max_channels, dtype=bool)
    attn_mask[:n_real_channels] = ch_idx[:n_real_channels] == -1

    fov_zero_mask = ds._zero_channel_cache.get(ds_idx)
    if fov_zero_mask is not None:
        attn_mask[:n_real_channels] |= fov_zero_mask[:n_real_channels]

    clear_mask = ch_idx[:n_real_channels] == -1
    if fov_zero_mask is not None:
        clear_mask = clear_mask | fov_zero_mask[:n_real_channels]
    if clear_mask.any():
        clear_idx = np.where(clear_mask)[0]
        sample[clear_idx] = -1.0
        mp_padded[clear_idx] = 0
        vm_padded[clear_idx] = False

    return attn_mask, sample, mp_padded, vm_padded


def test_all_zero_channel_is_masked_in_attn_mask():
    ds = _MinimalDataset(n_real_channels=3)
    # Pretend channel 1 (middle) is all-zero across the FOV
    ds._zero_channel_cache[0] = np.array([False, True, False])

    ch_idx = np.zeros(ds.max_channels, dtype=np.int64)
    ch_idx[:3] = [10, 20, 30]  # all valid
    sample = np.zeros((ds.max_channels, 1, 4, 4), dtype=np.float32)
    mp = np.ones(ds.max_channels, dtype=np.float32)
    vm = np.ones(ds.max_channels, dtype=bool)

    attn_mask, sample, mp, vm = _apply_zero_channel_mask(ds, 0, 3, ch_idx, sample, mp, vm)
    # Channel 1 was all-zero -> attn_mask True (padded out)
    assert attn_mask[0] == False
    assert attn_mask[1] == True
    assert attn_mask[2] == False
    # Channels 3..max are padding (not real)
    assert attn_mask[3:].all()


def test_zero_channel_clears_sample_mp_validity():
    """When a channel is masked, sample becomes -1, mp 0, validity False."""
    ds = _MinimalDataset(n_real_channels=3)
    ds._zero_channel_cache[0] = np.array([False, True, False])

    ch_idx = np.zeros(ds.max_channels, dtype=np.int64)
    ch_idx[:3] = [10, 20, 30]
    sample = np.full((ds.max_channels, 1, 4, 4), 5.0, dtype=np.float32)
    mp = np.ones(ds.max_channels, dtype=np.float32)
    vm = np.ones(ds.max_channels, dtype=bool)

    _, sample, mp, vm = _apply_zero_channel_mask(ds, 0, 3, ch_idx, sample, mp, vm)
    # Channel 1 cleared
    assert (sample[1] == -1.0).all()
    assert mp[1] == 0
    assert not vm[1]
    # Channels 0 and 2 untouched
    assert (sample[0] == 5.0).all()
    assert mp[0] == 1
    assert vm[0]


def test_no_zero_cache_means_no_extra_masking():
    """If _zero_channel_cache lacks ds_idx (e.g., cache disabled), behavior
    should fall back to the legacy unknown-channel-only masking."""
    ds = _MinimalDataset(n_real_channels=3)
    # No entry in _zero_channel_cache for ds_idx=0

    ch_idx = np.zeros(ds.max_channels, dtype=np.int64)
    ch_idx[:3] = [10, -1, 30]  # channel 1 is unknown via ch_idx
    sample = np.full((ds.max_channels, 1, 4, 4), 5.0, dtype=np.float32)
    mp = np.ones(ds.max_channels, dtype=np.float32)
    vm = np.ones(ds.max_channels, dtype=bool)

    attn_mask, sample, mp, vm = _apply_zero_channel_mask(ds, 0, 3, ch_idx, sample, mp, vm)
    # Only the unknown channel is masked
    assert attn_mask[0] == False
    assert attn_mask[1] == True  # ch_idx == -1
    assert attn_mask[2] == False
    assert (sample[1] == -1.0).all()
    # Channel 0 and 2 untouched
    assert (sample[0] == 5.0).all()


def test_combined_unknown_and_zero_channel():
    """When a channel is BOTH unknown (ch_idx=-1) and zero, both paths fire."""
    ds = _MinimalDataset(n_real_channels=3)
    ds._zero_channel_cache[0] = np.array([False, False, True])  # ch 2 zero

    ch_idx = np.zeros(ds.max_channels, dtype=np.int64)
    ch_idx[:3] = [10, -1, 30]  # ch 1 unknown
    sample = np.full((ds.max_channels, 1, 4, 4), 5.0, dtype=np.float32)
    mp = np.ones(ds.max_channels, dtype=np.float32)
    vm = np.ones(ds.max_channels, dtype=bool)

    attn_mask, sample, mp, vm = _apply_zero_channel_mask(ds, 0, 3, ch_idx, sample, mp, vm)
    assert attn_mask[0] == False
    assert attn_mask[1] == True  # unknown
    assert attn_mask[2] == True  # zero
    assert (sample[1] == -1.0).all()
    assert (sample[2] == -1.0).all()
