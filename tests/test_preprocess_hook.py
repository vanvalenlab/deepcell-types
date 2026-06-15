import numpy as np
import pytest
import deepcell_types.dataset as dsmod
from deepcell_types.preprocessing import patch_generator
from deepcell_types.config import DCTConfig
from deepcell_types.preprocessing_ops import make_preprocessor, DEFAULT_CONFIG


def _toy():
    rng = np.random.default_rng(1)
    raw = rng.gamma(2.0, 30.0, size=(2, 40, 40)).astype(np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[10:20, 10:20] = 1
    mask[25:33, 25:33] = 2
    return raw, mask


def test_patch_generator_invokes_preprocess_with_chw_and_names():
    raw, mask = _toy()
    cfg = DCTConfig()
    seen = {}

    def hook(arr, names):
        seen["shape"] = arr.shape
        seen["names"] = names
        return np.zeros_like(arr)  # forces all-zero patches

    patches = list(
        patch_generator(
            raw,
            mask,
            mpp=0.5,
            dct_config=cfg,
            preprocess=hook,
            channel_names=["CD3", "DAPI"],
        )
    )
    assert len(seen["shape"]) == 3 and seen["shape"][0] == 2  # (C,H,W)
    assert seen["names"] == ["CD3", "DAPI"]
    assert all(np.all(p[0] == 0.0) for p in patches)  # hook output used


def test_patchdataset_forwards_preprocess(monkeypatch):
    raw, mask = _toy()
    cfg = DCTConfig()
    captured = {}

    def fake_patch_generator(
        raw_, mask_, mpp_, dct_config, preprocess=None, channel_names=None
    ):
        captured["preprocess"] = preprocess
        captured["channel_names"] = channel_names
        return iter(())

    monkeypatch.setattr(dsmod, "patch_generator", fake_patch_generator)

    def hook(arr, names):
        return arr

    ds = dsmod.PatchDataset(raw, mask, ["CD3", "DAPI"], 0.5, cfg, preprocess=hook)
    list(ds)  # triggers __iter__
    assert captured["preprocess"] is hook
    assert captured["channel_names"] == ds.channel_names_standard


def test_default_config_hook_equals_builtin_in_patch_generator():
    raw, mask = _toy()
    cfg = DCTConfig()
    baseline = [p[0].copy() for p in patch_generator(raw, mask, 0.5, dct_config=cfg)]
    hooked = [
        p[0].copy()
        for p in patch_generator(
            raw,
            mask,
            0.5,
            dct_config=cfg,
            preprocess=make_preprocessor(DEFAULT_CONFIG),
            channel_names=["CD3", "DAPI"],
        )
    ]
    assert len(baseline) == len(hooked) and len(baseline) > 0
    for b, h in zip(baseline, hooked):
        np.testing.assert_allclose(b, h, rtol=1e-5, atol=1e-6)


def test_preprocess_hook_nonfinite_output_rejected():
    raw, mask = _toy()
    cfg = DCTConfig()

    def hook(arr, names):
        out = np.zeros_like(arr)
        out[0, 0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        list(
            patch_generator(
                raw,
                mask,
                0.5,
                dct_config=cfg,
                preprocess=hook,
                channel_names=["CD3", "DAPI"],
            )
        )


def test_preprocess_hook_out_of_range_output_rejected():
    raw, mask = _toy()
    cfg = DCTConfig()

    def hook(arr, names):
        return np.full_like(arr, 5.0)  # outside the required [0, 1] contract

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        list(
            patch_generator(
                raw,
                mask,
                0.5,
                dct_config=cfg,
                preprocess=hook,
                channel_names=["CD3", "DAPI"],
            )
        )
