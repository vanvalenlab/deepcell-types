import numpy as np
import pytest
from deepcell_types.preprocessing_ops import (
    apply_config,
    DEFAULT_CONFIG,
    make_preprocessor,
)
from deepcell_types.preprocessing import (
    _percentile_threshold_nonzero,
    _normalize_per_channel,
)


def _fov(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 50.0, size=(3, 24, 24)).astype(np.float32)
    x[1, :5, :5] = 5000.0  # bright outlier blob
    return x, ["CD3", "CD8", "DAPI"]


def test_default_config_matches_builtin_inference_path():
    raw, names = _fov()
    out = apply_config(raw, names, DEFAULT_CONFIG)  # (C,H,W)
    hwc = np.transpose(raw, (1, 2, 0))
    ref = _normalize_per_channel(_percentile_threshold_nonzero(hwc, percentile=99.9))
    ref = np.transpose(ref, (2, 0, 1))
    assert out.shape == raw.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_output_is_in_unit_range():
    raw, names = _fov()
    out = apply_config(raw, names, DEFAULT_CONFIG)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_channel_drop_zeros_named_channel():
    raw, names = _fov()
    out = apply_config(
        raw,
        names,
        [{"op": "channel_drop", "names": ["DAPI"]}, {"op": "min_max_normalize"}],
    )
    assert np.all(out[2] == 0.0)


@pytest.mark.parametrize(
    "step",
    [
        {"op": "background_subtract_per_channel", "names": ["TYPO"]},
        {"op": "channel_drop", "names": ["TYPO"]},
        {"op": "channel_weight", "weights": {"TYPO": 0.5}},
    ],
)
def test_named_ops_reject_unknown_channels(step):
    with pytest.raises(ValueError, match="unknown preprocessing channel"):
        apply_config(np.ones((1, 2, 2)), ["CD3"], [step])


def test_channel_weight_after_normalize_scales():
    raw, names = _fov()
    cfg = [
        {"op": "min_max_normalize"},
        {"op": "channel_weight", "weights": {"CD8": 0.25}},
    ]
    out = apply_config(raw, names, cfg)
    assert out[1].max() <= 0.25 + 1e-6


def test_all_table_ops_are_implemented():
    raw, names = _fov()
    for op in [
        {"op": "clip_percentile", "p": 99.0},
        {"op": "log1p"},
        {"op": "background_subtract", "value": 10.0},
        {"op": "background_subtract_per_channel", "p": 25.0},
        {"op": "gamma", "g": 0.5},
        {"op": "denoise", "kind": "median", "size": 3},
        {"op": "hot_pixel_removal", "z": 5.0},
    ]:
        apply_config(raw, names, [op, {"op": "min_max_normalize"}])  # must not raise


def test_background_subtract_per_channel_removes_pedestal():
    # CD3 sits on a high background pedestal (+2000); CD8 has a low floor.
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 30.0, size=(2, 32, 32)).astype(np.float32)
    x[0] += 2000.0  # CD3 pedestal
    names = ["CD3", "CD8"]
    out = apply_config(x, names, [{"op": "background_subtract_per_channel", "p": 25.0}])
    # the pedestal channel loses ~its floor; the clean channel keeps most signal
    assert out[0].min() == 0.0
    assert float(out[0].mean()) < float(x[0].mean()) - 1500.0
    # subtraction is bounded by the channel's own p25, never negative
    assert out.min() >= 0.0


def test_background_subtract_per_channel_can_target_named_channel():
    rng = np.random.default_rng(2)
    x = rng.gamma(2.0, 30.0, size=(2, 32, 32)).astype(np.float32)
    x += 1000.0  # both channels on a pedestal
    names = ["CD15", "CD8"]
    out = apply_config(
        x, names,
        [{"op": "background_subtract_per_channel", "p": 50.0, "names": ["CD15"]}],
    )
    # only CD15 is corrected; CD8 is untouched
    np.testing.assert_allclose(out[1], x[1])
    assert float(out[0].mean()) < float(x[0].mean())


def test_unknown_op_raises():
    raw, names = _fov()
    with pytest.raises(ValueError, match="unknown op"):
        apply_config(raw, names, [{"op": "nope"}])


def test_make_preprocessor_returns_hook():
    raw, names = _fov()
    hook = make_preprocessor(DEFAULT_CONFIG)
    np.testing.assert_allclose(
        hook(raw, names), apply_config(raw, names, DEFAULT_CONFIG)
    )


def test_public_exports():
    import deepcell_types as dct

    assert callable(dct.make_preprocessor)
    assert callable(dct.apply_config)
    assert isinstance(dct.DEFAULT_CONFIG, list)
