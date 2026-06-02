import numpy as np
from deepcell_types.preprocessing import patch_generator
from deepcell_types.config import DCTConfig


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
