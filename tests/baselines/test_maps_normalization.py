import numpy as np

from deepcell_types.baselines.maps.run import normalize_features


def test_maps_normalization_defaults_to_dct_zscore():
    # DCT marker means are already in [0, 1], while cellSize is a raw pixel count.
    # The default must therefore normalize both columns before the /255 factor so
    # the appended cellSize feature does not dominate by scale alone.
    X = np.array(
        [
            [0.0, 100.0],
            [1.0, 300.0],
        ],
        dtype=np.float32,
    )

    X_norm, mean, std = normalize_features(X)

    assert np.allclose(mean, np.array([0.5, 200.0], dtype=np.float32))
    assert np.allclose(std, np.array([0.5, 100.0], dtype=np.float32))
    assert np.allclose(X_norm, ((X - mean) / std) / 255.0)


def test_maps_no_znorm_keeps_div255_only_path():
    X = np.array(
        [
            [0.0, 100.0],
            [1.0, 300.0],
        ],
        dtype=np.float32,
    )
    mean = np.array([0.5, 200.0], dtype=np.float32)
    std = np.array([0.5, 100.0], dtype=np.float32)

    X_norm, returned_mean, returned_std = normalize_features(
        X, mean=mean, std=std, znorm=False
    )

    assert np.allclose(X_norm, X / 255.0)
    assert np.allclose(returned_mean, mean)
    assert np.allclose(returned_std, std)
