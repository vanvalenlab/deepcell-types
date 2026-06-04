"""Utilities for model / data access.

The registries below pin the MD5 checksums of the paper-release
checkpoints; uploads to ``users.deepcell.org`` use the asset paths
constructed below (``models/<filename>``). Some baselines ship more
than one file (e.g. ``maps`` needs its ``_stats.npz`` companion;
``xgboost`` needs its ``.remap.json`` label-remap), so each baseline
entry is a list of ``(filename, md5)`` tuples.
"""

_latest = "2026-05-17"

# Main model checkpoints. Values are ``(asset_filename, md5)``.
_model_registry = {
    "2026-05-17": (
        "deepcell-types_2026-05-17.pt",
        "6089cf35a0ab7357f94dc3030156dc33",
    ),
}

# Baseline-model checkpoints. Values are lists of ``(asset_filename, md5)``.
# Single-file baselines have a one-element list; ``maps`` and ``xgboost``
# additionally ship a companion file required at inference.
_baseline_registry = {
    "cellsighter": [
        (
            "deepcell-types_baseline-cellsighter.pth",
            "d06f8aeef485e7c40590cc35da80944b",
        ),
    ],
    "maps": [
        (
            "deepcell-types_baseline-maps.pth",
            "d2d1930d438c014c226202b8b7fa4a65",
        ),
        (
            "deepcell-types_baseline-maps_stats.npz",
            "e3a54e5a64d5376231abf1022b001a41",
        ),
    ],
    "nimbus": [
        (
            "deepcell-types_baseline-nimbus.pt",
            "47916fbebc3a58d5bee96a9289d157aa",
        ),
    ],
    "xgboost": [
        (
            "deepcell-types_baseline-xgboost.json",
            "00d110cc0e9f429b3014845f05a13060",
        ),
        (
            "deepcell-types_baseline-xgboost.remap.json",
            "0d94609aa7127672111797df920920b7",
        ),
    ],
}


def download_model(*, version=None):
    """Download the deepcell-types model checkpoint for local use.

    Downloaded files land in ``$HOME/.deepcell/models``.

    Parameters
    ----------
    version : str, optional
        Which checkpoint version to download. Defaults to ``None``,
        which resolves to the most-recently-released version
        (``_latest`` in this module).

    Returns
    -------
    pathlib.Path
        Local path to the downloaded checkpoint.
    """
    from ._auth import fetch_data

    version = version if version is not None else _latest
    if version not in _model_registry:
        raise ValueError(
            f"Unknown model version {version!r}. "
            f"Known versions: {sorted(_model_registry)}."
        )
    filename, md5 = _model_registry[version]
    return fetch_data(f"models/{filename}", cache_subdir="models", file_hash=md5)


def list_model_versions():
    """Return the available pre-trained model versions, newest first.

    Returns
    -------
    list of str
        Version identifiers accepted by :func:`download_model`. The first
        element is the default (latest) version.
    """
    return sorted(_model_registry, reverse=True)


def download_baseline_checkpoint(name):
    """Download a baseline-model checkpoint (and any companion files).

    Some baselines ship more than one file:

    * ``maps``: ``.pth`` weights + ``_stats.npz`` feature-norm statistics.
    * ``xgboost``: ``.json`` booster + ``.remap.json`` label remap.

    Downloaded files land in ``$HOME/.deepcell/models``.

    Parameters
    ----------
    name : str
        Baseline identifier. One of ``cellsighter``, ``maps``,
        ``nimbus``, or ``xgboost``.

    Returns
    -------
    list[pathlib.Path]
        Local paths to every file downloaded for this baseline, in the
        order declared in ``_baseline_registry``.
    """
    from ._auth import fetch_data

    if name not in _baseline_registry:
        raise ValueError(
            f"Unknown baseline {name!r}. Known baselines: {sorted(_baseline_registry)}."
        )
    return [
        fetch_data(f"models/{filename}", cache_subdir="models", file_hash=md5)
        for filename, md5 in _baseline_registry[name]
    ]


_training_data_asset_key = "data/deepcell-types/public_data_v1.1.zip"


def download_training_data(*, extract=False):
    """Download the public training-data corpus for deepcell-types (v1.1).

    The compressed archive is downloaded to ``$HOME/.deepcell/data``. The
    asset is pinned to a single released version; older versions are not
    available through this helper.

    Note that the downloaded asset is a ``.zip`` and must be extracted
    before the contained ``tissuenet-*.zarr`` archive can be used as
    ``predict(zarr_path=...)`` / ``DEEPCELL_TYPES_ZARR_PATH``. Pass
    ``extract=True`` to unpack it (via :func:`extract_archive`, which
    rejects path-traversal members) and receive the extraction directory.

    Parameters
    ----------
    extract : bool, default=False
        If ``True``, extract the downloaded ``.zip`` next to itself and
        return the extraction directory instead of the ``.zip`` path.

    Returns
    -------
    pathlib.Path
        Local path to the downloaded ``.zip`` (or, when ``extract=True``,
        the directory it was extracted into).
    """
    from ._auth import extract_archive, fetch_data

    zip_path = fetch_data(_training_data_asset_key, cache_subdir="data")
    if extract:
        return extract_archive(zip_path)
    return zip_path
