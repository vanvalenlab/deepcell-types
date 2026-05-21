"""Utilities for model/data access.

The asset hashes below are placeholders — fill them in at release time
once the corresponding ``.pt`` checkpoints are uploaded to
``users.deepcell.org``. While a hash is ``None``, ``fetch_data`` skips
the cached-MD5 check and re-downloads on every call (see
``_auth.fetch_data``); the verification fast-path becomes active as
soon as the real hash is set.
"""

_latest = "2025-06-09"

# Main deepcell-types model checkpoints. Keys map to a checkpoint
# version slug; the asset path is ``models/deepcell-types_{version}.pt``.
_model_registry = {
    _latest: None,  # MD5 placeholder; fill at release
}

# Baseline-model checkpoints. The asset path is
# ``models/deepcell-types_baseline-{name}.pt``.
_baseline_registry = {
    "cellsighter": None,
    "maps": None,
    "nimbus": None,
    "xgboost": None,
}


def download_model(*, version=None):
    """Download the deepcell-types model for local use.

    The model will be downloaded to ``$HOME/.deepcell/models``.

    Parameters
    ----------
    version : str, optional
       Which version of the model to download. Default is ``None``, which
       results in the latest (i.e. most-recently-released) version being
       downloaded.
    """
    from ._auth import fetch_data

    version = version if version is not None else _latest
    if version not in _model_registry:
        raise ValueError(
            f"Unknown model version {version!r}. "
            f"Known versions: {sorted(_model_registry)}."
        )
    asset_key = f"models/deepcell-types_{version}.pt"
    fetch_data(
        asset_key, cache_subdir="models", file_hash=_model_registry[version]
    )


def download_baseline_checkpoint(name):
    """Download a baseline-model checkpoint for local use.

    The checkpoint will be downloaded to ``$HOME/.deepcell/models``.

    Parameters
    ----------
    name : str
        Baseline identifier. One of ``cellsighter``, ``maps``, ``nimbus``,
        or ``xgboost``.
    """
    from ._auth import fetch_data

    if name not in _baseline_registry:
        raise ValueError(
            f"Unknown baseline {name!r}. "
            f"Known baselines: {sorted(_baseline_registry)}."
        )
    asset_key = f"models/deepcell-types_baseline-{name}.pt"
    fetch_data(
        asset_key, cache_subdir="models", file_hash=_baseline_registry[name]
    )


def download_training_data(*, version=None):
    """Download the complete corpus of training data for the deepcell-types model.

    The compressed dataset will be downloaded to ``$HOME/.deepcell/data``.

    Parameters
    ----------
    version : str, optional
       Which version of the training data to download. Default is `None`, which results
       in the latest (i.e. most-recently-released) version being downloaded.
    """
    from ._auth import fetch_data


    asset_key = "data/deepcell-types/public_data_v1.1.zip"

    fetch_data(asset_key, cache_subdir="data")
