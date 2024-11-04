"""Utilities for model/data access."""

_latest = "v0.1"
_model_registry = {
    "v0.1": "d98d7333dd00d37608180892a5e88d54"
}


def download_model(*, version=None):
    """Download the deepcell-types model for local use.

    The model will be downloaded to ``$HOME/.deepcell/models``.

    Parameters
    ----------
    version : str, optional
       Which version of the model to download. Default is `None`, which results
       in the latest (i.e. most-recently-released) version being downloaded.
    """
    from ._auth import fetch_data


    version = version if version is not None else _latest
    asset_key = f"models/deepcell-types_combined_ct_{version}.pt"

    fetch_data(
        asset_key, cache_subdir="models", file_hash=_model_registry.get(version)
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


    asset_key = f"data/deepcell-types/deepcell-types-data.tar.gz"

    fetch_data(
        asset_key, cache_subdir="data", file_hash=_dataset_registry.get(version)
    )
