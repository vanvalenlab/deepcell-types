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