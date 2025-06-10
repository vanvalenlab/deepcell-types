"""Utilities for model/data access."""

_latest = "2025-06-09"
_model_registry = {
    # Original model version uploaded with preprint
    "specific_ct_v0.1": "e499da92509821161be88a47237960a9",
    # Versions released June 9th 2025. The public-data-only version is trained
    # only on the subset of data that is publicly available (for reproducibility).
    # Users are recommended to use the *non* public-data-only option.
    "2025-06-09": "19b669675c06816414e8677f542ff542",
    "2025-06-09_public-data-only": "19b669675c06816414e8677f542ff542",
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
    asset_key = f"models/deepcell-types_{version}.pt"

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

    fetch_data(asset_key, cache_subdir="data")
