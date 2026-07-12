"""Utilities for model / data access.

Model, baseline, and training-data downloads are delegated to the shared
``deepcell-auth`` client, whose bundled ``asset_manifest.yaml`` is the single
source of truth for asset keys and integrity hashes. The functions here are
thin adapters that preserve this package's public API (return a ``Path`` /
``list[Path]`` and keep the friendly Nimbus pointer) on top of
``deepcell_auth.download_deepcell_types_*``. Because those functions return
``None``, the downloaded file is recovered by globbing the ``~/.deepcell``
cache (mirroring ``torch-mesmer``).

The version / baseline-name lists below are mirrored statically from the
manifest; ``tests/test_download_delegation.py`` asserts they stay in sync.
"""

from pathlib import Path

__all__ = [
    "download_model",
    "download_baseline_checkpoint",
    "download_training_data",
    "list_model_versions",
    "list_baseline_names",
    "list_supported_markers",
    "list_supported_cell_types",
    "resolve_supported_marker",
]

# Mirrors deepcell-auth's asset_manifest.yaml (models.deepcell-types /
# models.deepcell-types-baselines). Kept static to avoid a runtime manifest
# read in production; drift is caught by tests/test_download_delegation.py.
_DEFAULT_MODEL_VERSION = "2026-06-15"
_MODEL_VERSIONS = ("2026-06-15", "2026-06-23")
_BASELINE_NAMES = ("cellsighter", "maps", "xgboost")


def download_model(*, version=None):
    """Download the deepcell-types model checkpoint for local use.

    Delegates to ``deepcell_auth`` and returns the local path. Downloaded
    files land in ``$HOME/.deepcell/models``.

    Parameters
    ----------
    version : str, optional
        Which checkpoint version to download. Defaults to ``None``, which
        resolves to the most-recently-released version.

    Returns
    -------
    pathlib.Path
        Local path to the downloaded checkpoint.
    """
    from deepcell_auth import download_deepcell_types_model

    version = version if version is not None else _DEFAULT_MODEL_VERSION
    download_deepcell_types_model(version)
    models_dir = Path.home() / ".deepcell" / "models"
    return sorted(models_dir.glob(f"deepcell-types_{version}_*.pt"))[-1]


def list_model_versions():
    """Return the available pre-trained model versions, default first.

    Returns
    -------
    list of str
        Version identifiers accepted by :func:`download_model`. The first
        element is the default (``download_model()`` with no argument).
    """
    return list(_MODEL_VERSIONS)


def download_baseline_checkpoint(name):
    """Download a baseline-model checkpoint (and any companion files).

    Some baselines ship more than one file:

    * ``maps``: ``.pth`` weights + ``_stats.npz`` feature-norm statistics.
    * ``xgboost``: ``.json`` booster + ``.remap.json`` label remap.

    Downloaded files land in ``$HOME/.deepcell/models``.

    Parameters
    ----------
    name : str
        Baseline identifier. One of ``cellsighter``, ``maps``, or
        ``xgboost``. ``nimbus`` is not served here (its weights are
        distributed upstream); requesting it raises with a pointer to the
        official source.

    Returns
    -------
    list[pathlib.Path]
        Local paths to every file downloaded for this baseline.
    """
    from deepcell_auth import download_deepcell_types_baseline

    if name == "nimbus":
        raise ValueError(
            "The Nimbus baseline is inference-only and its pretrained weights "
            "are distributed upstream, not re-hosted by this project. Install "
            "the official library (`pip install -e '.[baseline-nimbus]'`), which "
            "downloads the weights automatically; see "
            "https://github.com/angelolab/Nimbus-Inference."
        )
    return download_deepcell_types_baseline(name)


def list_baseline_names():
    """Return the available baseline identifiers, sorted.

    Returns
    -------
    list of str
        Names accepted by :func:`download_baseline_checkpoint`.
    """
    return sorted(_BASELINE_NAMES)


def list_supported_markers(*, zarr_path=None):
    """Return the canonical marker names in the active registry, sorted.

    Lets a user pre-flight-check whether their marker panel overlaps the
    model's registry before downloading a checkpoint or running inference.
    Use :func:`resolve_supported_marker` to check acquisition names that may
    be aliases or differ in capitalization.
    Reads the packaged ``vocab.json`` snapshot via :class:`.DCTConfig`
    (no archive or checkpoint required); pass ``zarr_path`` to inspect an
    archive's registry instead.

    Parameters
    ----------
    zarr_path : str or Path, optional
        Forwarded to :class:`.DCTConfig`. If ``None`` (default), resolves the
        ``DEEPCELL_TYPES_ZARR_PATH`` environment variable and falls back to
        the packaged ``vocab.json``.

    Returns
    -------
    list of str
        Recognized marker names, sorted.
    """
    from ..config import DCTConfig

    return sorted(DCTConfig(zarr_path=zarr_path).marker2idx)


def resolve_supported_marker(marker, *, zarr_path=None):
    """Resolve a marker name or alias to its canonical registry name.

    Resolution matches inference behavior, including configured aliases and
    case-insensitive names.

    Parameters
    ----------
    marker : str
        Marker/channel name to resolve.
    zarr_path : str or Path, optional
        Forwarded to :class:`.DCTConfig`. If ``None`` (default), resolves the
        ``DEEPCELL_TYPES_ZARR_PATH`` environment variable and falls back to
        the packaged ``vocab.json``.

    Returns
    -------
    str or None
        Canonical marker name, or ``None`` when the marker is unsupported.
    """
    from ..config import DCTConfig

    return DCTConfig(zarr_path=zarr_path).resolve_channel_name(marker)


def list_supported_cell_types(*, zarr_path=None):
    """Return the cell-type names the packaged registry recognizes, sorted.

    The ``list``-returning counterpart to :func:`list_supported_markers`; see
    its docstring for the pre-flight-check motivation.

    Parameters
    ----------
    zarr_path : str or Path, optional
        Forwarded to :class:`.DCTConfig`. If ``None`` (default), resolves the
        ``DEEPCELL_TYPES_ZARR_PATH`` environment variable and falls back to
        the packaged ``vocab.json``.

    Returns
    -------
    list of str
        Recognized cell-type names, sorted.
    """
    from ..config import DCTConfig

    return sorted(DCTConfig(zarr_path=zarr_path).ct2idx)


def download_training_data(*, extract=False):
    """Download the public training-data corpus for deepcell-types (v1.1).

    Delegates the download to ``deepcell_auth``; the compressed archive lands
    in ``$HOME/.deepcell/data``. The asset is a ``.zip`` that must be
    extracted before the contained ``tissuenet-*.zarr`` archive can be used as
    ``predict(zarr_path=...)`` / ``DEEPCELL_TYPES_ZARR_PATH``.

    Parameters
    ----------
    extract : bool, default=False
        If ``True``, extract the downloaded ``.zip`` next to itself (via the
        path-traversal-safe :func:`~deepcell_types.utils._archive.extract_archive`)
        and return the extraction directory instead of the ``.zip`` path.

    Returns
    -------
    pathlib.Path
        Local path to the downloaded ``.zip`` (or, when ``extract=True``,
        the directory it was extracted into).
    """
    from deepcell_auth import download_deepcell_types_data

    from ._archive import extract_archive

    download_deepcell_types_data()
    data_dir = Path.home() / ".deepcell" / "data"
    zip_path = sorted(data_dir.glob("public_data*.zip"))[-1]
    if extract:
        return extract_archive(zip_path)
    return zip_path
