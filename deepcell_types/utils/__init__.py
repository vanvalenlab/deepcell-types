"""Utilities for model / data access.

Model, baseline, and training-data downloads are delegated to the shared
``deepcell-auth`` client, whose bundled ``asset_manifest.yaml`` is the single
source of truth for asset keys and integrity hashes. The functions here are
thin adapters that keep this package's public API: the ``ValueError`` on an
unknown identifier, the pointer to Nimbus' upstream weights, and the
``Path`` / ``list[Path]`` return types.

The version and baseline names below are mirrored statically from that
manifest so an unknown identifier is rejected without a manifest read;
``tests/test_download_delegation.py`` fails if they drift.
"""

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

_DEFAULT_MODEL_VERSION = "2026-06-15"

# Released residual-MLP checkpoints, sharing the current vocabulary (51 cell
# types, 278 markers) and loading via stock ``predict.py`` (the resMLP head is
# auto-detected):
#   - ``2026-06-15`` (default): the from-scratch resMLP (paper Fig-3c headline).
#   - ``2026-06-23-ptft``: the pretrain -> finetune (SSL) resMLP arm.
_MODEL_VERSIONS = ("2026-06-15", "2026-06-23-ptft")

# Also in the manifest, but deliberately not offered here: the 2025 CLIP
# checkpoints predate the current architecture and cannot be loaded by this
# code. They stay served for reproducibility with the matching historical
# commit, so ``download_model`` rejects them rather than handing back a
# checkpoint that fails at ``load_state_dict``.
_LEGACY_MODEL_VERSIONS = ("2025-06-09", "2025-06-09_public-data-only")

# All three baselines are served as a single ``.tar.gz`` bundle, one
# subdirectory per baseline, holding each one's weights plus any companion file
# required at inference (maps -> ``_stats.npz``; xgboost -> ``.remap.json``).
# Requesting any one baseline therefore downloads the whole bundle; the other
# two are then served from the cache without a second transfer.
#
# Nimbus is intentionally absent: it is inference-only with pretrained weights
# produced and distributed upstream (angelolab/Nimbus-Inference), which this
# project does not redistribute. ``download_baseline_checkpoint("nimbus")``
# raises with a pointer to the official source instead (see below).
_BASELINE_NAMES = ("cellsighter", "maps", "xgboost")

# Nimbus pretrained weights are distributed upstream (via the
# ``nimbus-inference`` library / Hugging Face Hub), not re-hosted here.
_NIMBUS_UPSTREAM_URL = "https://github.com/angelolab/Nimbus-Inference"


def download_model(*, version=None):
    """Download the deepcell-types model checkpoint for local use.

    Downloaded files land in ``$HOME/.deepcell/models``.

    Parameters
    ----------
    version : str, optional
        Which checkpoint version to download. Defaults to ``None``,
        which resolves to the default version
        (``_DEFAULT_MODEL_VERSION`` in this module).

    Returns
    -------
    pathlib.Path
        Local path to the downloaded checkpoint.
    """
    from deepcell_auth import download_deepcell_types_model

    version = version if version is not None else _DEFAULT_MODEL_VERSION
    if version not in _MODEL_VERSIONS:
        raise ValueError(
            f"Unknown model version {version!r}. "
            f"Known versions: {sorted(_MODEL_VERSIONS)}."
        )
    return download_deepcell_types_model(version)


def list_model_versions():
    """Return the available pre-trained model versions, default first.

    Returns
    -------
    list of str
        Version identifiers accepted by :func:`download_model`. The first
        element is always the default version that ``download_model()``
        resolves to with no argument.
    """
    others = sorted(
        (v for v in _MODEL_VERSIONS if v != _DEFAULT_MODEL_VERSION), reverse=True
    )
    return [_DEFAULT_MODEL_VERSION, *others]


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
        Local paths to every checkpoint file for this baseline, sorted by
        filename. Note the asymmetry with :func:`download_model`, which
        returns a single ``Path``: baselines return a *list* because some
        ship companion files. Call :func:`list_baseline_names` for the
        accepted identifiers.
    """
    from deepcell_auth import download_deepcell_types_baseline

    from ._archive import extract_archive

    if name == "nimbus":
        raise ValueError(
            "The Nimbus baseline is inference-only and its pretrained weights "
            "are distributed upstream, not re-hosted by this project. Install "
            "the official library (`pip install nimbus-inference==0.0.5` on "
            "Python 3.11), which downloads the weights automatically; see "
            f"{_NIMBUS_UPSTREAM_URL}."
        )
    if name not in _BASELINE_NAMES:
        raise ValueError(
            f"Unknown baseline {name!r}. Known baselines: {sorted(_BASELINE_NAMES)}."
        )

    (archive,) = download_deepcell_types_baseline(name)
    bundle_dir = archive.parent / archive.name.removesuffix(".tar.gz")
    # Mirrors cellSAM: the bundle is unpacked once, and its presence is what
    # skips re-extraction on later calls. ``fetch_data`` still re-checks the
    # archive's pinned hash on every call, so a corrupt *download* is caught;
    # a hand-edited extraction directory is not.
    if not bundle_dir.is_dir():
        extract_archive(archive, archive.parent)
    baseline_dir = bundle_dir / name
    contents = (
        sorted(p for p in baseline_dir.iterdir() if p.is_file())
        if baseline_dir.is_dir()
        else []
    )
    if not contents:
        raise ValueError(
            f"Baseline bundle {archive.name} did not unpack to a directory of "
            f"checkpoint files at {baseline_dir}. Delete {bundle_dir} and retry."
        )
    return contents


def list_baseline_names():
    """Return the available baseline identifiers, sorted.

    Returns
    -------
    list of str
        Names accepted by :func:`download_baseline_checkpoint` (the
        ``list``-returning counterpart to :func:`list_model_versions`).
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
    from deepcell_auth import download_deepcell_types_data

    from ._archive import extract_archive

    zip_path = download_deepcell_types_data()
    if extract:
        return extract_archive(zip_path)
    return zip_path
