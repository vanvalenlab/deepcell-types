"""Utilities for model / data access.

The registries below pin checksums of the paper-release checkpoints;
uploads to ``users.deepcell.org`` use the asset paths constructed below
(``models/<filename>``). The hash algorithm is auto-detected from the
digest length (32 hex → md5, 64 hex → sha256), so entries can be migrated
to the stronger sha256 individually; new entries should pin sha256. Some
baselines ship more than one file (e.g. ``maps`` needs its ``_stats.npz``
companion; ``xgboost`` needs its ``.remap.json`` label-remap), so each
baseline entry is a list of ``(filename, hash)`` tuples.
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

_latest = "2026-06-15"

# Main model checkpoints. Values are ``(asset_filename, md5)``.
# Two released residual-MLP checkpoints share the current vocabulary (51 cell
# types, 278 markers) and load via stock ``predict.py`` (the resMLP head is
# auto-detected):
#   - ``2026-06-15`` (default): the from-scratch resMLP (paper Fig-3c headline).
#   - ``2026-06-23``: the pretrain -> finetune (SSL) resMLP arm.
#
# Each md5 tracks the vocab-bundled asset (carries ``ct2idx`` +
# ``canonical_channels`` so ``validate_checkpoint_vocabulary`` can verify
# ordering). The original un-bundled assets predated the guard and raised
# ``ValueError: ... does not bundle a ct2idx`` on every ``predict()`` call;
# they were re-packaged with ``scripts/repackage_release_checkpoint.py``. Each
# bundled asset must be uploaded at its filename below before its md5 is served.
_model_registry = {
    "2026-06-15": (
        "deepcell-types_2026-06-15_resmlp.pt",
        "b819a7e0b177ad5330394eab3c6c7ad8",
    ),
    "2026-06-23": (
        "deepcell-types_2026-06-23_resmlp_ptft.pt",
        "402e94c103c5e489a57433cd107009d3",
    ),
}

# Baseline-model checkpoints. Values are lists of ``(asset_filename, md5)``.
# Single-file baselines have a one-element list; ``maps`` and ``xgboost``
# additionally ship a companion file required at inference. The md5s track the
# 2026-06-30 DCT-sampler retrains -- the release-of-record that produced the
# published baseline prediction CSVs and headline numbers.
#
# Nimbus is intentionally absent: it is inference-only with pretrained weights
# produced and distributed upstream (angelolab/Nimbus-Inference), which this
# project does not redistribute. ``download_baseline_checkpoint("nimbus")``
# raises with a pointer to the official source instead (see below).
_baseline_registry = {
    "cellsighter": [
        (
            "deepcell-types_baseline-cellsighter.pth",
            "c5a105fb044ad82c82817a32aabbae7c",
        ),
    ],
    "maps": [
        (
            "deepcell-types_baseline-maps.pth",
            "1ad5e30ceaccfdb91050b663099258fb",
        ),
        (
            "deepcell-types_baseline-maps_stats.npz",
            "1f462d0c8bf531af73026d415d52728d",
        ),
    ],
    "xgboost": [
        (
            "deepcell-types_baseline-xgboost.json",
            "9e51cf1af8c6c43b00871a821fde0f57",
        ),
        (
            "deepcell-types_baseline-xgboost.remap.json",
            "fba1b0e705e5f7747eb2f9cae30815ba",
        ),
    ],
}

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
        element is always the default (``_latest``) version that
        ``download_model()`` resolves to with no argument.
    """
    others = sorted((v for v in _model_registry if v != _latest), reverse=True)
    return [_latest, *others]


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
        Local paths to every file downloaded for this baseline, in the
        order declared in ``_baseline_registry``. Note the asymmetry with
        :func:`download_model`, which returns a single ``Path``: baselines
        return a *list* because some ship companion files. Call
        :func:`list_baseline_names` for the accepted identifiers.
    """
    from ._auth import fetch_data

    if name == "nimbus":
        raise ValueError(
            "The Nimbus baseline is inference-only and its pretrained weights "
            "are distributed upstream, not re-hosted by this project. Install "
            "the official library (`pip install nimbus-inference==0.0.5` on "
            "Python 3.11), which downloads the weights automatically; see "
            f"{_NIMBUS_UPSTREAM_URL}."
        )
    if name not in _baseline_registry:
        raise ValueError(
            f"Unknown baseline {name!r}. Known baselines: {sorted(_baseline_registry)}."
        )
    return [
        fetch_data(f"models/{filename}", cache_subdir="models", file_hash=md5)
        for filename, md5 in _baseline_registry[name]
    ]


def list_baseline_names():
    """Return the available baseline identifiers, sorted.

    Returns
    -------
    list of str
        Names accepted by :func:`download_baseline_checkpoint` (the
        ``list``-returning counterpart to :func:`list_model_versions`).
    """
    return sorted(_baseline_registry)


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
