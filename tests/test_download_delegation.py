"""The static version/baseline name lists in ``deepcell_types.utils`` mirror
deepcell-auth's bundled asset manifest. These read the packaged YAML via
``load_manifest`` (no network) and fail loudly if deepcell-auth adds a
version/baseline this repo hasn't mirrored.
"""

from deepcell_auth._auth import load_manifest

from deepcell_types.utils import (
    _DEFAULT_MODEL_VERSION,
    list_baseline_names,
    list_model_versions,
)


def test_model_versions_match_manifest():
    manifest = load_manifest()
    assert set(list_model_versions()) == set(manifest["models"]["deepcell-types"])


def test_baseline_names_match_manifest():
    manifest = load_manifest()
    assert set(list_baseline_names()) == set(
        manifest["models"]["deepcell-types-baselines"]
    )


def test_default_version_present_in_manifest():
    manifest = load_manifest()
    assert _DEFAULT_MODEL_VERSION in manifest["models"]["deepcell-types"]
    assert list_model_versions()[0] == _DEFAULT_MODEL_VERSION
