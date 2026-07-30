"""The static version / baseline lists in ``deepcell_types.utils`` mirror
deepcell-auth's bundled asset manifest. These read the packaged YAML via
``load_manifest`` (no network) and fail loudly if deepcell-auth adds or renames
an entry this repo hasn't mirrored -- the failure mode that would otherwise
surface as a ``KeyError`` from inside ``deepcell_auth`` at download time.
"""

import pytest

from deepcell_auth._auth import load_manifest

from deepcell_types.utils import (
    _DEFAULT_MODEL_VERSION,
    _LEGACY_MODEL_VERSIONS,
    _MODEL_VERSIONS,
    download_baseline_checkpoint,
    download_model,
    list_baseline_names,
    list_model_versions,
)


def test_model_versions_match_manifest():
    manifest_versions = set(load_manifest()["models"]["deepcell-types"])
    assert set(_MODEL_VERSIONS) | set(_LEGACY_MODEL_VERSIONS) == manifest_versions


def test_baseline_names_match_manifest():
    assert set(list_baseline_names()) == set(
        load_manifest()["models"]["deepcell-types-baselines"]
    )


def test_default_version_present_in_manifest():
    assert _DEFAULT_MODEL_VERSION in load_manifest()["models"]["deepcell-types"]
    assert list_model_versions()[0] == _DEFAULT_MODEL_VERSION


def test_manifest_records_carry_key_and_hash():
    models = load_manifest()["models"]
    for version in _MODEL_VERSIONS:
        record = models["deepcell-types"][version]
        assert record["asset_key"].startswith("models/")
        assert len(record["asset_hash"]) in (32, 64)
        int(record["asset_hash"], 16)  # valid hex digest
    for records in models["deepcell-types-baselines"].values():
        for record in records:
            assert record["asset_key"].startswith("models/")
            assert len(record["asset_hash"]) in (32, 64)
            int(record["asset_hash"], 16)


@pytest.mark.parametrize("version", _LEGACY_MODEL_VERSIONS)
def test_download_model_rejects_legacy_clip_versions(version):
    # Served for reproducibility with the matching historical commit, but not
    # loadable by this code -- reject before delegating (so: no network).
    with pytest.raises(ValueError, match="Unknown model version"):
        download_model(version=version)


def test_download_baseline_rejects_nimbus():
    with pytest.raises(ValueError, match="distributed upstream"):
        download_baseline_checkpoint("nimbus")
