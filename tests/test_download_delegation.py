"""The static version / baseline lists in ``deepcell_types.utils`` mirror
deepcell-auth's bundled asset manifest. These read the packaged YAML via
``load_manifest`` (no network) and fail loudly if deepcell-auth adds or renames
an entry this repo hasn't mirrored -- the failure mode that would otherwise
surface as a ``KeyError`` from inside ``deepcell_auth`` at download time.
"""

import io
import tarfile

import pytest

import deepcell_auth
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
        # Each baseline resolves to exactly one .tar.gz bundle;
        # download_baseline_checkpoint unpacks it and returns the files inside,
        # so a manifest that split a baseline back into loose per-file records
        # would break the unpacking.
        assert len(records) == 1
        (record,) = records
        assert record["asset_key"].startswith("models/")
        assert record["asset_key"].endswith(".tar.gz")
        assert len(record["asset_hash"]) in (32, 64)
        int(record["asset_hash"], 16)


def test_all_baselines_share_one_bundle():
    # All three baselines ship in a single archive, so every name must resolve
    # to the same asset -- otherwise requesting one baseline would unpack a
    # bundle that does not contain the others' subdirectories.
    baselines = load_manifest()["models"]["deepcell-types-baselines"]
    assets = {records[0]["asset_key"] for records in baselines.values()}
    hashes = {records[0]["asset_hash"] for records in baselines.values()}
    assert len(assets) == 1
    assert len(hashes) == 1


@pytest.mark.parametrize("version", _LEGACY_MODEL_VERSIONS)
def test_download_model_rejects_legacy_clip_versions(version):
    # Served for reproducibility with the matching historical commit, but not
    # loadable by this code -- reject before delegating (so: no network).
    with pytest.raises(ValueError, match="Unknown model version"):
        download_model(version=version)


def test_download_baseline_rejects_nimbus():
    with pytest.raises(ValueError, match="distributed upstream"):
        download_baseline_checkpoint("nimbus")


_STEM = "deepcell-types_baselines_2026-06-30"
_TREE = {
    "cellsighter": {"deepcell-types_baseline-cellsighter.pth": b"cs-weights"},
    "maps": {
        "deepcell-types_baseline-maps.pth": b"maps-weights",
        "deepcell-types_baseline-maps_stats.npz": b"maps-stats",
    },
    "xgboost": {
        "deepcell-types_baseline-xgboost.json": b"xgb-booster",
        "deepcell-types_baseline-xgboost.remap.json": b"xgb-remap",
    },
}


def _make_bundle(models_dir, tree, stem=_STEM):
    """Write a ``<stem>.tar.gz`` laid out the way the served bundle is: a single
    top-level ``<stem>/`` directory with one subdirectory per baseline."""
    archive = models_dir / f"{stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for baseline, members in tree.items():
            for filename, payload in members.items():
                info = tarfile.TarInfo(f"{stem}/{baseline}/{filename}")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
    return archive


@pytest.mark.parametrize("baseline", sorted(_TREE))
def test_download_baseline_returns_only_that_baselines_files(
    tmp_path, monkeypatch, baseline
):
    archive = _make_bundle(tmp_path, _TREE)
    monkeypatch.setattr(
        deepcell_auth, "download_deepcell_types_baseline", lambda name: [archive]
    )

    paths = download_baseline_checkpoint(baseline)

    # One shared archive, but a request returns only the requested baseline --
    # and sorting puts weights ahead of the companion file, matching the order
    # the loose per-file assets were declared in before bundling.
    expected = _TREE[baseline]
    assert [p.name for p in paths] == sorted(expected)
    assert {p.name: p.read_bytes() for p in paths} == expected
    assert all(p.parent.name == baseline for p in paths)


def test_download_baseline_reuses_unpacked_bundle(tmp_path, monkeypatch):
    # Extraction is skipped once the bundle directory exists, so a later call
    # succeeds even if the archive itself is gone from the cache. This is also
    # what makes the other two baselines free after the first download.
    archive = _make_bundle(tmp_path, _TREE)
    monkeypatch.setattr(
        deepcell_auth, "download_deepcell_types_baseline", lambda name: [archive]
    )

    first = download_baseline_checkpoint("cellsighter")
    archive.unlink()
    assert download_baseline_checkpoint("cellsighter") == first
    assert [p.name for p in download_baseline_checkpoint("maps")] == sorted(
        _TREE["maps"]
    )


@pytest.mark.parametrize("member_type", [tarfile.DIRTYPE, tarfile.REGTYPE])
def test_download_baseline_rejects_bundle_without_checkpoints(
    tmp_path, monkeypatch, member_type
):
    # A bundle missing the requested baseline's subdirectory, or holding a
    # plain file where that directory belongs, should surface a readable error
    # rather than a NotADirectoryError from the unpack path.
    archive = tmp_path / f"{_STEM}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo(f"{_STEM}/xgboost")
        info.type = member_type
        tf.addfile(info, io.BytesIO(b"") if member_type == tarfile.REGTYPE else None)
    monkeypatch.setattr(
        deepcell_auth, "download_deepcell_types_baseline", lambda name: [archive]
    )

    with pytest.raises(ValueError, match="did not unpack to a directory"):
        download_baseline_checkpoint("xgboost")
