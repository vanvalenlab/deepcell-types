"""Asserts the maps/cellsighter submodules are folded in and packaging is updated."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_maps_cellsighter_dirs_gone():
    assert not (ROOT / "baselines" / "maps").exists()
    assert not (ROOT / "baselines" / "cellsighter").exists()


def test_no_baseline_submodules_remain():
    gm_path = ROOT / ".gitmodules"
    gm = gm_path.read_text() if gm_path.exists() else ""
    assert "baselines/" not in gm  # all four baselines are folded in now


def test_pyproject_has_maps_cellsighter_extras():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "baseline-maps" in extras
    assert "baseline-cellsighter" in extras
    assert "torchvision" in " ".join(extras["baseline-cellsighter"])


def test_notice_has_maps_and_cellsighter():
    notice = (ROOT / "deepcell_types" / "baselines" / "NOTICE").read_text()
    assert "maps" in notice.lower()
    assert "cellsighter" in notice.lower()
