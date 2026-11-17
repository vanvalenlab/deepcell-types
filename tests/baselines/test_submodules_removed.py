"""Asserts the xgboost/nimbus submodules are folded in and packaging is updated."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_submodule_dirs_gone():
    assert not (ROOT / "baselines" / "xgboost").exists()
    assert not (ROOT / "baselines" / "nimbus").exists()


def test_gitmodules_has_no_xgboost_or_nimbus():
    gm = (ROOT / ".gitmodules").read_text() if (ROOT / ".gitmodules").exists() else ""
    assert "baselines/xgboost" not in gm
    assert "baselines/nimbus" not in gm
    # round 2 folded in maps + cellsighter too; no baseline submodules remain.
    assert "baselines/" not in gm


def test_notice_file_present():
    assert (ROOT / "deepcell_types" / "baselines" / "NOTICE").exists()


def test_pyproject_has_per_method_extras():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "baseline-xgboost" in extras
    assert "baseline-nimbus" in extras
    joined = " ".join(extras["baseline-xgboost"])
    assert "xgboost" in joined and "optuna" in joined
    joined_n = " ".join(extras["baseline-nimbus"])
    assert "nimbus-inference" in joined_n.lower()
