"""Equivalence proof for the relocated maps/cellsighter baselines.

model.py and __init__.py moved byte-identical to upstream (sha256, modulo the
relocation import rewrite `from {pkg}.model import` -> `from .model import`),
proving the model definition and package surface carry no local logic.

run.py is NO LONGER asserted byte-identical to upstream: it intentionally
deviates so the baselines select their best checkpoint on a held-out,
FOV-grouped inner-validation set carved from the training FOVs, rather than on
the set they then report on (selection-on-the-reported-set is leakage). This
mirrors the XGBoost baseline's GroupShuffleSplit early-stopping set. The
``test_run_py_selects_on_inner_val`` behavioral test below pins that deviation;
the byte-equivalence pin for run.py was removed deliberately.
"""

import hashlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "deepcell_types" / "baselines"

MODEL_ORIG_SHA = {
    "maps": "29202958b4326a542732663eb92541681d1d3a10ebc0767bad547416249edc00",
    "cellsighter": "fccb04d5d1eb87159d6afcac473b5b872d5c5aafa54a8c56a65457adbeb2f7f2",
}
INIT_ORIG_SHA = {
    "maps": "5a0a765d62d2f11c841da99f34ccd63b226b47285fe85b6a9edbf92636a58f75",
    "cellsighter": "2ebb0af69494e85871ec5df7f4ced019ec296bc88d8e049200f370cb625d53a0",
}

# Packages present at each stage: maps lands in Task 1, cellsighter in Task 2.
PKGS = ["maps", "cellsighter"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("pkg", PKGS)
def test_model_py_byte_identical(pkg):
    data = (PKG / pkg / "model.py").read_bytes()
    assert _sha(data) == MODEL_ORIG_SHA[pkg]


# Substrings that must appear in each run.py, proving model selection happens on
# a held-out inner-val set rather than on the reported (test) set.
INNER_VAL_MARKERS = {
    "maps": ["GroupShuffleSplit", "X_inner_val_tensor", "inner-val"],
    "cellsighter": ["inner_val_ratio=0.1", "sel_loader", "inner_val_loader"],
}


@pytest.mark.parametrize("pkg", PKGS)
def test_run_py_selects_on_inner_val(pkg):
    """run.py intentionally deviates from upstream: it selects on a held-out,
    FOV-grouped inner-val set, not on the reported test set. Pin that deviation
    behaviorally (the upstream byte-equivalence pin was removed on purpose)."""
    text = (PKG / pkg / "run.py").read_text(encoding="utf-8")
    for marker in INNER_VAL_MARKERS[pkg]:
        assert marker in text, (
            f"{pkg}/run.py is missing the inner-val selection marker {marker!r}; "
            f"checkpoint selection must not run on the reported test set"
        )


@pytest.mark.parametrize("pkg", PKGS)
def test_init_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "__init__.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == INIT_ORIG_SHA[pkg]
