"""Three-part equivalence proof for the relocated maps/cellsighter baselines.

model.py moved byte-identical (sha256). run.py/__init__.py changed ONLY by the
relocation import rewrite `from {pkg}.model import` -> `from .model import`; this
test inverts that single rewrite and asserts the result is byte-identical to the
recorded upstream original, proving no logic changed.
"""

import hashlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "deepcell_types" / "baselines"

MODEL_ORIG_SHA = {
    "maps": "29202958b4326a542732663eb92541681d1d3a10ebc0767bad547416249edc00",
    "cellsighter": "fccb04d5d1eb87159d6afcac473b5b872d5c5aafa54a8c56a65457adbeb2f7f2",
}
# Re-pinned after removing the locally-added ``--min_channels`` CLI option
# (an unused channel-count filter that caused unfair baseline comparisons via
# mismatched defaults).
# Re-pinned again for the public release: two non-logic strings were scrubbed
# in both files — the wandb project default (``deepcelltypes-temp-train`` ->
# ``os.environ.get("WANDB_PROJECT", "deepcell-types")``) and the lab-internal
# ``DATA_DIR`` fallback (``/data2`` -> ``""``). No model, data, or evaluation
# logic changed. The import rewrite remains the only structural delta vs. these.
RUN_ORIG_SHA = {
    "maps": "dfff207acaa086c2e77eb888114860a239497995603cfb662d0414683b172fca",
    "cellsighter": "9903cc3e6861140e65d512da7cf2e0cdbd1b680741e25a444c206437b7b170bf",
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


@pytest.mark.parametrize("pkg", PKGS)
def test_run_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "run.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == RUN_ORIG_SHA[pkg], (
        f"{pkg}/run.py differs from upstream beyond the import rewrite"
    )


@pytest.mark.parametrize("pkg", PKGS)
def test_init_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "__init__.py").read_text(encoding="utf-8")
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode("utf-8")) == INIT_ORIG_SHA[pkg]
