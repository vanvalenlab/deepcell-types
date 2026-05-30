# Baseline Migration Round 2 (maps + cellsighter) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the `maps` and `cellsighter` baseline submodules into the in-repo `deepcell_types.baselines` package behind the existing `LazyGroup` runner, with per-method extras — preserving computation, proven by a three-part equivalence check (sha256 on `model.py`, a mechanical transform-and-compare on `run.py`/`__init__.py`, and behavioral characterization).

**Architecture:** Unlike round 1 (xgboost/nimbus moved byte-for-byte), `maps`/`cellsighter` import themselves by name (`from maps.model import …`), so those references are rewritten to **relative** imports (`from .model import …`) — the only change. `model.py` files import only `deepcell_types.*`/external and still move byte-identical. Equivalence is proven by: (a) sha256 byte-identity on each `model.py`; (b) a self-checking test that inverts the single import rewrite on `run.py`/`__init__.py` and asserts the result is byte-identical to the recorded upstream original; (c) a hand-derived golden test on `cellsighter`'s pure helper `convert_batch_for_cellsighter` plus fixed-seed forward smoke tests on both models. The old `__main__.py` entrypoints are dropped; the runner replaces them. After this round, **no baselines remain as submodules**.

**Tech Stack:** Python ≥3.11, setuptools, click, pytest, torch, torchvision, numpy/pandas/scikit-learn.

**Worktree:** `/data/xwang3/Projects/dct-baseline-migration` on branch `refactor/fold-in-baselines` (continues after round 1; round-1 HEAD before this plan is `32e6d90`, plus the round-2 spec commit `2453ad3`). **Never** edit via a path under `/data/xwang3/Projects/deepcell-types/` (the main repo). Verify each commit with `git -C /data/xwang3/Projects/dct-baseline-migration status`.

**Source of truth for the move:** the main-repo checkout `/data/xwang3/Projects/deepcell-types/baselines/{maps,cellsighter}/`. Its `maps` HEAD is `85fa3229` (the chosen version — latest main, early-stopping removed, 15 options) and `cellsighter` HEAD is `cebc391`. Use these as the copy source.

**Spec:** `docs/superpowers/specs/2026-05-30-baseline-migration-round2-maps-cellsighter-design.md`.

**Recorded golden sha256 (verified):**
- `maps/model.py` = `29202958b4326a542732663eb92541681d1d3a10ebc0767bad547416249edc00`
- `cellsighter/model.py` = `fccb04d5d1eb87159d6afcac473b5b872d5c5aafa54a8c56a65457adbeb2f7f2`
- upstream `maps/run.py` = `e6810b88a47ad3239ae3670e0dccba964ba96328b4537f98005cd463c85ddf54`
- upstream `cellsighter/run.py` = `e31f7634b0201217e0c9afe8bf42f4f6548937460c5e273e44a2f449a4b89114`
- upstream `maps/__init__.py` = `5a0a765d62d2f11c841da99f34ccd63b226b47285fe85b6a9edbf92636a58f75`
- upstream `cellsighter/__init__.py` = `2ebb0af69494e85871ec5df7f4ced019ec296bc88d8e049200f370cb625d53a0`

---

## File Structure

**Created:**
- `deepcell_types/baselines/maps/__init__.py` — docstring + relative re-export of `MAPSModel`.
- `deepcell_types/baselines/maps/model.py` — **byte-identical** copy of upstream `maps/model.py`.
- `deepcell_types/baselines/maps/run.py` — upstream `maps/run.py` with `from maps.model import …` → `from .model import …`.
- `deepcell_types/baselines/cellsighter/__init__.py` — docstring + relative re-export of `CellSighterModel`.
- `deepcell_types/baselines/cellsighter/model.py` — **byte-identical** copy of upstream `cellsighter/model.py`.
- `deepcell_types/baselines/cellsighter/run.py` — upstream `cellsighter/run.py` with `from cellsighter.model import …` → `from .model import …`.
- `tests/baselines/test_runner_round2.py` — registry + CLI option-snapshot tests for `maps` and `cellsighter`.
- `tests/baselines/test_maps_cellsighter_equivalence.py` — sha256(model.py ×2) + transform-and-compare proof on run.py/__init__.py ×2.
- `tests/baselines/test_cellsighter_convert_batch_characterization.py` — hand-derived golden test for `convert_batch_for_cellsighter`.
- `tests/baselines/test_models_smoke.py` — fixed-seed forward smoke tests for both models.
- `tests/baselines/test_submodules_removed_round2.py` — asserts maps/cellsighter folded in + extras present (round-2 end-state).

**Modified:**
- `deepcell_types/baselines/__init__.py` — add `maps`, `cellsighter` to `REGISTRY`.
- `tests/baselines/conftest.py` — skip-guards for the new files.
- `tests/baselines/test_submodules_removed.py` — update the round-1 `.gitmodules` assertions (maps/cellsighter no longer remain).
- `pyproject.toml` — add `baseline-maps`, `baseline-cellsighter`; recompose `baselines`/`all`; register two new subpackages.
- `deepcell_types/baselines/NOTICE` — append maps + cellsighter attribution.
- `.gitmodules` — remove the `baselines/maps` and `baselines/cellsighter` stanzas (none remain).
- `README.md` — "## Baselines": all four now in-repo; drop the submodule subsection.

**Removed:**
- `baselines/maps/`, `baselines/cellsighter/` (submodules).

---

## Task 0: Environment & clean source

**Files:** none (setup only).

- [ ] **Step 1: Confirm round-1 green starting point + torch/torchvision present**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -3
.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
```
Expected: round-1 suite green (`246 passed, 4 skipped` modulo data/GPU skips); torch + torchvision import. (torchvision was installed during planning; if missing, `.venv/bin/pip install torchvision`.)

- [ ] **Step 2: Refresh clean maps/cellsighter source into the worktree from the main-repo checkout**

The worktree's `baselines/{maps,cellsighter}` dirs may be in a messy state (broken submodule `.git`). Overwrite their source files from the authoritative main-repo checkout (maps @ `85fa3229`, cellsighter @ `cebc391`):
```bash
cd /data/xwang3/Projects/dct-baseline-migration
SRC=/data/xwang3/Projects/deepcell-types/baselines
cp "$SRC/maps/maps/model.py"            baselines/maps/maps/model.py
cp "$SRC/maps/maps/run.py"              baselines/maps/maps/run.py
cp "$SRC/maps/maps/__init__.py"         baselines/maps/maps/__init__.py
cp "$SRC/cellsighter/cellsighter/model.py"    baselines/cellsighter/cellsighter/model.py
cp "$SRC/cellsighter/cellsighter/run.py"      baselines/cellsighter/cellsighter/run.py
cp "$SRC/cellsighter/cellsighter/__init__.py" baselines/cellsighter/cellsighter/__init__.py
ls baselines/maps/maps/{model,run,__init__}.py baselines/cellsighter/cellsighter/{model,run,__init__}.py
```
Expected: all six paths exist. (These working-tree files are inside the submodule dirs and are not tracked by the parent repo, so they won't appear in `git status` of the parent — they're just the copy source.)

- [ ] **Step 3: Verify the source matches the recorded golden sha256**

```bash
sha256sum baselines/maps/maps/model.py baselines/cellsighter/cellsighter/model.py \
          baselines/maps/maps/run.py baselines/cellsighter/cellsighter/run.py \
          baselines/maps/maps/__init__.py baselines/cellsighter/cellsighter/__init__.py
```
Expected (must match the plan header's golden list exactly):
```
29202958...  maps/model.py
fccb04d5...  cellsighter/model.py
e6810b88...  maps/run.py
e31f7634...  cellsighter/run.py
5a0a765d...  maps/__init__.py
2ebb0af6...  cellsighter/__init__.py
```
If any hash differs, STOP — the source SHA is wrong (e.g. maps not at `85fa3229`). Do not proceed. (No commit in Task 0.)

---

## Task 1: Fold maps + wire runner

**Files:**
- Create: `deepcell_types/baselines/maps/{__init__.py,model.py,run.py}`, `tests/baselines/test_runner_round2.py`, `tests/baselines/test_maps_cellsighter_equivalence.py`
- Modify: `deepcell_types/baselines/__init__.py`, `tests/baselines/conftest.py`

- [ ] **Step 1: Write the failing runner test (maps portion)**

Create `tests/baselines/test_runner_round2.py`:
```python
"""Registry + CLI option-snapshot tests for the round-2 baselines (maps, cellsighter).

Frozen option snapshots are verbatim from the original per-baseline click commands
(maps @ 85fa3229; cellsighter @ cebc391). cellsighter's command imports torchvision
(via cellsighter.model), so its tests importorskip it.
"""
import click
import pytest

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


MAPS_OPTS = {
    "model_name", "device_num", "enable_wandb", "zarr_dir", "skip_datasets",
    "keep_datasets", "split_file", "features_cache", "min_channels", "batch_size",
    "dropout", "hidden_dim", "learning_rate", "max_epochs", "seed",
}
CELLSIGHTER_OPTS = {
    "model_name", "device_num", "enable_wandb", "zarr_dir", "skip_datasets",
    "keep_datasets", "split_file", "split_mode", "test_split_file", "min_channels",
    "batch_size", "epochs", "learning_rate", "model_size", "no_amp", "no_compile",
    "pretrained", "val_every_n_epochs",
}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def test_registry_has_maps():
    assert REGISTRY["maps"] == "deepcell_types.baselines.maps.run:main"


def test_maps_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "maps")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == MAPS_OPTS


def test_registry_has_cellsighter():
    assert REGISTRY["cellsighter"] == "deepcell_types.baselines.cellsighter.run:main"


def test_cellsighter_subcommand_options_frozen():
    pytest.importorskip("torchvision")
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "cellsighter")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == CELLSIGHTER_OPTS
```

Add to `tests/baselines/conftest.py` (after the existing guards) a guard for this file (maps/cellsighter `run.py` import `deepcell_types.training`, which needs the train stack; `pandas` proxies for it as elsewhere in this conftest):
```python
if not _have("pandas"):
    collect_ignore.append("test_runner_round2.py")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/baselines/test_runner_round2.py -q
```
Expected: FAIL — `KeyError: 'maps'` (REGISTRY has no maps entry yet) / the maps subcommand resolves to None.

- [ ] **Step 3: Create the maps package (model byte-identical; run.py + __init__.py relative-import rewrite)**

Use **Bash `cp` + `sed`** (NOT the Edit/Write tools) for `run.py` and `__init__.py` so a formatter hook cannot reflow them and break byte-equivalence-minus-imports:
```bash
cd /data/xwang3/Projects/dct-baseline-migration
mkdir -p deepcell_types/baselines/maps
SRC=/data/xwang3/Projects/deepcell-types/baselines/maps/maps
cp "$SRC/model.py" deepcell_types/baselines/maps/model.py        # byte-identical
cp "$SRC/run.py"   deepcell_types/baselines/maps/run.py
cp "$SRC/__init__.py" deepcell_types/baselines/maps/__init__.py
# Relocation fix: the ONLY change — package-name import -> relative import.
sed -i 's/^from maps\.model import /from .model import /' \
    deepcell_types/baselines/maps/run.py deepcell_types/baselines/maps/__init__.py
```
Verify the rewrite touched exactly one line in each and nothing else changed:
```bash
grep -n "from .model import\|from maps.model import" deepcell_types/baselines/maps/run.py deepcell_types/baselines/maps/__init__.py
diff <(sed 's/^from \.model import /from maps.model import /' deepcell_types/baselines/maps/run.py) "$SRC/run.py" && echo "run.py == upstream modulo import"
diff <(sed 's/^from \.model import /from maps.model import /' deepcell_types/baselines/maps/__init__.py) "$SRC/__init__.py" && echo "__init__.py == upstream modulo import"
sha256sum deepcell_types/baselines/maps/model.py   # must equal 29202958...
```
Expected: each file shows exactly one `from .model import` (no `from maps.model import` remains); both `== upstream modulo import` lines print; model.py hash = `29202958...`.

- [ ] **Step 4: Register maps in the runner**

Edit `deepcell_types/baselines/__init__.py` — add the maps entry to `REGISTRY` so it reads:
```python
REGISTRY = {
    "xgboost": "deepcell_types.baselines.xgb.run:main",
    "xgboost-tune": "deepcell_types.baselines.xgb.tuning:main",
    "nimbus": "deepcell_types.baselines.nimbus.run:main",
    "maps": "deepcell_types.baselines.maps.run:main",
}
```

- [ ] **Step 5: Write the equivalence test (maps portion now; cellsighter added in Task 2)**

Create `tests/baselines/test_maps_cellsighter_equivalence.py`:
```python
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
RUN_ORIG_SHA = {
    "maps": "e6810b88a47ad3239ae3670e0dccba964ba96328b4537f98005cd463c85ddf54",
    "cellsighter": "e31f7634b0201217e0c9afe8bf42f4f6548937460c5e273e44a2f449a4b89114",
}
INIT_ORIG_SHA = {
    "maps": "5a0a765d62d2f11c841da99f34ccd63b226b47285fe85b6a9edbf92636a58f75",
    "cellsighter": "2ebb0af69494e85871ec5df7f4ced019ec296bc88d8e049200f370cb625d53a0",
}

# Packages present at each stage: maps lands in Task 1, cellsighter in Task 2.
PKGS = ["maps"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("pkg", PKGS)
def test_model_py_byte_identical(pkg):
    data = (PKG / pkg / "model.py").read_bytes()
    assert _sha(data) == MODEL_ORIG_SHA[pkg]


@pytest.mark.parametrize("pkg", PKGS)
def test_run_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "run.py").read_text()
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode()) == RUN_ORIG_SHA[pkg], (
        f"{pkg}/run.py differs from upstream beyond the import rewrite"
    )


@pytest.mark.parametrize("pkg", PKGS)
def test_init_py_is_only_import_rewrite(pkg):
    text = (PKG / pkg / "__init__.py").read_text()
    restored = text.replace("from .model import", f"from {pkg}.model import")
    assert _sha(restored.encode()) == INIT_ORIG_SHA[pkg]
```

- [ ] **Step 6: Run maps tests + live CLI to verify green**

```bash
.venv/bin/python -m pytest tests/baselines/test_runner_round2.py::test_registry_has_maps \
    tests/baselines/test_runner_round2.py::test_maps_subcommand_options_frozen \
    tests/baselines/test_maps_cellsighter_equivalence.py -q
.venv/bin/python -m deepcell_types.baselines maps --help
.venv/bin/python -c "from deepcell_types.baselines.maps import MAPSModel; print('re-export OK', MAPSModel.__name__)"
```
Expected: maps tests + the 3 parametrized equivalence tests PASS; `maps --help` lists all 15 options; re-export prints `re-export OK MAPSModel`. (The cellsighter tests in `test_runner_round2.py` will error/skip until Task 2 — run only the maps-named tests as shown.)

- [ ] **Step 7: Commit**

```bash
git add deepcell_types/baselines/maps/ deepcell_types/baselines/__init__.py \
        tests/baselines/conftest.py tests/baselines/test_runner_round2.py \
        tests/baselines/test_maps_cellsighter_equivalence.py
git commit -m "feat(baselines): fold maps into deepcell_types.baselines

Relocate maps @ 85fa3229: model.py byte-identical (sha256); run.py/__init__.py
changed only by the relocation import rewrite (from maps.model -> from .model),
proven by an invert-and-compare test. Registered in the LazyGroup runner.
cellsighter submodule still present; folded in next commit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Do NOT use `git add -A` — the submodule working trees under `baselines/` must NOT be committed.

---

## Task 2: Fold cellsighter + characterization

**Files:**
- Create: `deepcell_types/baselines/cellsighter/{__init__.py,model.py,run.py}`, `tests/baselines/test_cellsighter_convert_batch_characterization.py`, `tests/baselines/test_models_smoke.py`
- Modify: `deepcell_types/baselines/__init__.py` (add cellsighter), `tests/baselines/test_maps_cellsighter_equivalence.py` (add cellsighter to `PKGS`), `tests/baselines/conftest.py`

- [ ] **Step 1: Create the cellsighter package (model byte-identical; run.py + __init__.py relative rewrite)**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
mkdir -p deepcell_types/baselines/cellsighter
SRC=/data/xwang3/Projects/deepcell-types/baselines/cellsighter/cellsighter
cp "$SRC/model.py" deepcell_types/baselines/cellsighter/model.py       # byte-identical
cp "$SRC/run.py"   deepcell_types/baselines/cellsighter/run.py
cp "$SRC/__init__.py" deepcell_types/baselines/cellsighter/__init__.py
sed -i 's/^from cellsighter\.model import /from .model import /' \
    deepcell_types/baselines/cellsighter/run.py deepcell_types/baselines/cellsighter/__init__.py
diff <(sed 's/^from \.model import /from cellsighter.model import /' deepcell_types/baselines/cellsighter/run.py) "$SRC/run.py" && echo "run.py == upstream modulo import"
diff <(sed 's/^from \.model import /from cellsighter.model import /' deepcell_types/baselines/cellsighter/__init__.py) "$SRC/__init__.py" && echo "__init__.py == upstream modulo import"
sha256sum deepcell_types/baselines/cellsighter/model.py   # must equal fccb04d5...
```
Expected: both `== upstream modulo import`; model.py hash = `fccb04d5...`.

- [ ] **Step 2: Register cellsighter + extend the equivalence test**

Edit `deepcell_types/baselines/__init__.py` — `REGISTRY` now:
```python
REGISTRY = {
    "xgboost": "deepcell_types.baselines.xgb.run:main",
    "xgboost-tune": "deepcell_types.baselines.xgb.tuning:main",
    "nimbus": "deepcell_types.baselines.nimbus.run:main",
    "maps": "deepcell_types.baselines.maps.run:main",
    "cellsighter": "deepcell_types.baselines.cellsighter.run:main",
}
```

Edit `tests/baselines/test_maps_cellsighter_equivalence.py` — change the `PKGS` line to include cellsighter:
```python
PKGS = ["maps", "cellsighter"]
```

- [ ] **Step 3: Write the hand-derived characterization test for `convert_batch_for_cellsighter`**

The golden values below are **hand-derived** from the function's confusion-free scatter and **verified against the original** (see Step 4). They pin current behavior, including the known `scatter_` index-0 aliasing quirk (a padded channel whose `ch_idx=-1` is clamped to 0 scatters a 0 into global index 0, after C-order last-write-wins on CPU, clobbering a real marker that maps there). This quirk is part of the vendored baseline and is **not** to be "fixed."

Create `tests/baselines/test_cellsighter_convert_batch_characterization.py`:
```python
"""Hand-derived golden test for cellsighter's pure tensor helper.

convert_batch_for_cellsighter() scatters per-dataset channels to their global
marker positions and appends the cell + neighbor masks. The two synthetic cases
below have hand-derived outputs (cross-checked against the original function).

Layout: B=1, C_max=3, H=W=1, num_markers=4. sample[:, :, 0, 0, 0] = channel values.

Case 1 (no index collision):
  values=[2,3,5], ch_idx=[1,3,-1], mask=[F,F,T] (channel 2 is padding)
  -> masked values [2,3,0]; scatter [2,3,_] to global [1,3,(0 via clamp, but src 0)]
  -> global_patches = [0, 2, 0, 3]; append cell=0.7, neighbor=0.2
  -> [0, 2, 0, 3, 0.7, 0.2]

Case 2 (documents the scatter index-0 clobber quirk; CPU last-write-wins):
  values=[2,3,5], ch_idx=[0,3,-1], mask=[F,F,T]
  -> masked [2,3,0]; clamped idx [0,3,0]; channel0 writes 2 to global0, then the
     padded channel2 writes 0 to global0 LAST -> global0 clobbered to 0
  -> global_patches = [0, 0, 0, 3]; append cell=0.1, neighbor=0.9
  -> [0, 0, 0, 3, 0.1, 0.9]
"""
import dataclasses

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")  # cellsighter.model imports torchvision at module load


def _make_batch(values, ch_idx, mask, cell, neigh, H=1, W=1):
    from deepcell_types.training.utils import BatchData

    C = len(values)
    sample = (
        torch.tensor(values, dtype=torch.float32)
        .reshape(1, C, 1, 1, 1)
        .expand(1, C, 1, H, W)
        .clone()
    )
    spatial = torch.zeros(1, 3, H, W)
    spatial[0, 0] = cell
    spatial[0, 1] = neigh
    kw = {}
    for f in dataclasses.fields(BatchData):
        kw[f.name] = {
            "sample": sample,
            "spatial_context": spatial,
            "ch_idx": torch.tensor([ch_idx], dtype=torch.long),
            "mask": torch.tensor([mask], dtype=torch.bool),
            "marker_positivity_mask": torch.ones(1, C, dtype=torch.bool),
        }.get(f.name, None)
    return BatchData(**kw)


def test_convert_batch_no_collision_hand_derived():
    from deepcell_types.baselines.cellsighter.model import convert_batch_for_cellsighter

    bd = _make_batch([2.0, 3.0, 5.0], [1, 3, -1], [False, False, True], 0.7, 0.2)
    out = convert_batch_for_cellsighter(bd, num_markers=4)
    assert tuple(out.shape) == (1, 6, 1, 1)  # num_markers + 2
    assert out.reshape(-1).tolist() == pytest.approx([0.0, 2.0, 0.0, 3.0, 0.7, 0.2])


def test_convert_batch_index0_clobber_quirk_preserved():
    from deepcell_types.baselines.cellsighter.model import convert_batch_for_cellsighter

    # A real marker at global index 0 is clobbered to 0 by a padded channel
    # (scatter_ duplicate-index last-write-wins on CPU). Pinned, not fixed.
    bd = _make_batch([2.0, 3.0, 5.0], [0, 3, -1], [False, False, True], 0.1, 0.9)
    out = convert_batch_for_cellsighter(bd, num_markers=4)
    assert out.reshape(-1).tolist() == pytest.approx([0.0, 0.0, 0.0, 3.0, 0.1, 0.9])
```

- [ ] **Step 4: Verify the golden values against the ORIGINAL (pre-move) function**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
.venv/bin/python - <<'PY'
import sys, dataclasses, torch
sys.path.insert(0, "/data/xwang3/Projects/deepcell-types/baselines/cellsighter")
from cellsighter.model import convert_batch_for_cellsighter
from deepcell_types.training.utils import BatchData
def mk(values, ch, mask, cell, neigh):
    C=len(values)
    s=torch.tensor(values,dtype=torch.float32).reshape(1,C,1,1,1).clone()
    sp=torch.zeros(1,3,1,1); sp[0,0]=cell; sp[0,1]=neigh
    kw={f.name:{"sample":s,"spatial_context":sp,"ch_idx":torch.tensor([ch]),
        "mask":torch.tensor([mask],dtype=torch.bool),
        "marker_positivity_mask":torch.ones(1,C,dtype=torch.bool)}.get(f.name) for f in dataclasses.fields(BatchData)}
    return BatchData(**kw)
o1=convert_batch_for_cellsighter(mk([2.,3.,5.],[1,3,-1],[False,False,True],0.7,0.2),4).reshape(-1).tolist()
o2=convert_batch_for_cellsighter(mk([2.,3.,5.],[0,3,-1],[False,False,True],0.1,0.9),4).reshape(-1).tolist()
assert o1==[0.,2.,0.,3.,0.7,0.2] and o2==[0.,0.,0.,3.,0.1,0.9], (o1,o2)
print("ORIGINAL matches golden:", o1, o2)
PY
```
Expected: `ORIGINAL matches golden: [0.0, 2.0, 0.0, 3.0, 0.7..., 0.2...] [0.0, 0.0, 0.0, 3.0, 0.1..., 0.9...]`. If this fails, the hand-derived values are wrong — fix the test, do not weaken it.

- [ ] **Step 5: Write the fixed-seed forward smoke tests for both models**

Create `tests/baselines/test_models_smoke.py`:
```python
"""Fixed-seed forward smoke tests for the relocated baseline models.

Pins output shape, finiteness, and seed-determinism (NOT exact numeric values,
which would be brittle across torch/CUDA versions). MAPSModel needs only torch;
CellSighterModel needs torchvision (ResNet backbone)."""
import pytest

torch = pytest.importorskip("torch")


def test_maps_model_forward_shape_and_determinism():
    from deepcell_types.baselines.maps.model import MAPSModel

    torch.manual_seed(0)
    model = MAPSModel(input_dim=10, num_classes=3, hidden_dim=16)
    model.eval()
    x = torch.randn(4, 10)
    with torch.no_grad():
        logits, probs = model(x)
    assert tuple(logits.shape) == (4, 3)
    assert tuple(probs.shape) == (4, 3)
    assert torch.isfinite(logits).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    # determinism: same seed -> same weights -> same output
    torch.manual_seed(0)
    model2 = MAPSModel(input_dim=10, num_classes=3, hidden_dim=16)
    model2.eval()
    with torch.no_grad():
        logits2, _ = model2(x)
    assert torch.allclose(logits, logits2)


def test_cellsighter_model_forward_shape_and_determinism():
    pytest.importorskip("torchvision")
    from deepcell_types.baselines.cellsighter.model import CellSighterModel

    torch.manual_seed(0)
    model = CellSighterModel(
        input_channels=6, num_classes=4, model_size="resnet18", pretrained=False
    )
    model.eval()
    x = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 4)
    assert torch.isfinite(y).all()
    torch.manual_seed(0)
    model2 = CellSighterModel(
        input_channels=6, num_classes=4, model_size="resnet18", pretrained=False
    )
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y, y2)
```

Add to `tests/baselines/conftest.py` a guard for the characterization file (torchvision-gated). `test_models_smoke.py` and `test_maps_cellsighter_equivalence.py` need no guard (the smoke file importorskips torchvision per-test and torch is a base dep; the equivalence file only reads files):
```python
if not _have("torchvision"):
    collect_ignore.append("test_cellsighter_convert_batch_characterization.py")
```

- [ ] **Step 6: Run all baseline tests + live CLI to verify green**

```bash
.venv/bin/python -m pytest tests/baselines/ -q
.venv/bin/python -m deepcell_types.baselines cellsighter --help
.venv/bin/python -m deepcell_types.baselines --help
.venv/bin/python -c "from deepcell_types.baselines.cellsighter import CellSighterModel; print('re-export OK', CellSighterModel.__name__)"
```
Expected: all baseline tests PASS (round-1 tests + round-2 maps/cellsighter snapshots + equivalence (now 6 parametrized) + 2 characterization + 2 smoke); `cellsighter --help` lists all 18 options; group `--help` lists `cellsighter maps nimbus xgboost xgboost-tune`; re-export prints `re-export OK CellSighterModel`.

- [ ] **Step 7: Commit**

```bash
git add deepcell_types/baselines/cellsighter/ deepcell_types/baselines/__init__.py \
        tests/baselines/test_maps_cellsighter_equivalence.py \
        tests/baselines/test_cellsighter_convert_batch_characterization.py \
        tests/baselines/test_models_smoke.py tests/baselines/conftest.py
git commit -m "feat(baselines): fold cellsighter into deepcell_types.baselines

Relocate cellsighter @ cebc391: model.py byte-identical (sha256); run.py/
__init__.py changed only by the relocation import rewrite (invert-and-compare
proven). Adds a hand-derived golden test for convert_batch_for_cellsighter
(pins the known scatter index-0 clobber quirk) + fixed-seed forward smoke tests
for both models. maps/cellsighter submodules removed in the next commit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Remove submodules + per-method extras + packaging

**Files:**
- Modify: `.gitmodules`, `pyproject.toml`, `deepcell_types/baselines/NOTICE`, `tests/baselines/test_submodules_removed.py`
- Create: `tests/baselines/test_submodules_removed_round2.py`
- Remove: `baselines/maps/`, `baselines/cellsighter/`

- [ ] **Step 1: Write the failing round-2 removal/packaging test**

Create `tests/baselines/test_submodules_removed_round2.py`:
```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/baselines/test_submodules_removed_round2.py -q
```
Expected: FAIL — submodule dirs still exist / extras absent.

- [ ] **Step 3: Carry attribution, then remove the submodules**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
# Append maps + cellsighter attribution to the existing combined NOTICE.
{ for n in maps cellsighter; do
    if [ -f "baselines/$n/NOTICE" ]; then echo "## $n"; cat "baselines/$n/NOTICE"; echo; fi
  done; } >> deepcell_types/baselines/NOTICE

# De-integrate the two submodules.
git rm baselines/maps baselines/cellsighter
rm -rf .git/modules/baselines/maps .git/modules/baselines/cellsighter
```
Then `cat .gitmodules`: it should now contain **no** stanzas (all four baselines folded in). If `git rm` left it empty, that's fine; if any baseline stanza remains, hand-edit so none remain (an empty `.gitmodules` may also be deleted — either is acceptable, the test allows both). Note: the `baselines/maps` NOTICE source must still exist at this point (the `git rm` removes the working tree, so do the NOTICE append BEFORE `git rm`, as ordered above).

- [ ] **Step 4: Update `pyproject.toml` extras**

Replace the existing baselines block under `[project.optional-dependencies]`:
```toml
baseline-xgboost = ["deepcell-types[train]", "xgboost", "optuna"]
baseline-nimbus = ["deepcell-types[train]", "nimbus-inference==0.0.5", "opencv-python-headless"]
baselines = ["deepcell-types[baseline-xgboost,baseline-nimbus]"]
all = ["deepcell-types[train,baselines]"]
```
with (add the two new per-method extras and recompose `baselines`):
```toml
baseline-xgboost = ["deepcell-types[train]", "xgboost", "optuna"]
baseline-nimbus = ["deepcell-types[train]", "nimbus-inference==0.0.5", "opencv-python-headless"]
baseline-maps = ["deepcell-types[train]"]
baseline-cellsighter = ["deepcell-types[train]", "torchvision"]
baselines = ["deepcell-types[baseline-xgboost,baseline-nimbus,baseline-maps,baseline-cellsighter]"]
all = ["deepcell-types[train,baselines]"]
```
(Leave the explanatory comment block above these lines intact.)

- [ ] **Step 5: Register the new packages with setuptools**

In `[tool.setuptools] packages`, append the two new subpackages so the list is:
```toml
[tool.setuptools]
packages = [
    'deepcell_types',
    'deepcell_types.training',
    'deepcell_types.utils',
    'deepcell_types.baselines',
    'deepcell_types.baselines.xgb',
    'deepcell_types.baselines.nimbus',
    'deepcell_types.baselines.maps',
    'deepcell_types.baselines.cellsighter'
]
```
(`[tool.setuptools.package-data]` already ships `deepcell_types.baselines` `NOTICE`; no change needed.)

- [ ] **Step 6: Update the round-1 removal test's stale assertion**

`tests/baselines/test_submodules_removed.py::test_gitmodules_has_no_xgboost_or_nimbus` currently asserts `baselines/maps` and `baselines/cellsighter` **remain** in `.gitmodules` — now false. Replace that test body so it no longer requires their presence:
```python
def test_gitmodules_has_no_xgboost_or_nimbus():
    gm = (ROOT / ".gitmodules").read_text() if (ROOT / ".gitmodules").exists() else ""
    assert "baselines/xgboost" not in gm
    assert "baselines/nimbus" not in gm
    # round 2 folded in maps + cellsighter too; no baseline submodules remain.
    assert "baselines/" not in gm
```

- [ ] **Step 7: Reinstall, run packaging tests + full baseline suite**

```bash
.venv/bin/pip install -e ".[baseline-cellsighter]"
.venv/bin/python -m pytest tests/baselines/ -q
.venv/bin/python -c "import deepcell_types.baselines.maps.run, deepcell_types.baselines.cellsighter.run; print('imports OK')"
```
Expected: reinstall succeeds; all baseline tests PASS; `imports OK`.

- [ ] **Step 8: Commit**

```bash
git add .gitmodules pyproject.toml deepcell_types/baselines/NOTICE \
        tests/baselines/test_submodules_removed.py \
        tests/baselines/test_submodules_removed_round2.py
git status --short   # confirm: D baselines/maps, D baselines/cellsighter, M .gitmodules, M pyproject.toml, M NOTICE, M+A tests
git commit -m "refactor(baselines): drop maps/cellsighter submodules; per-method extras

Removes the last two folded-in submodules and their .gitmodules stanzas (none
remain), adds self-contained baseline-maps / baseline-cellsighter extras,
registers the new packages, and carries attribution into the combined NOTICE.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Inspect `git status --short` before committing; if anything unexpected is staged (`.venv`, `__pycache__`), unstage it.

---

## Task 4: Docs, full verification, handoff

**Files:** Modify `README.md`.

- [ ] **Step 1: Update the README Baselines section (all four now in-repo)**

The current "## Baselines" section (after round 1) lists cellsighter+maps as git submodules and xgboost+nimbus as in-repo. Rewrite it so **all four** are in-repo and the "Git submodules" subsection is gone. Requirements:
- A short intro: all paper baselines are folded into `deepcell_types.baselines`, run via `python -m deepcell_types.baselines <name>`.
- Keep each baseline's citation link: cellsighter (`10.1038/s41467-023-40066-7`), maps (`10.1038/s41467-023-44188-w`), nimbus (`10.1038/s41592-025-02683-6`).
- Show install + invocation per baseline:
  - xgboost: `pip install -e ".[baseline-xgboost]"`; `python -m deepcell_types.baselines xgboost ...` and `... xgboost-tune ...`
  - nimbus: `pip install -e ".[baseline-nimbus]"`; `python -m deepcell_types.baselines nimbus ...`; keep the `nimbus-inference==0.0.5` / Python <3.12 note.
  - maps: `pip install -e ".[baseline-maps]"`; `python -m deepcell_types.baselines maps ...`
  - cellsighter: `pip install -e ".[baseline-cellsighter]"`; `python -m deepcell_types.baselines cellsighter ...` (note it pulls `torchvision`).
- Do not invent CLI flags beyond the subcommand names + `...` placeholders.
- Drop the "Each submodule has its own README… `git submodule update --init --recursive`" line (no submodules remain).

Write clean markdown consistent with the surrounding README style.

- [ ] **Step 2: Final equivalence + full-suite verification**

```bash
# Re-prove byte-identity of the two model.py against the recorded originals.
sha256sum deepcell_types/baselines/maps/model.py deepcell_types/baselines/cellsighter/model.py
echo "expect: 29202958... maps/model.py ; fccb04d5... cellsighter/model.py"
# Full test suite.
.venv/bin/python -m pytest tests/ -q
```
Expected: the two hashes match `29202958...` / `fccb04d5...`; full suite green (modulo pre-existing data/GPU skips). If any hash differs or any test fails, STOP and report — do not commit.

- [ ] **Step 3: Commit docs**

```bash
git add README.md
git commit -m "docs(baselines): all four baselines now in-repo under the unified runner

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual follow-up (NOT executed here)**

Record for the user — to be done by them on GitHub after merge:
- Archive `xuefei-wang/deepcelltypes-maps` (folded version `85fa3229`; gitlink had pinned `64de63a`) and `xuefei-wang/deepcelltypes-cellsighter` (`cebc391`).
- Round 2 completes the fold-in: all four baselines (xgboost, nimbus, maps, cellsighter) are now in-repo; no baseline submodules remain.

---

## Self-Review

**Spec coverage:**
- Relative-import relocation (run.py/__init__.py) + byte-identical model.py → Tasks 1–2. ✓
- Three-part equivalence proof (sha256 model.py; invert-and-compare run.py/__init__.py; behavioral characterization) → `test_maps_cellsighter_equivalence.py` (Tasks 1–2), `test_cellsighter_convert_batch_characterization.py` + `test_models_smoke.py` (Task 2). ✓
- Keep `__init__` re-export (relative) → Task 1/2 Step (sed only rewrites the import; docstring + re-export preserved). ✓
- Runner wiring (maps, cellsighter in REGISTRY) → Task 1 Step 4, Task 2 Step 2. ✓
- CLI option snapshots (maps 15, cellsighter 18; subsumes upstream `--test_split_file` test) → `test_runner_round2.py`. ✓
- Per-method extras + packaging + NOTICE → Task 3. ✓
- maps @ `85fa3229` (user decision) → Task 0 golden + header. ✓
- Update round-1 removal test (maps/cellsighter no longer remain) → Task 3 Step 6. ✓
- Docs: all four in-repo, submodule subsection dropped → Task 4. ✓
- Manual archival handoff → Task 4 Step 4. ✓

**Placeholder scan:** No TBD/TODO; every code/test step shows full content; commands show expected output. The two large `run.py` files are intentionally not pasted — they are copied byte-for-byte then a single `sed` rewrites one import, and the invert-and-compare test proves the result equals the recorded upstream sha256.

**Type/name consistency:** `REGISTRY` values `"module:attr"` consistent across `__init__.py`, `LazyGroup.get_command`, and the registry tests; click attr is `main` in both `run.py` (verified). `MAPS_OPTS` (15) / `CELLSIGHTER_OPTS` (18) match the verbatim `@click.option` names (verified at maps `85fa3229`, cellsighter `cebc391`). `convert_batch_for_cellsighter(batch_data, num_markers)` and the model constructors (`MAPSModel(input_dim, num_classes, hidden_dim=…)`, `CellSighterModel(input_channels, num_classes, pretrained=False, model_size=…)`) match the source. Golden tensors in the characterization test verified against the original function.

**Known deviations from round 1 (intentional):** equivalence is three-part (not sha256-only) because relocation requires rewriting intra-package imports; the `convert_batch_for_cellsighter` scatter index-0 quirk is pinned, not fixed; maps is folded at `85fa3229` (current main) rather than the branch's stale gitlink `64de63a`.
