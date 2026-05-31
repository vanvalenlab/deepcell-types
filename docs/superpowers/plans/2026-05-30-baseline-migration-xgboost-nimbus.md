# Baseline Migration (xgboost + nimbus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the `xgboost` and `nimbus` baseline submodules into an in-repo `deepcell_types.baselines` package behind one lazy click-group runner, with per-method extras — with provably identical computation.

**Architecture:** The three source files (`xgb/run.py`, `xgb/tuning.py`, `nimbus_baseline/run.py`) import only `deepcell_types.*` (absolute) and external libraries — never their own package or relative paths. So they relocate **byte-for-byte**; equivalence is proven by `sha256sum` (the parent repo never tracked these files, so they appear as additions, not git renames). A `LazyGroup` runner imports each baseline's module only when its subcommand runs, so optional per-baseline deps stay decoupled. Behavior preservation is pinned by (a) sha256 file-identity, (b) a hand-derived characterization test on nimbus's only importable pure helper, and (c) CLI option-snapshot tests.

**Tech Stack:** Python ≥3.11, setuptools, click, pytest, xgboost, optuna, Nimbus-Inference, numpy/pandas/scikit-learn.

**Worktree:** `/data/xwang3/Projects/dct-baseline-migration` on branch `refactor/fold-in-baselines` (base `refactor/simplify-pr41` @ `caccb4e`). All paths below are relative to this worktree root. **Never** edit via a path under `/data/xwang3/Projects/deepcell-types/` (the main repo) — that lands changes in the wrong tree. Verify each commit with `git -C /data/xwang3/Projects/dct-baseline-migration status`.

**Source of truth for the move:** the main repo has the submodule contents checked out at `/data/xwang3/Projects/deepcell-types/baselines/{xgboost,nimbus}/` (same SHAs the worktree's gitlinks point to: xgboost `f6fafbf`, nimbus `d3cd960`). Use these as the copy source if `git submodule update --init` cannot reach GitHub.

---

## File Structure

**Created:**
- `deepcell_types/baselines/__init__.py` — `REGISTRY` mapping subcommand → `"module:attr"`.
- `deepcell_types/baselines/__main__.py` — `LazyGroup` click runner.
- `deepcell_types/baselines/xgb/__init__.py` — package marker (1-line docstring).
- `deepcell_types/baselines/xgb/run.py` — **byte-identical copy** of `baselines/xgboost/xgb/run.py`.
- `deepcell_types/baselines/xgb/tuning.py` — **byte-identical copy** of `baselines/xgboost/xgb/tuning.py`.
- `deepcell_types/baselines/nimbus/__init__.py` — package marker (1-line docstring).
- `deepcell_types/baselines/nimbus/run.py` — **byte-identical copy** of `baselines/nimbus/nimbus_baseline/run.py`.
- `deepcell_types/baselines/NOTICE` — attribution for the reimplemented/ wrapped methods (carried from the submodules).
- `tests/baselines/__init__.py`
- `tests/baselines/conftest.py` — skip-guard when baseline extras absent.
- `tests/baselines/test_runner.py` — registry + CLI option-snapshot tests.
- `tests/baselines/test_nimbus_metrics_characterization.py` — hand-derived golden test for `compute_marker_positivity_metrics`.
- `tests/baselines/test_submodules_removed.py` — asserts submodules gone + extras present.

**Modified:**
- `pyproject.toml` — per-method extras + `[tool.setuptools]` packages.
- `.gitmodules` — remove `baselines/xgboost` and `baselines/nimbus` stanzas (keep `maps`, `cellsighter`).
- `README.md` (and any baseline docs) — new invocation.

**Removed:**
- `baselines/xgboost/` (submodule), `baselines/nimbus/` (submodule), incl. their `pyproject.toml`, `*.egg-info`, old `__main__.py`.

---

## Task 0: Environment & clean baseline

**Files:** none (setup only).

- [ ] **Step 1: Create a venv in the worktree and confirm Python ≥3.11**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
python3 --version            # must be >=3.11; if not, use an explicit python3.11/3.12
python3 -m venv .venv
.venv/bin/python --version
```
Expected: `Python 3.11.x` or `3.12.x`. (`.venv` is already gitignored at repo root via the standard `.gitignore`; confirm with `git check-ignore .venv` → prints `.venv`.)

Note: the main repo's venv is Python **3.12** (`/data/xwang3/Projects/deepcell-types/.venv` → python3.12). 3.12 is fine for this plan because the tests only install the `baseline-xgboost` extra. If you later want to actually *run* the nimbus baseline (which needs `nimbus-inference==0.0.5`, pinned `<3.12`), create a separate Python 3.11 venv for it.

- [ ] **Step 2: Install the package with the xgboost extra (covers all test imports)**

`baseline-xgboost` pulls `train` (pandas, scikit-learn, click, zarr) + xgboost + optuna. That is enough to import `xgb.run`, `xgb.tuning`, **and** `nimbus.run` (nimbus's `nimbus_inference`/`torch` imports live *inside* `main()`, not at module load). `nimbus-inference` is intentionally **not** installed for tests.

```bash
.venv/bin/pip install -e ".[baseline-xgboost]"
```
Expected: ends with `Successfully installed ... deepcell-types-0.1.0 ...`. (Note: the extra does not exist in `pyproject.toml` yet at base `caccb4e`; if this errors with "does not provide the extra", install `-e ".[train]" xgboost optuna` instead, and revisit after Task 3 adds the extra.)

- [ ] **Step 3: Run the existing test suite to confirm a green starting point**

```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: all collected tests pass or skip (data/GPU-gated tests may skip). If anything *fails* unrelated to our change, STOP and report before proceeding.

- [ ] **Step 4: Obtain submodule source into the worktree (needed to copy files)**

```bash
git submodule update --init baselines/xgboost baselines/nimbus 2>&1 || true
# Fallback if GitHub is unreachable — copy from the main repo checkout:
[ -f baselines/xgboost/xgb/run.py ] || cp -r /data/xwang3/Projects/deepcell-types/baselines/xgboost/. baselines/xgboost/
[ -f baselines/nimbus/nimbus_baseline/run.py ] || cp -r /data/xwang3/Projects/deepcell-types/baselines/nimbus/. baselines/nimbus/
ls baselines/xgboost/xgb/run.py baselines/xgboost/xgb/tuning.py baselines/nimbus/nimbus_baseline/run.py
```
Expected: all three paths listed (exist).

- [ ] **Step 5: Record golden sha256 of the originals (used to prove the move is byte-identical)**

```bash
sha256sum baselines/xgboost/xgb/run.py baselines/xgboost/xgb/tuning.py \
          baselines/nimbus/nimbus_baseline/run.py | tee /tmp/baseline_orig_sha.txt
```
Expected: three hashes printed and saved. (No commit in this task.)

---

## Task 1: Move xgboost + create the lazy runner

**Files:**
- Create: `deepcell_types/baselines/__init__.py`, `deepcell_types/baselines/__main__.py`, `deepcell_types/baselines/xgb/__init__.py`, `deepcell_types/baselines/xgb/run.py`, `deepcell_types/baselines/xgb/tuning.py`
- Create: `tests/baselines/__init__.py`, `tests/baselines/test_runner.py`

- [ ] **Step 1: Write the failing runner test (xgboost portion)**

Create `tests/baselines/__init__.py` (empty).

Create `tests/baselines/conftest.py` so the baseline tests *skip* (not error) in an inference-only checkout that lacks the baseline extras — mirroring the existing `tests/conftest.py` pattern. `test_runner.py`'s option-snapshot tests lazily import `xgboost`; the characterization test imports `pandas`:
```python
"""Skip baseline tests when their optional deps are absent (inference-only env)."""
import importlib


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
    except ImportError:
        return False
    return True


# filenames relative to this conftest's directory
collect_ignore = []
if not _have("xgboost"):
    collect_ignore.append("test_runner.py")
if not _have("pandas"):
    collect_ignore.append("test_nimbus_metrics_characterization.py")
```

Create `tests/baselines/test_runner.py`:

```python
"""Structure + CLI option-snapshot tests for the unified baselines runner.

These pin the public surface (which subcommands exist and their exact options)
without running any baseline. Option sets are frozen snapshots of the original
per-baseline click commands.
"""
import click
import pytest

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


# Frozen option snapshots (verbatim from the original run.py click commands).
XGBOOST_OPTS = {
    "model_name", "enable_wandb", "zarr_dir", "skip_datasets", "keep_datasets",
    "n_estimators", "max_depth", "learning_rate", "split_file", "features_cache",
    "min_channels",
}
NIMBUS_OPTS = {
    "model_name", "device_num", "enable_wandb", "zarr_dir", "skip_datasets",
    "keep_datasets", "checkpoint", "batch_size", "test_time_aug", "threshold",
    "max_fovs",
}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def test_registry_has_xgboost():
    assert "xgboost" in REGISTRY
    assert REGISTRY["xgboost"] == "deepcell_types.baselines.xgb.run:main"


def test_xgboost_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == XGBOOST_OPTS


def test_xgboost_tune_subcommand_exists():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost-tune")
    assert isinstance(cmd, click.Command)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/baselines/test_runner.py -q
```
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'deepcell_types.baselines'`.

- [ ] **Step 3: Create the package, registry, and lazy runner**

`deepcell_types/baselines/__init__.py`:
```python
"""deepcell-types comparison baselines.

Each baseline is a self-contained click command, registered here by name and
imported lazily by the runner (``python -m deepcell_types.baselines``) so that a
baseline's optional dependencies are only required when that subcommand runs.
"""

# subcommand name -> "module_path:click_command_attr"
REGISTRY = {
    "xgboost": "deepcell_types.baselines.xgb.run:main",
    "xgboost-tune": "deepcell_types.baselines.xgb.tuning:main",
}
```

`deepcell_types/baselines/__main__.py`:
```python
"""Unified runner for deepcell-types baselines.

Usage:
    python -m deepcell_types.baselines <baseline> [options]
    python -m deepcell_types.baselines xgboost --split_file ... --zarr_dir ...
    python -m deepcell_types.baselines nimbus --zarr_dir ...
"""
import importlib

import click

from deepcell_types.baselines import REGISTRY


class LazyGroup(click.Group):
    """Click group that imports a baseline's module only when invoked.

    Returning the original click command object preserves every option and
    default verbatim, while deferring the import keeps per-baseline optional
    dependencies (xgboost, nimbus-inference) out of the import path unless that
    subcommand actually runs.
    """

    def list_commands(self, ctx):
        return sorted(REGISTRY)

    def get_command(self, ctx, name):
        target = REGISTRY.get(name)
        if target is None:
            return None
        module_path, attr = target.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, attr)


@click.command(cls=LazyGroup)
def cli():
    """Run deepcell-types comparison baselines."""


if __name__ == "__main__":
    cli()
```

`deepcell_types/baselines/xgb/__init__.py`:
```python
"""XGBoost baseline for cell type classification."""
```

- [ ] **Step 4: Copy the xgboost source byte-for-byte**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
mkdir -p deepcell_types/baselines/xgb
cp baselines/xgboost/xgb/run.py    deepcell_types/baselines/xgb/run.py
cp baselines/xgboost/xgb/tuning.py deepcell_types/baselines/xgb/tuning.py
```

- [ ] **Step 5: Prove the copies are byte-identical to the originals**

```bash
cmp baselines/xgboost/xgb/run.py    deepcell_types/baselines/xgb/run.py    && echo "run.py IDENTICAL"
cmp baselines/xgboost/xgb/tuning.py deepcell_types/baselines/xgb/tuning.py && echo "tuning.py IDENTICAL"
sha256sum deepcell_types/baselines/xgb/run.py deepcell_types/baselines/xgb/tuning.py
```
Expected: both `... IDENTICAL`; the two new hashes equal the corresponding lines in `/tmp/baseline_orig_sha.txt`.

- [ ] **Step 6: Run the test + the live CLI to verify green**

```bash
.venv/bin/python -m pytest tests/baselines/test_runner.py::test_registry_has_xgboost \
    tests/baselines/test_runner.py::test_xgboost_subcommand_options_frozen \
    tests/baselines/test_runner.py::test_xgboost_tune_subcommand_exists -q
.venv/bin/python -m deepcell_types.baselines xgboost --help
.venv/bin/python -m deepcell_types.baselines --help
```
Expected: the 3 tests PASS; `xgboost --help` lists all 11 options (`--model_name … --min_channels`); the group `--help` lists `xgboost` and `xgboost-tune`. (The `test_xgboost_subcommand_options_frozen`/`nimbus` test and `test_registry_has_*nimbus*` are added in Task 2; the nimbus tests in this file will error until then — run only the xgboost-named tests as shown.)

- [ ] **Step 7: Commit**

```bash
git add deepcell_types/baselines/__init__.py deepcell_types/baselines/__main__.py \
        deepcell_types/baselines/xgb/ tests/baselines/__init__.py \
        tests/baselines/conftest.py tests/baselines/test_runner.py
git commit -m "feat(baselines): fold xgboost into deepcell_types.baselines + lazy runner

Byte-identical copy of xgb/{run,tuning}.py (sha256-verified) behind a
LazyGroup click runner. Submodule still present; removed in a later commit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Move nimbus + characterization test + wire runner

**Files:**
- Create: `deepcell_types/baselines/nimbus/__init__.py`, `deepcell_types/baselines/nimbus/run.py`, `tests/baselines/test_nimbus_metrics_characterization.py`
- Modify: `deepcell_types/baselines/__init__.py` (add `nimbus` to `REGISTRY`)

- [ ] **Step 1: Write the hand-derived characterization test for the metric reducer**

This is the one importable pure helper. The inputs below yield an analytically known confusion matrix; the asserted numbers are derived by hand (not copied from the function), so the test independently pins behavior.

Create `tests/baselines/test_nimbus_metrics_characterization.py`:
```python
"""Characterization test for nimbus's pure metric reducer.

compute_marker_positivity_metrics() is the only importable pure helper in the
nimbus baseline (load_fov_data needs zarr; _predict_with_tta is an inner closure
in main()). The synthetic case below has a hand-derived confusion matrix:

  threshold = 0.5
  GT (per cell_type): Tcell -> CD3+ CD20- ; Bcell -> CD3- CD20+
  4 cells:
    cell0 Tcell  CD3=0.9 CD20=0.1   -> CD3:TP  CD20:TN
    cell1 Tcell  CD3=0.4 CD20=0.2   -> CD3:FN  CD20:TN
    cell2 Bcell  CD3=0.1 CD20=0.8   -> CD3:TN  CD20:TP
    cell3 Bcell  CD3=0.2 CD20=0.3   -> CD3:TN  CD20:FN

  Per marker (CD3 and CD20 identical): tp=1 fp=0 fn=1 tn=2
    accuracy=0.75 precision=1.0 recall=0.5 f1=2/3  n_samples=4
  Global pool (8 rows): tp=2 fp=0 fn=2 tn=4
    accuracy=0.75 precision=1.0 recall=0.5 f1=2/3  n_samples=8
"""
import pandas as pd
import pytest

from deepcell_types.baselines.nimbus.run import compute_marker_positivity_metrics


def _make_inputs():
    gt_df = pd.DataFrame(
        {"CD3": {"Tcell": 1, "Bcell": 0}, "CD20": {"Tcell": 0, "Bcell": 1}}
    )  # index = cell types, columns = markers
    ground_truth = {"ds1": gt_df}
    predictions = pd.DataFrame(
        {
            "cell_index": [0, 1, 2, 3],
            "cell_type": ["Tcell", "Tcell", "Bcell", "Bcell"],
            "dataset_name": ["ds1", "ds1", "ds1", "ds1"],
            "CD3": [0.9, 0.4, 0.1, 0.2],
            "CD20": [0.1, 0.2, 0.8, 0.3],
        }
    )
    return predictions, ground_truth


def test_overall_metrics_hand_derived():
    predictions, ground_truth = _make_inputs()
    out = compute_marker_positivity_metrics(predictions, ground_truth, threshold=0.5)
    o = out["overall"]
    assert o["n_samples"] == 8
    assert o["accuracy"] == pytest.approx(0.75)
    assert o["precision"] == pytest.approx(1.0)
    assert o["recall"] == pytest.approx(0.5)
    assert o["f1"] == pytest.approx(2 / 3)
    # mp_* reduction keys present, finite, in range; marker count known.
    for k in ("mp_macro_f1", "mp_micro_f1", "mp_macro_precision",
              "mp_macro_recall", "mp_macro_accuracy"):
        assert 0.0 <= o[k] <= 1.0
    assert o["mp_num_markers"] == 2


def test_per_marker_metrics_hand_derived():
    predictions, ground_truth = _make_inputs()
    out = compute_marker_positivity_metrics(predictions, ground_truth, threshold=0.5)
    for marker in ("CD3", "CD20"):
        m = out["per_marker"][marker]
        assert m["n_samples"] == 4
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Verify the golden values are correct against the ORIGINAL (pre-move) function**

Confirm the test encodes *current* behavior before the module exists at the new path:
```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "baselines/nimbus")
from nimbus_baseline.run import compute_marker_positivity_metrics
import pandas as pd
gt = {"ds1": pd.DataFrame({"CD3":{"Tcell":1,"Bcell":0},"CD20":{"Tcell":0,"Bcell":1}})}
pred = pd.DataFrame({"cell_index":[0,1,2,3],"cell_type":["Tcell","Tcell","Bcell","Bcell"],
                     "dataset_name":["ds1"]*4,"CD3":[0.9,0.4,0.1,0.2],"CD20":[0.1,0.2,0.8,0.3]})
o = compute_marker_positivity_metrics(pred, gt, 0.5)["overall"]
assert o["n_samples"]==8 and abs(o["accuracy"]-0.75)<1e-9 and abs(o["f1"]-2/3)<1e-9 and o["mp_num_markers"]==2, o
print("ORIGINAL matches golden values:", {k:o[k] for k in ("n_samples","accuracy","precision","recall","f1","mp_num_markers")})
PY
```
Expected: `ORIGINAL matches golden values: {...}`. If this assertion fails, the hand-derived values are wrong — fix the test, do **not** weaken it.

- [ ] **Step 3: Run the new test to verify it fails at the new path**

```bash
.venv/bin/python -m pytest tests/baselines/test_nimbus_metrics_characterization.py -q
```
Expected: ERROR — `No module named 'deepcell_types.baselines.nimbus'`.

- [ ] **Step 4: Create the nimbus package, copy source byte-for-byte, register it**

`deepcell_types/baselines/nimbus/__init__.py`:
```python
"""Nimbus baseline for marker positivity prediction."""
```
```bash
mkdir -p deepcell_types/baselines/nimbus
cp baselines/nimbus/nimbus_baseline/run.py deepcell_types/baselines/nimbus/run.py
cmp baselines/nimbus/nimbus_baseline/run.py deepcell_types/baselines/nimbus/run.py && echo "nimbus run.py IDENTICAL"
sha256sum deepcell_types/baselines/nimbus/run.py
```
Expected: `nimbus run.py IDENTICAL`; hash equals the nimbus line in `/tmp/baseline_orig_sha.txt`.

Edit `deepcell_types/baselines/__init__.py` — add the nimbus entry to `REGISTRY`:
```python
REGISTRY = {
    "xgboost": "deepcell_types.baselines.xgb.run:main",
    "xgboost-tune": "deepcell_types.baselines.xgb.tuning:main",
    "nimbus": "deepcell_types.baselines.nimbus.run:main",
}
```

Append the nimbus registry/option tests to `tests/baselines/test_runner.py`:
```python
def test_registry_has_nimbus():
    assert REGISTRY["nimbus"] == "deepcell_types.baselines.nimbus.run:main"


def test_nimbus_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "nimbus")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == NIMBUS_OPTS
```

- [ ] **Step 5: Run all baseline tests + live CLI to verify green**

```bash
.venv/bin/python -m pytest tests/baselines/ -q
.venv/bin/python -m deepcell_types.baselines nimbus --help
```
Expected: all baseline tests PASS; `nimbus --help` lists all 11 options (`--model_name … --max_fovs`).

- [ ] **Step 6: Commit**

```bash
git add deepcell_types/baselines/nimbus/ deepcell_types/baselines/__init__.py \
        tests/baselines/test_runner.py tests/baselines/test_nimbus_metrics_characterization.py
git commit -m "feat(baselines): fold nimbus into deepcell_types.baselines

Byte-identical copy of nimbus run.py (sha256-verified); registered in the
runner. Adds a hand-derived characterization test for the pure marker-
positivity metric reducer. Submodule removed in the next commit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Remove submodules + per-method extras + packaging

**Files:**
- Modify: `.gitmodules`, `pyproject.toml`
- Create: `deepcell_types/baselines/NOTICE`, `tests/baselines/test_submodules_removed.py`
- Remove: `baselines/xgboost/`, `baselines/nimbus/`

- [ ] **Step 1: Write the failing removal/packaging test**

Create `tests/baselines/test_submodules_removed.py`:
```python
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
    # round-2 submodules remain
    assert "baselines/maps" in gm
    assert "baselines/cellsighter" in gm


def test_pyproject_has_per_method_extras():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "baseline-xgboost" in extras
    assert "baseline-nimbus" in extras
    joined = " ".join(extras["baseline-xgboost"])
    assert "xgboost" in joined and "optuna" in joined
    joined_n = " ".join(extras["baseline-nimbus"])
    assert "nimbus-inference" in joined_n.lower()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/baselines/test_submodules_removed.py -q
```
Expected: FAIL — submodule dirs still exist / extras absent.

- [ ] **Step 3: Carry attribution, then remove the submodules**

```bash
cd /data/xwang3/Projects/dct-baseline-migration
# Preserve attribution from the submodules (concatenate their NOTICE files if present).
{ echo "deepcell-types baselines — third-party method attributions"; echo;
  for n in xgboost nimbus; do
    if [ -f "baselines/$n/NOTICE" ]; then echo "## $n"; cat "baselines/$n/NOTICE"; echo; fi
  done; } > deepcell_types/baselines/NOTICE

# De-integrate the two submodules (removes gitlink, working tree, and .gitmodules stanza).
git rm baselines/xgboost baselines/nimbus
rm -rf .git/modules/baselines/xgboost .git/modules/baselines/nimbus
```
Then verify `.gitmodules` still contains the `maps` and `cellsighter` stanzas and **not** xgboost/nimbus; if `git rm` left empty/oversized stanzas, hand-edit `.gitmodules` so only `baselines/maps` and `baselines/cellsighter` remain.

- [ ] **Step 4: Update `pyproject.toml` extras**

Replace the existing `baselines`/`all` lines under `[project.optional-dependencies]` with:
```toml
# Comparison baselines, folded into deepcell_types.baselines. Each extra is
# self-contained (pulls the train stack it needs) so it installs standalone.
# NOTE: baseline-nimbus pins nimbus-inference==0.0.5, which requires Python <3.12.
baseline-xgboost = ["deepcell-types[train]", "xgboost", "optuna"]
baseline-nimbus = ["deepcell-types[train]", "nimbus-inference==0.0.5", "opencv-python-headless"]
baselines = ["deepcell-types[baseline-xgboost,baseline-nimbus]"]
all = ["deepcell-types[train,baselines]"]
```

- [ ] **Step 5: Register the new packages with setuptools**

`[tool.setuptools]` uses an **explicit** packages list (confirmed at base `caccb4e`). Replace:
```toml
[tool.setuptools]
packages = [
    'deepcell_types',
    'deepcell_types.training',
    'deepcell_types.utils'
]
```
with (append the three new subpackages):
```toml
[tool.setuptools]
packages = [
    'deepcell_types',
    'deepcell_types.training',
    'deepcell_types.utils',
    'deepcell_types.baselines',
    'deepcell_types.baselines.xgb',
    'deepcell_types.baselines.nimbus'
]
```
(If a future rebase has switched this to `[tool.setuptools.packages.find]`, no change is needed — the new subpackages auto-discover via their `__init__.py`.)

- [ ] **Step 6: Reinstall, run packaging test + full baseline suite**

```bash
.venv/bin/pip install -e ".[baseline-xgboost]"
.venv/bin/python -m pytest tests/baselines/ -q
.venv/bin/python -c "import deepcell_types.baselines.xgb.run, deepcell_types.baselines.xgb.tuning, deepcell_types.baselines.nimbus.run; print('imports OK')"
```
Expected: reinstall succeeds; all baseline tests PASS; `imports OK`.

- [ ] **Step 7: Commit**

```bash
git add -A
git status --short   # confirm only intended paths: removed baselines/{xgboost,nimbus}, edited .gitmodules/pyproject.toml, new NOTICE + test
git commit -m "refactor(baselines): drop xgboost/nimbus submodules; per-method extras

Removes the two folded-in submodules and their .gitmodules stanzas, adds
self-contained baseline-xgboost / baseline-nimbus extras, registers the new
packages, and carries third-party attribution into deepcell_types/baselines/NOTICE.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Docs, full verification, handoff

**Files:** Modify `README.md` and any baseline-invocation docs.

- [ ] **Step 1: Update invocation docs**

Find and replace old invocations:
```bash
grep -rn -e "python -m xgb" -e "python -m nimbus_baseline" --include="*.md" --include="*.rst" . | grep -v "/.venv/"
```
For each hit, replace with the unified form:
- `python -m xgb ...` → `python -m deepcell_types.baselines xgboost ...`
- `python -m xgb tune ...` → `python -m deepcell_types.baselines xgboost-tune ...`
- `python -m nimbus_baseline ...` → `python -m deepcell_types.baselines nimbus ...`

And document install: `pip install -e ".[baseline-xgboost]"` / `".[baseline-nimbus]"` (note the nimbus → Python 3.11 constraint).

- [ ] **Step 2: Final equivalence + suite verification**

```bash
# Re-prove byte-identity of the moved files against the recorded originals.
sha256sum deepcell_types/baselines/xgb/run.py deepcell_types/baselines/xgb/tuning.py \
          deepcell_types/baselines/nimbus/run.py
cat /tmp/baseline_orig_sha.txt   # compare hashes by eye (basenames differ, hashes must match)
# Full test suite.
.venv/bin/python -m pytest tests/ -q
```
Expected: the three new-path hashes equal the three original hashes; full suite green (modulo pre-existing data/GPU skips from Task 0 Step 3).

- [ ] **Step 3: Commit docs**

```bash
git add -A
git commit -m "docs(baselines): update invocation to unified runner

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual follow-up (NOT executed here)**

Record for the user — to be done by them on GitHub after merge:
- Archive `xuefei-wang/deepcelltypes-xgboost` (last SHA `f6fafbf`) and `xuefei-wang/deepcelltypes-nimbus` (last SHA `d3cd960`).
- Round 2 (separate spec/plan): fold `maps` and `cellsighter` the same way, then remove their submodules and extend the extras.

---

## Self-Review

**Spec coverage:**
- Target structure `deepcell_types/baselines/{xgb,nimbus}` + runner → Tasks 1–2. ✓
- Unified runner reusing exact click commands → `LazyGroup` (Task 1 Step 3), option-snapshot tests. ✓
- Per-method extras + packaging + nimbus Py<3.12 note → Task 3 Steps 4–5. ✓
- TDD: characterization (nimbus metric reducer, hand-derived + verified against original) Task 2; structure/option tests Task 1–2; sha256 byte-identity Tasks 1,2,4 → replaces the spec's `git diff -M` rename proof, which does **not** apply across the submodule boundary (parent never tracked the files). This is a deliberate, stronger correction to the spec. ✓
- Sequencing: xgboost+nimbus only; old entrypoints dropped (no shims); `nimbus_baseline`→`nimbus`; archive repos manual → Tasks 1–4 + Task 4 Step 4. ✓
- maps/cellsighter deferred to round 2 → Task 4 Step 4. ✓

**Placeholder scan:** No TBD/TODO/"add error handling"; every code/test step shows full content; every command shows expected output.

**Type/name consistency:** `REGISTRY` values `"module:attr"` used consistently in `__init__.py`, `LazyGroup.get_command`, and `test_registry_has_*`. Click command attr is `main` in all three source files (verified). Option-name sets `XGBOOST_OPTS`/`NIMBUS_OPTS` match the verbatim `@click.option` names in the source. `compute_marker_positivity_metrics` signature `(predictions, ground_truth, threshold)` matches the source.

**Known deviation from spec (intentional):** equivalence proof is `sha256sum`/`cmp`, not `git diff -M`, because folding across a submodule boundary registers as file additions in the parent repo, not renames.
