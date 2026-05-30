# Baseline migration: fold submodules into `deepcell_types.baselines`

- **Date:** 2026-05-30
- **Branch:** `refactor/fold-in-baselines` (worktree `/data/xwang3/Projects/dct-baseline-migration`)
- **Base:** `refactor/simplify-pr41` @ `caccb4e`
- **Status:** approved design, pending implementation plan

## Problem

The four comparison baselines are each a separate git submodule
(`baselines/{xgboost,nimbus,maps,cellsighter}` → `xuefei-wang/deepcelltypes-*`),
each with its own `pyproject.toml`, venv, `LICENSE`/`NOTICE`, and `python -m <pkg>`
entrypoint. None of them needs to be a submodule:

- `xgboost` and `nimbus` are **thin wrappers** that call a PyPI package
  (`xgboost`, `Nimbus-Inference==0.0.5`) — they vendor zero upstream code, only our glue.
- `maps` and `cellsighter` are **our own reimplementations** (custom MLP / ResNet-50
  variant) — first-party code, not a tracked dependency.

Every submodule already `import`s `deepcell-types`, so they are not independently
useful. The submodule structure costs a stream of `chore(submodule): bump …`
commits, four build files, and four histories, and the per-baseline CLIs drift
in flags and output handling.

## Goal

Fold all baseline code into a single in-repo package `deepcell_types/baselines/`,
behind one unified runner, with per-method optional-dependency extras — **with no
change to the computation** (features, training config, metrics, saved-prediction
schema are identical). Remove the submodules; the four GitHub repos are archived
manually afterward.

## Non-goals (YAGNI)

- No abstract `Baseline` base-class/protocol. The click-group registry is enough;
  an ABC would change call patterns (= behavior risk).
- No golden-master data/GPU runs. Behavior is pinned by characterization unit
  tests on extractable pure logic + a verbatim move (see "TDD strategy").
- No back-compat CLI shims for `python -m xgb` / `python -m nimbus_baseline`.
  The repos are being archived; old entrypoints are dropped.

## Decisions (locked)

| Decision | Choice |
| --- | --- |
| Sequencing | **Vertical slice first**: xgboost + nimbus end-to-end, then a second pass for maps + cellsighter. |
| Behavior guarantee | **Verbatim move + characterization unit tests** (no GPU/real-data run). |
| Upstream repos | **Fold in, then archive** the four GitHub repos (manual GitHub step at the end). |
| Old entrypoints | **Dropped** (no deprecation shims). |
| nimbus package name | **Rename** `nimbus_baseline` → `nimbus`. |

## Target structure (vertical slice)

```
deepcell_types/baselines/
    __init__.py          # REGISTRY = {"xgboost": <cmd>, "nimbus": <cmd>}
    __main__.py          # unified click.Group runner
    xgb/                 # ← baselines/xgboost/xgb/   (keep name "xgb", NOT "xgboost")
        __init__.py
        run.py           # verbatim; only import lines change
        tuning.py        # verbatim; only import lines change
    nimbus/              # ← baselines/nimbus/nimbus_baseline/  (renamed)
        __init__.py
        run.py           # verbatim; only import lines change
```

Round 2 adds `deepcell_types/baselines/maps/` and `…/cellsighter/` with the
identical pattern.

**Naming / shadowing:** the internal package keeps the name `xgb` (its current
name). Because everything is nested under `deepcell_types`, `import xgboost as xgb`
still resolves to the PyPI library (absolute import). A top-level `xgboost/`
directory — the only thing that could shadow the library — is never created.

## Unified runner — reuse the exact click commands

Each baseline already is a `@click.command()` (named `main` in its `run.py`).
The runner is a `click.Group` that attaches those exact command objects:

```python
# deepcell_types/baselines/__main__.py
import click
from deepcell_types.baselines.xgb.run import main as xgb_cmd
from deepcell_types.baselines.nimbus.run import main as nimbus_cmd

@click.group()
def cli():
    """deepcell-types comparison baselines."""

cli.add_command(xgb_cmd, name="xgboost")
cli.add_command(nimbus_cmd, name="nimbus")

if __name__ == "__main__":
    cli()
```

Invocation: `python -m deepcell_types.baselines xgboost --model_name … --zarr_dir …`

Reusing the same command objects preserves every flag and default verbatim — zero
option drift. `__init__.py` exposes a `REGISTRY` dict mapping method name → command
for programmatic discovery.

## Import rewrites (the only change to moved code)

The moved `run.py` / `tuning.py` files are byte-identical **except** import lines.
Known imports to update:

- `xgb/run.py`: imports from `deepcell_types.training.config` and
  `deepcell_types.training.baseline_features` stay valid as-is. Any
  `from xgb.tuning import …` / `import xgb.tuning` becomes a relative
  `from .tuning import …` (absolute top-level `xgb` no longer exists).
- `nimbus/run.py`: imports `nimbus_inference.*` (PyPI, unchanged) and
  `deepcell_types.training.config` (unchanged). The package self-name changes
  from `nimbus_baseline` to `nimbus`; update any `from nimbus_baseline… ` and the
  `__main__.py`.

Round-2 note: `maps/run.py` uses `from maps.model import MAPSModel` and
`cellsighter/run.py` uses `from cellsighter.model import …, convert_batch_for_cellsighter`
— these become relative `from .model import …` after the move.

## Packaging — per-method extras

Refine the existing `baselines` extra in the root `pyproject.toml`:

```toml
[project.optional-dependencies]
baseline-xgboost = ["xgboost", "optuna"]          # optuna for tuning.py sweeps
baseline-nimbus  = ["nimbus-inference==0.0.5", "opencv-python-headless",
                    "scikit-image", "scikit-learn", "scipy", "pandas", "zarr>=3.1.0,<4"]
# round 2: baseline-maps, baseline-cellsighter
baselines        = ["deepcell-types[train,baseline-xgboost,baseline-nimbus]"]
all              = ["deepcell-types[train,baselines]"]
```

Also add `deepcell_types.baselines*` to `[tool.setuptools] packages`.

**Constraint to document (not enforce):** `nimbus-inference==0.0.5` requires
`python <3.12` (current submodule `requires-python = ">=3.11,<3.12"`). pyproject
extras cannot carry a per-extra python bound, so the README notes the nimbus extra
requires Python 3.11. This constraint is exactly why nimbus stays an *extra*, not a
core dependency.

## TDD strategy — "no behavior change" = identical computation

"No behavior change" means features, training config, metrics, and the
saved-prediction schema are bit-for-bit identical. The CLI entry path
*intentionally* changes. Proven in three layers:

1. **Phase 0 — safety net (tests first, green against current in-place code).**
   Characterization unit tests on the deterministic, CPU-only, extractable logic
   each baseline owns, using small synthetic fixtures (no data/GPU):
   - xgboost: FOV-grouped train/val split, NaN missing-value handling, label remap.
   - nimbus: TTA transform (`rot90`/flip and its inversion), per-cell aggregation,
     marker-positivity metric reduction.
   - Plus the existing `tests/test_baseline_feature_splits.py` for the shared layer.
2. **Phase 1 — structure (red → green).** Tests asserting the new API: `deepcell_types.baselines`
   imports; the runner exposes `{xgboost, nimbus}` subcommands; each subcommand's
   option set matches a frozen snapshot of today's options. Red now; the file move +
   runner turns it green. The Phase-0 tests must stay green after the move.
3. **Phase 2 — de-submodule + packaging (red → green).** Tests asserting the
   submodules are gone, the per-method extras resolve, and no stale top-level
   `import xgb` / `import nimbus_baseline` remain.

**Move-equivalence evidence:** moved files are byte-identical except import lines;
`git diff -M` renders them as renames with only import hunks — the diff is the proof
that no logic changed.

## Execution sequence (commits, each independently green)

1. Phase-0 characterization tests (in-place against submodule code), green.
2. Move xgboost → `deepcell_types/baselines/xgb/`; add runner + registry; rewrite imports.
3. Move nimbus → `deepcell_types/baselines/nimbus/`; wire into runner.
4. `git rm` xgboost + nimbus submodules; update `.gitmodules`; refine `pyproject.toml`
   extras + packages; update docs/README.
5. (Round 2, separate plan) maps + cellsighter, same pattern; then remove the last
   two submodules and finalize extras.

Final manual step (listed, not executed): archive the four GitHub repos
(`deepcelltypes-{xgboost,nimbus,maps,cellsighter}`); record their last SHAs —
xgboost `f6fafbf`, nimbus `d3cd960`, maps `64de63a`, cellsighter `cebc391`.

## Fold-in mechanics (submodule → tracked files)

The migration worktree has gitlinks but no submodule contents. To fold a baseline in:

1. Populate working-tree files (copy from the main repo's already-checked-out
   `/data/xwang3/Projects/deepcell-types/baselines/<name>/`, or
   `git submodule update --init baselines/<name>`).
2. `git rm` the gitlink and remove its `.gitmodules` stanza.
3. Move the source files to `deepcell_types/baselines/<name>/`, `git add` as normal
   tracked files (drop `*.egg-info`, per-submodule `pyproject.toml`; keep a `NOTICE`
   attribution for reimplemented published methods).

## Risks / watch-items

- **Editing the wrong tree.** All file ops must use paths under the migration
  worktree (`/data/xwang3/Projects/dct-baseline-migration/…`); verify each write
  with `git -C <worktree> status`.
- **Submodule self-pyproject / egg-info** must not leak into the package.
- **nimbus python<3.12** constraint — documented, not enforced.
- **Baseline test env.** The worktree has no venv yet; Phase-0 verification and the
  unit tests need `deepcell_types` importable. Environment setup is the first step of
  implementation (deferred from worktree setup due to runtime instability during setup).

## Acceptance criteria

- `python -m deepcell_types.baselines xgboost --help` and `… nimbus --help` show the
  same options as the current `python -m xgb` / `python -m nimbus_baseline`.
- All Phase-0 characterization tests pass before and after the move.
- `git diff -M` shows the moved `run.py`/`tuning.py` as renames with import-only hunks.
- `baselines/xgboost` and `baselines/nimbus` submodules and their `.gitmodules`
  stanzas are removed; `pip install -e ".[baseline-xgboost]"` / `".[baseline-nimbus]"`
  resolve.
- Full existing test suite remains green.
