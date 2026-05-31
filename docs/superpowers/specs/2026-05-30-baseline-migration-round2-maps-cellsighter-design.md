# Baseline Migration Round 2 (maps + cellsighter) — Design Spec

**Date:** 2026-05-30
**Branch/worktree:** `/data/xwang3/Projects/dct-baseline-migration` on `refactor/fold-in-baselines` (continues after round 1; round-1 HEAD `32e6d90`, PR #13 on the `xuefei` fork).
**Status:** Approved design; implementation plan to follow via `superpowers:writing-plans` → `superpowers:subagent-driven-development`.

## Goal

Fold the last two comparison-baseline submodules — `maps` and `cellsighter` — into the in-repo `deepcell_types.baselines` package behind the existing `LazyGroup` runner, with self-contained per-method extras, in **one plan/PR**. After this round, **no baselines remain as git submodules**; all four (xgboost, nimbus, maps, cellsighter) live in-repo.

## Root-cause context: why this differs from round 1

Round 1's `xgboost`/`nimbus` source files imported **only** `deepcell_types.*` (absolute) + external libraries, so they relocated **byte-for-byte** and equivalence was proven by `sha256sum` alone.

`maps` and `cellsighter` are different: each was a standalone installable package and therefore imports **itself by its distribution name**:

- `maps/run.py:22` and `maps/__init__.py:2`: `from maps.model import MAPSModel`
- `cellsighter/run.py:30`: `from cellsighter.model import CellSighterModel, convert_batch_for_cellsighter`
- `cellsighter/__init__.py:2`: `from cellsighter.model import CellSighterModel`

Those references encode a hidden assumption about *where the package sits in the namespace*. Once the code lives at `deepcell_types.baselines.{maps,cellsighter}`, a reference hardcoded to the bare top-level name is wrong. This is a pure **relocation** problem.

Their `model.py` files, by contrast, import only `deepcell_types.*`/external (`maps/model.py` → torch; `cellsighter/model.py` → torch, torchvision, `deepcell_types.training.utils.BatchData`), so **they still move byte-identical**.

## Decision: relative imports + a three-part equivalence proof

**Import handling (Approach A — relative imports).** Rewrite intra-package references to explicit **relative** imports (`from .model import …`). This is the Python-prescribed form for code inside a package (PEP 328 / PEP 8) and makes the subpackage *relocatable* — it removes the coupling at its source rather than re-hardcoding a new location (which an absolute `from deepcell_types.baselines.maps.model import …` would do). The `__main__.py` entrypoints (top-level + inner) are **dropped**, replaced by the unified runner, exactly as in round 1.

**`__init__.py` re-export is preserved** (rewritten relative), not replaced by a bare docstring, because the re-export (`from deepcell_types.baselines.maps import MAPSModel`) is part of the public surface and is behavior worth keeping.

**Equivalence proof (three parts, strictly stronger than round-1's sha256-only):**

1. **`model.py` (both) → sha256 byte-identical** against the recorded originals. Strongest proof; unchanged from round 1.
2. **`run.py` / `__init__.py` (both) → mechanically-derived transform proof.** A test re-derives the committed file from the upstream original by applying *exactly one* transformation — `from {pkg}.` → `from .` (where `{pkg}` is `maps`/`cellsighter`) — and asserts equality. This proves the diff vs upstream is *only* the import rewrites and **no logic changed**. Self-checking, not eyeballed.
3. **Behavioral characterization** (independent confirmation):
   - **Hand-derived golden test** for `cellsighter`'s pure tensor helper `convert_batch_for_cellsighter(batch_data: BatchData, num_markers) -> Tensor`. Build a tiny synthetic `BatchData` with known values; hand-derive the expected globally-aligned output; **cross-check the golden values against the original (pre-move) function before trusting them** (round-1 nimbus discipline). The test pins **current** behavior, including the `scatter_` index-0 aliasing quirk (invalid `ch_idx=-1` clamped to 0, zeroed via the `valid` mask) — this is vendored baseline code and is **not** to be "fixed."
   - **Fixed-seed forward smoke test** per model: construct small (`MAPSModel(input_dim, num_classes, hidden_dim)`, forward `(B, input_dim)` → `(logits, probs)` both `(B, num_classes)`; `CellSighterModel(input_channels, num_classes, model_size="resnet18", pretrained=False)`, forward `(B, C, 32, 32)` → `(B, num_classes)`). Seed with `torch.manual_seed`; assert output shapes, finiteness, and that two runs with the same seed match. **No exact numeric golden values** (avoids torch/CUDA-version brittleness on a published baseline).

## Files

**Created:**
- `deepcell_types/baselines/maps/__init__.py` — re-export `MAPSModel` via relative import (transformed from upstream).
- `deepcell_types/baselines/maps/model.py` — **byte-identical** copy of `baselines/maps/maps/model.py`.
- `deepcell_types/baselines/maps/run.py` — upstream `maps/run.py` with `from maps.model …` → `from .model …`.
- `deepcell_types/baselines/cellsighter/__init__.py` — re-export `CellSighterModel` via relative import (transformed).
- `deepcell_types/baselines/cellsighter/model.py` — **byte-identical** copy of `baselines/cellsighter/cellsighter/model.py`.
- `deepcell_types/baselines/cellsighter/run.py` — upstream `cellsighter/run.py` with `from cellsighter.model …` → `from .model …`.
- `tests/baselines/test_maps_cellsighter_equivalence.py` — sha256(model.py ×2) + the transform-and-compare proof for run.py/__init__.py ×2.
- `tests/baselines/test_cellsighter_convert_batch_characterization.py` — hand-derived golden test for `convert_batch_for_cellsighter`.
- `tests/baselines/test_models_smoke.py` — fixed-seed forward smoke tests for both models.

**Modified:**
- `deepcell_types/baselines/__init__.py` — add `maps`, `cellsighter` to `REGISTRY`.
- `tests/baselines/test_runner.py` — add `MAPS_OPTS` (15) + `CELLSIGHTER_OPTS` (18) frozen sets and their registry + option-snapshot tests (the cellsighter snapshot subsumes the upstream `--test_split_file` existence test).
- `tests/baselines/conftest.py` — skip-guards: maps tests gated on `torch` (base); cellsighter tests gated on `torchvision`; the convert-batch test on `torch`+`pandas`/train.
- `tests/baselines/test_submodules_removed.py` — **update** the round-1 assertions: `baselines/maps` and `baselines/cellsighter` dirs gone; `.gitmodules` has no baseline submodules (file may be removed entirely once empty); `baseline-maps` and `baseline-cellsighter` extras present.
- `pyproject.toml` — add `baseline-maps = ["deepcell-types[train]"]` and `baseline-cellsighter = ["deepcell-types[train]", "torchvision"]`; recompose `baselines`/`all`; register `deepcell_types.baselines.maps` and `…cellsighter` in `[tool.setuptools] packages`.
- `deepcell_types/baselines/NOTICE` — append the maps and cellsighter attribution blocks (already shipped via package-data).
- `.gitmodules` — remove `baselines/maps` and `baselines/cellsighter` stanzas (leaving none).
- `README.md` — "## Baselines": move maps + cellsighter into the in-repo list; the "git submodules" subsection disappears (none remain); note cellsighter needs `torchvision`.

**Removed:**
- `baselines/maps/` (submodule), `baselines/cellsighter/` (submodule, SHA `cebc391`).

## Post-execution correction (cellsighter drift)

> **Added after execution.** The claim below that "cellsighter has no drift (`cebc391` everywhere)" turned out to be **wrong**. The `deepcelltypes-cellsighter` repo was rebased/force-pushed; `cebc391` no longer exists on it and `main` had advanced to **`f9e336a`** (num_markers 271→269, `--zarr_dir` default fix, restored hierarchy-collapse + F1 eval, post-refactor imports; the `--test_split_file` option was removed → 17 options). cellsighter was **re-folded at `f9e336a`** in commit `0465aa0`. `model.py`/`__init__.py` and `convert_batch_for_cellsighter` were unchanged across `cebc391..f9e336a`, so only `run.py` (golden sha → `915b77d7`) and the option snapshot changed. References to `cebc391` below are historical.

## Source-SHA decision (maps drift)

The branch's `maps` gitlink pins `64de63a`, but the maps repo's `main` has since advanced to **`85fa3229`** (commit "default to 50 epochs, remove early stopping"), which the main-repo checkout has. **Decision (user): fold in `85fa3229`** — the current maps, not the stale gitlink. This intentionally includes the early-stopping removal, so maps's option surface is **15** options (the `--min_epochs`/`--patience` options present at `64de63a` are gone). `maps/model.py` and `maps/__init__.py` are byte-identical across the two SHAs; only `run.py` differs, so this choice only affects the folded `run.py` and the `MAPS_OPTS` snapshot. cellsighter has no drift (`cebc391` everywhere). Task 0 obtains maps source at `85fa3229` and cellsighter at `cebc391` from the main-repo checkout `/data/xwang3/Projects/deepcell-types/baselines/{maps,cellsighter}/`. Golden sha256 recorded at `/tmp/baseline_round2_model_sha.txt` (model.py ×2) and `/tmp/baseline_round2_transform_src_sha.txt` (upstream run.py/__init__.py ×2, transform-proof inputs).

## CLI option snapshots (frozen public surface)

- **maps** (15): `model_name, device_num, enable_wandb, zarr_dir, skip_datasets, keep_datasets, split_file, features_cache, min_channels, batch_size, dropout, hidden_dim, learning_rate, max_epochs, seed`.
- **cellsighter** (18): `model_name, device_num, enable_wandb, zarr_dir, skip_datasets, keep_datasets, split_file, split_mode, test_split_file, min_channels, batch_size, epochs, learning_rate, model_size, no_amp, no_compile, pretrained, val_every_n_epochs`.

## Packaging / dependency facts (verified)

- Base `deepcell-types` deps include `torch`, `tqdm`, `numpy`, `scipy`, `scikit-image`, `numcodecs`.
- `[train]` adds `pandas`, `scikit-learn`, `click`, `wandb`, `zarr`, `torchinfo`, `torchmetrics`, `plotly`, `kaleido`, `tifffile`, `openai`.
- maps needs torch(base)+click(train)+numpy(base)+pandas(train)+tqdm(base)+scikit-learn(train) → fully covered by `deepcell-types[train]`.
- cellsighter additionally needs **torchvision** (not in base/train) → `["deepcell-types[train]", "torchvision"]`.

## Task breakdown (for the plan)

- **Task 0:** Env — obtain clean maps+cellsighter source into the worktree (submodule init or copy from `/data/xwang3/Projects/deepcell-types/baselines/{maps,cellsighter}/`); record golden sha256 of the two `model.py` and of the upstream `run.py`/`__init__.py` (the latter as transform-proof inputs); confirm round-1 green starting point.
- **Task 1:** Fold maps — copy model.py byte-identical; transform run.py/__init__.py; register in REGISTRY; option-snapshot test; transform+sha256 test (maps portion). TDD.
- **Task 2:** Fold cellsighter — same; plus the `convert_batch_for_cellsighter` hand-derived golden test (cross-checked vs original) and the forward smoke tests for both models.
- **Task 3:** Remove both submodules; add `baseline-maps`/`baseline-cellsighter` extras; register packages; carry NOTICE; **update `test_submodules_removed.py`** for the all-folded end-state.
- **Task 4:** Docs (README) + final equivalence/suite verification + handoff (manual GitHub archival of `xuefei-wang/deepcelltypes-{maps,cellsighter}`).

## Known deviations from round 1 (intentional)

- Equivalence proof is **three-part** (sha256 + transform-and-compare + behavioral), not sha256-only, because the relocation requires rewriting intra-package imports in `run.py`/`__init__.py`.
- The round-1 `test_submodules_removed.py` assertion that maps/cellsighter **remain** in `.gitmodules` is now inverted; `.gitmodules` may be deleted entirely once it has no stanzas.
- `convert_batch_for_cellsighter`'s `scatter_` index-0 aliasing is preserved verbatim (vendored code); the characterization test pins it rather than correcting it.
