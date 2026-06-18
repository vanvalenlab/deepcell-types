# Deep Review: deepcell-types v0.1.0 (PR #41) @ master @ 2026-06-15-2116

**Scope:** the v0.1.0 monorepo-merge diff vs `origin/master` (126 files, +42k/−1.6k). Worktree on `master` = PR head (95a3b3e), so on-disk code is the PR. 10 specialist reviewers; reports in this directory.

**Headline count (deduped): 7 blocker themes, 9 high, ~16 medium, ~8 low.**

## Review team
| Dimension | Findings | Report |
|---|---|---|
| experimental-design | 3B / 1H / 2M / 1L | [experimental-design.md](experimental-design.md) |
| numerical-stability | 1B / 2H / 2M / 2L | [numerical-stability.md](numerical-stability.md) |
| security | 1B / 1H / 2M | [security.md](security.md) |
| performance | 0B / 3H / 4M / 2L | [performance.md](performance.md) |
| tests | 3B / 4H / 4M / 2L | [tests.md](tests.md) |
| docs | 2B / 1H / 2M | [docs.md](docs.md) |
| API | 2B / 3H / 4M / 2L | [api.md](api.md) |
| errors | 2B / 3H / 3M | [errors.md](errors.md) |
| complexity | 1B / 4H / 4M / 1L | [complexity.md](complexity.md) |
| deps | 1B / 2H / 3M | [deps.md](deps.md) |

---

## Blockers

### B-1. Comparison fairness: the headline macro-F1 lead is not apples-to-apples (3 reviewers)
Three independent failure modes, each documented in the code itself, all biasing "ours" up vs the baselines:
- **Class-weight leakage** — `compute_class_weights` reads `FullImageDataset.ct_counts`, computed over the whole archive (train+val+test), not the train split. Eval label frequencies leak into the training objective. *(experimental-design B1, tests B3; dataset.py:143-147, train.py:63-69,450-455)*
- **Double-weighting** — by default the `WeightedRandomSampler` (floored at 1000) AND FocalLoss class weights (no floor) are both active; net rare-class boost ≈ `total/316` vs intended `sqrt(total/N)`. The metrics.py warning describes exactly this; the default does it anyway. Baselines don't double-weight. *(experimental-design B2, numerical BLOCKER, complexity B1; train.py:408,615-617, samplers.py:42-45)*
- **Abstention applied to "ours" only** — `apply_abstention` docstring mandates DCT-only, `k=0.2` "chosen to widen macro_F1 separation over the strongest baseline." Per prior project analysis, scoring all methods at matched coverage erases the +4.3pp lead. *(experimental-design B3+M1, tests B1, api B2; abstention.py:20-31,86-91, scripts/predict.py:446-511)*

> **Directly relevant to this task.** Ripping out the weighted sampler resolves the double-weighting half of B-1. The class-weight leak and the abstention asymmetry are separate decisions that still need to be made and re-validated on the retrained model.

### B-2. `predict()` silently breaks on NumPy ≥ 2.0 (install/release blocker)
`np.ptp` at `preprocessing.py:246` is on the live inference path; removed as a free function in NumPy 2.0; `numpy>=1.24` permits 2.x. The identical call was already fixed at line 140 but this one was missed. A fresh `pip install deepcell-types` today pulls NumPy 2.x → `AttributeError` on the first `predict()`. *(deps HIGH — calibrated up to blocker: it breaks the advertised one-line install.)*

### B-3. Checkpoint deserialization RCE on old torch
`_torch_load_weights` falls back to `weights_only=False` (arbitrary pickle execution) on torch <1.13, with only a silenceable warning. Composes with caller-controlled paths and MD5-only `download_model`. Baseline runners load their own checkpoints with `weights_only=False` outright. *(security BLOCKER+HIGH; predict.py:107-122, baselines/*/run.py)*

### B-4. README contradicts the flagship archive-free-inference feature
README still says you *must* supply a multi-GB TissueNet zarr before `predict()`; the code ships `vocab.json` and runs without it. First thing a new user reads, and it's wrong. *(docs BLOCKER; README.md:5-6,42-46 vs config.py:236-254)*

### B-5. Resume silently loads an incompatible architecture
`--resume_path` config check validates only `resnet_channels`/`d_model`; `n_heads` (not recoverable from tensor shapes) and `n_celltypes` are unchecked → a mismatched-`n_heads` resume runs to completion with the wrong attention config and no error. *(errors B1; train.py:688-698)*

### B-6. Nimbus baseline reports success (exit 0) on fatal failure
"Nimbus not installed" / "no datasets" / "no predictions" all hit a bare `return` in a Click command → exit 0, no metrics written. A comparison suite records "Nimbus ran" when it never produced a number. *(errors B2; baselines/nimbus/run.py:323-368,596-599)*

### B-7. Checkpoint round-trip + fairness contracts are untested
The round-trip test exercises a `_TinyNet`, not `CellTypeAnnotator`: ct2idx ordering, `compat_marker0_zero`, and `n_heads` have no round-trip test; there is no test that abstention stays DCT-only or that class weights are train-only. The result-critical invariants have no regression guard. *(tests B1/B2/B3)*

---

## Highs (by theme)
- **CellSighter test-set leakage into model selection** — its checkpoint is selected on the same held-out set it's finally scored on; "ours" and XGBoost use proper inner-val. Baseline number optimistic. *(experimental-design H1)*
- **CellSighter `scatter_` CUDA bug** — padding writes can clobber the real marker-0 channel (undefined duplicate-index order on CUDA); corrupts CellSighter training. Main model already uses the sink-column fix. *(numerical HIGH)*
- **`predict()` API footguns** — returns `list[str]` or `PredictionResult` depending on a kwarg (no union annotation); abstention ON by default silently relabels to "Unknown" with no warning and no mask in the default return. *(api B1/B2)*
- **Double-normalization footgun** — `preprocess_fov` output fed to `predict()` gets normalized twice silently. *(api H2)*
- **`PredictionResult` not self-describing** — probability columns carry no class names; saving the array loses the mapping. *(api H3)*
- **Nimbus silent error-swallowing** — bare `except Exception` + `print` with no rate guard can drop a large fraction of FOVs and still report a metric. *(errors H2/H3)*
- **Data-loader hot-path waste** — per-cell dict rebuilds, full-generator-per-worker sharding, and redundant full-FOV normalize; ~87% crop compute wasted at workers=8. *(performance H1/H2/H3)*
- **`tissue_idx` dead but load-bearing** — required at dataset init (raises) yet never used in the loss. *(complexity H2)*
- **`main()` god function + fragile `dataset.dataset.dataset` unwrapping** — both will be stressed by removing the sampler. *(complexity H3/H4)*

## Mediums (brief)
Stale `compat_marker0_zero`/clamp comments; pretraining `cell_area` bias; resume legacy-path no validation; `_infer_n_domains`/`_infer_spatial_pool_size` fragility (hardcoded 64, KeyError on pre-DANN); CLI `--zarr_dir` vs API `zarr_path`; `k=0` vs `k=None` semantics; frozen-dataclass mutable arrays; `DATA_DIR` baked into Click defaults; mean-intensity duplicated train/inference; `DataLoaderConfig` missing `fov_grouped_train`; circular-import lazy workaround; kaleido/torchmetrics/torchinfo extra placement; embedding-JSON not in package-data; CHANGELOG missing `preprocess_fov`/`PreprocessedFov`; several untested paths (CSV `_max_softmax`, sole-source-class forcing, `device_num` alias, column ordering).

## Lows
Dead `if mean_intensity is not None`; fp16 softmax for metrics; lambda preprocessor repr; `DataLoaderConfig` persistent_workers default; masking-logic copy in tests; `PreprocessedFov` export placement; freeze string-literal module names; README import path.

---

## Cross-cutting themes (flagged by ≥2 independent reviewers — strongest signal)
1. **Sampler ⊗ class-weight double-counting** (experimental, numerical, complexity) — the single most-corroborated issue; the maintainer's sampler-removal directly addresses it.
2. **Eval signal leaking into training/selection** (class-weight leak, abstention tuning, CellSighter checkpoint-on-test) — a recurring fairness pattern across experimental-design + tests.
3. **Silent-default fallbacks** (compat_marker0_zero, n_heads on resume, weights_only, Nimbus exit-0) — flagged by numerical, errors, security: the codebase prefers a quiet default over a loud failure in several result-critical spots.
4. **Self-describing-checkpoint gaps not yet matched by tests** (errors + tests).

## Disagreements
None substantive. Two reviewers (numerical, complexity) initially flagged a `tissue_idx`/`BatchData` arity blocker and both self-corrected (arity matches: 12 fields) to "dead-but-collected" HIGH — verified correct.

## Strengths
Genuinely strong fundamentals: the main-model `scatter_` sink-column fix with bit-for-bit v0.1.0 compat gating; self-describing checkpoints with vocab-ordering validation; the FOV-grouped sampler + per-worker LRU cache answer to cold-zarr I/O; atomic checkpoint/cache writes with pickle-injection hardening; a subprocess CI test enforcing the inference/train dependency boundary; shared metric code across all methods; bounded (non-eval) preprocessing op library; correct IQR abstention edge-case handling.
