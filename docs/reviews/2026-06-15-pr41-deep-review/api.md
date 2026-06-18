# Public API Audit — deepcell-types v0.1.0 (PR #41)

(2 blockers, 3 highs, 4 mediums, 2 lows)

## BLOCKER 1: `predict()` returns two incompatible types (`list[str]` vs `PredictionResult`)
**Location:** `predict.py:401-409` (no return annotation, line 244)
Default returns a bare list; `return_probabilities=True` returns a dataclass. Static checkers infer a union; downstream `result[0]` breaks. Default drops `abstained`/`cell_indices`/`probabilities`. No prior release to preserve.
**Recommendation:** Always return `PredictionResult`; callers use `.cell_types`.

## BLOCKER 2: Abstention ON by default silently relabels every label to "Unknown" with no warning
**Location:** `predict.py:309-316`, `:392-399`
First-time `predict()` yields "Unknown" the user never requested; with the default bare-list return there's no `abstained` mask to even detect it.
**Recommendation:** Default `ct_abstention_k=None`, or emit a one-time warning naming the relabeled count and how to disable.

## HIGH 1: `device`/`model_name` required-but-defaulted-to-None; `device_num` alias emits no DeprecationWarning
**Location:** `predict.py:244-259, 339-341`.

## HIGH 2: `preprocess_fov` and `make_preprocessor(DEFAULT_CONFIG)` use divergent normalization; passing `preprocess_fov` output to `predict()` double-normalizes silently
**Location:** `preprocessing.py:239-241`, `preprocessing_ops.py:8-10`
**Recommendation:** Warn in both docstrings; `predict()` re-normalizes unless `preprocess=lambda raw,ch: raw`.

## HIGH 3: `PredictionResult.probabilities` column order not self-describing (class names not embedded)
**Location:** `predict.py:29-31, 73-74` — add a `class_names` field.

## MEDIUM 1: CLI `--zarr_dir` vs API `zarr_path` naming inconsistency (`predict.py:255` vs `scripts/predict.py:45`).
## MEDIUM 2: `k=0` vs `k=None` inconsistent disable semantics (`predict.py:312-316` vs `abstention.py:27`).
## MEDIUM 3: `preprocess_fov` channel-namespace expectation undocumented (`preprocessing.py:162-170`).
## MEDIUM 4: `PredictionResult` frozen dataclass holds mutable arrays (in-place mutation silently corrupts).

## LOW 1: `PreprocessedFov` exported top-level but is an archive-ingestion type, not accepted by `predict()`.
## LOW 2: `make_preprocessor` returns an anonymous lambda (no repr/introspection).

## Strengths
Frozen `PredictionResult` with clear `cell_types` vs `cell_types_raw` + `abstained` mask. `_model_path` handles name and path. `FileNotFoundError` names `download_model()`. `weights_only=True` with pinned fallback. `device_num` precedence correct. Case-insensitive channel resolution. Explicit `__all__`.
