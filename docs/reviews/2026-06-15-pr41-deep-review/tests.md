# Test Quality Audit — deepcell-types v0.1.0 (PR #41)

(3 blockers, 4 highs, 4 mediums, 2 lows)

## BLOCKER 1: No test that abstention is NOT applied to baselines (the paper-fairness contract)
**Location:** `abstention.py:86-88` (untested contract); `tests/test_ct_abstention_cli.py`
The headline lead depends on abstention being DCT-only. No test enforces it.
**Recommendation:** Add a `source` guard to `apply_abstention` and test it raises on non-DCT; or grep-anchor the ownership contract in the eval scripts.

## BLOCKER 2: Checkpoint round-trip test uses a `_TinyNet`, not `CellTypeAnnotator` — ct2idx ordering / compat_marker0_zero / n_heads untested
**Location:** `tests/test_checkpoint_roundtrip.py:36-105`; `predict.py:196-226`
The ordering guard and the un-shape-recoverable fields have no round-trip test.
**Recommendation:** Save a real checkpoint with permuted `ct2idx` (same count) → assert ValueError; round-trip `compat_marker0_zero=False`.

## BLOCKER 3: No test that class weights are computed from train indices only
**Location:** `dataset.py:142-147`, `train.py:62-69`
The full-archive `ct_counts` leak (experimental-design BLOCKER 1) has no guarding test.
**Recommendation:** Mock a split where a class is val-only; assert it gets no non-default weight.

## HIGH 4: `test_samplers.py` doesn't assert the actual sampling distribution (weight tests live in test_v2.py)
**Recommendation:** Assert a 10:1 weight ratio produces ~proportional draws.

## HIGH 5: `test_train_loop_smoke.py` never asserts weights changed or loss dropped (uses `_TinyNet`)
**Recommendation:** Capture pre/post param values; assert at least one changed.

## HIGH 6: Baseline equivalence tests pin source-file SHA, not numeric output — can't catch a behavior change without a byte change
**Location:** `tests/baselines/test_maps_cellsighter_equivalence.py:46-64`
**Recommendation:** Add a fixed-seed synthetic-input forward pass with pinned expected logits (atol=1e-4).

## HIGH 7: ct2idx ordering guard skipped silently for pre-self-describing checkpoints (`ckpt_ct2idx is None`)
**Location:** `predict.py:200-206` — add a warning when absent; test the legacy path.

## MEDIUM 8: `PredictionResult` probability column ordering not asserted against ct2idx (`test_canonical_inference.py:271-298`).
## MEDIUM 9: Mock checkpoints omit `ct2idx`/`config` so all inference tests exercise only the legacy count-only path (`test_canonical_inference.py:221-234`).
## MEDIUM 10: `_max_softmax` CSV column contract untested (`test_predlogger_dataframe.py`).
## MEDIUM 11: Sole-source-class forcing path untested (`test_stratified_splits.py`).

## LOW 12: `test_zero_channel_masking.py` tests a copy of the masking logic, not the production function (anchor only).
## LOW 13: No test for `predict(device_num=...)` deprecated alias.

## Strengths
Sampler math (`TestComputeSampleWeightsCorrectness`), abstention units (IQR formula, k=0, double-apply guard, predict wiring), `test_inference_deps.py` subprocess dep-isolation probe, checkpoint count-mismatch guards, preprocessing contracts, archive-contract validator, PredLogger string-name + CSV schema.
