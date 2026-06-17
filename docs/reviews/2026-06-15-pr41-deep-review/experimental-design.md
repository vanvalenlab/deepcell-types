# Experimental Design & Comparison Fairness Audit — deepcell-types v0.1.0 (PR #41)

(3 blockers, 1 high, 2 mediums, 1 low)

## BLOCKER 1: `ct_counts` covers the entire archive, not train-only — class-weight leakage into FocalLoss
**Location:** `deepcell_types/training/dataset.py:143-147`, `scripts/train.py:63-83`, `scripts/train.py:450-455`
`FullImageDataset.ct_counts` is built over every cell (`for idx in self.indices`); `compute_class_weights` reads `dataset.ct_counts` directly. The `Subset` train wrapper narrows iteration but not `ct_counts`, so FocalLoss alpha weights are derived from class frequencies that include val+test. Rare classes concentrated in val/test get over-weighted in the training objective — label-frequency leakage from eval into training.
**Recommendation:** Compute counts over `train_indices` only and pass an explicit counts dict to `compute_class_weights`.

## BLOCKER 2: Default config double-weights rare classes — FocalLoss class weights + WeightedRandomSampler both active
**Location:** `scripts/train.py:290-293`, `:408`, `:615-617`; `deepcell_types/training/dataloader.py:170-182`
`use_weighted_sampler=True` hardcoded; `--no_class_weights` defaults False, so `focal_alpha = class_weights`. Both paths active by default. Sampler floors counts at 1000; FocalLoss `sqrt(total/count)` has NO floor. Combined inflation ≈ `total/316` instead of intended `sqrt(total/N)`. Baselines (CellSighter plain CE, XGBoost none) don't double-weight → asymmetric comparison. The code's own warning (metrics.py) documents this hazard.
**Recommendation:** Use `--no_class_weights` when sampler on, OR floor `compute_class_weights` at 1000, OR drop the sampler (maintainer's plan resolves this).

## BLOCKER 3: Abstention applied only to "ours" — headline macro-F1 comparison is structurally asymmetric
**Location:** `deepcell_types/abstention.py:86-91`, `scripts/predict.py:446-511`, baseline `run.py`
`apply_abstention` docstring: *"must only ever be applied to DCT model predictions — never to baseline predictions. Baselines are scored at full coverage."* `k=0.2` default drops the cells "ours" is most confused about from the macro-F1 denominator; baselines scored on all cells. Per project memory: applying abstention fairly to all methods erases the +4.3pp lead (ours 83.33% vs XGBoost 83.76% kept).
**Recommendation:** Score "ours" at full coverage for the headline (abstention reported separately as a deployment capability), OR apply matched-coverage confidence filtering to all baselines.

## HIGH 1: CellSighter selects its checkpoint on the test set — test-set leakage into model selection
**Location:** `deepcell_types/baselines/cellsighter/run.py:319-337`, `:392-438`
CellSighter's `test_loader` (the FOV-split val/test set) drives `best_macro_acc` checkpointing every `val_every_n_epochs`, then the same loader is the final reported eval. "Ours" uses a separate valsubset for early stopping and an untouched test set. XGBoost correctly uses an inner `GroupShuffleSplit`. CellSighter's number is optimistic by an unknown margin.
**Recommendation:** Give CellSighter an inner FOV-grouped val split for checkpoint selection; reserve the test set for the final number.

## MEDIUM 1: Abstention `k=0.2` "chosen to widen macro_F1 separation over the strongest baseline" — operating point tuned against eval signal
**Location:** `deepcell_types/abstention.py:20-27`
Docstring says k=0.2 was chosen to widen the gap. If selected by observing test-set macro-F1, it's test-set tuning. Cannot verify from code whether tuned on valsubset or test.
**Recommendation:** Document which split selected k=0.2; if test, re-tune on valsubset and re-report.

## MEDIUM 2: Sampler removal alters effective class distribution and requires re-validating the headline
**Location:** `dataloader.py:170-182`, baselines (CellSighter no weighted sampler, MAPS uses one, XGBoost none)
Removing "ours" sampler → trains on natural distribution. More symmetric vs CellSighter, less vs MAPS. Likely lowers rare-class macro-F1; headline cannot be extrapolated from the current checkpoint — needs a retrain + re-eval. Net fairness gain: removes the double-weighting interaction (BLOCKER 2).
**Recommendation:** Retrain & re-evaluate after removal; report the no-sampler number explicitly.

## LOW 1: `missing_value` sentinel differs (XGBoost nan, MAPS/CellSighter 0.0) — legitimate but undocumented in the paper
**Location:** `baseline_features.py:31-75`, `xgb/run.py:113-121`, `maps/run.py:283-291`
Each model gets its native missing-value treatment — defensible, but an asymmetry worth a methods footnote.

## Strengths
Split design sound (two-stage stratified, overlap checks at load). Shared metric code (`_conf_mat_summary`) across all methods. XGBoost early stopping uses a proper inner-val split. Abstention correctly excluded from the training-time val loop (early stopping not optimized for abstention-adjusted metric).
