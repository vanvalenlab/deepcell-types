# XGBoost baseline

Gradient-boosted trees (the `xgboost` library) applied to the DeepCell Types
evaluation harness, using per-cell mean marker intensities as features. XGBoost
is a standard ML method, so there is no upstream paper implementation being
adapted — the "deviations" below are simply the choices made to keep the inputs,
split, and scoring identical to the other baselines and to DeepCell Types.

Run with:

```bash
pip install -e ".[baseline-xgboost]"
python -m deepcell_types.baselines xgboost ...        # single fit (default config)
python -m deepcell_types.baselines xgboost-tune ...   # Optuna hyperparameter search
```

## Shared interface (identical to DeepCell Types)

- Reads the same zarr-format Expanded TissueNet archive, the same universal
  marker vocabulary, and the same train/validation/test split JSON
  (`extract_features_from_zarr`, `run.py`).
- Features are per-cell mean intensities aligned to the global marker vocabulary.
- Evaluated with the same hierarchical cell-type accuracy and macro/weighted
  metrics (`compute_baseline_metrics` with `CELL_TYPE_HIERARCHY`).

## Deviations from a vanilla XGBoost fit (recorded for reproducibility)

- **Absent markers are encoded as `NaN`, not `0.0`** (`missing_value=np.nan`,
  `run.py:140`). This routes absent channels through XGBoost's built-in
  `missing=NaN` default-direction logic at every split, so a marker that is
  absent from a panel is not conflated with a real channel whose mean intensity
  happens to be `0.0`.
- **Early stopping on a deterministic, FOV-grouped 10% inner split.** A
  `GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)` grouped by
  training-FOV name carves an inner-validation set out of the *training* data
  (`run.py:176-179`). The frozen test set is never used as the early-stopping
  `eval_set`, so test signal cannot leak into the boosting-round count.
- **Early-stopping rounds** = `max(10, n_estimators // 10)` (`run.py:238`); with
  the default `n_estimators=100` this is 10 rounds.
- **Contiguous label remap.** Labels are remapped to a contiguous `0..K-1`
  index over the classes actually present (`run.py:153-234`); test-only classes
  are appended after the train classes. On the canonical split this means
  train-absent classes (e.g. Erythrocyte) are not modeled, which is why the
  released checkpoint covers 43 of the 51 cell types.

## Default vs released (tuned) configuration

- The default single-fit config is `n_estimators=100`, `max_depth=6`
  (`run.py:50-60`).
- The **released** XGBoost result is the Optuna-tuned model produced by
  `xgboost-tune` (`tuning.py`), which searches `n_estimators` 100–1500,
  `max_depth` 3–12, `learning_rate` 0.005–0.3 (log), `min_child_weight` 1–10,
  and `subsample`/`colsample_bytree` 0.5–1.0.

## Fairness notes vs the neural baselines

- **No `cellSize` feature.** XGBoost uses the per-marker mean intensities only;
  MAPS additionally appends `cellSize` (`maps/run.py`). XGBoost also uses **no**
  class weighting or balanced sampler, whereas MAPS and CellSighter both use
  full-inverse-frequency samplers (CellSighter's default is `--class_balance
  equal`). On rare-class macro
  metrics these make the XGBoost number *conservative* relative to the neural
  baselines, not advantaged.
- **Tuning budget is not matched.** Only XGBoost has an automated hyperparameter
  search (`tuning.py`, ~100 Optuna trials); MAPS and CellSighter run at fixed
  configurations. When comparing the tuned XGBoost number head-to-head with the
  neural baselines, either report the default-config XGBoost (`n_estimators=100`,
  `max_depth=6`) alongside it or note that the neural baselines were not tuned to
  a comparable budget.
