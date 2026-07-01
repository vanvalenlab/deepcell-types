# Changelog

All notable changes to `deepcell-types` are listed here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

This release ships the training and inference pipelines in a single
repository and switches the inference path to checkpoints that embed the
full canonical metadata. **Breaking changes** are noted below.

### Added
- **`--val_split_file` canonical external-val selection for baselines.**
  MAPS, CellSighter, and XGBoost training/tuning scripts now accept
  `--val_split_file` to select model checkpoints against a fixed, canonical
  validation split (200k-cell cap, seed 42) shared across all baselines and
  the main model, instead of each script carving its own validation subset.
- **Custom preprocessing hook.** `predict(..., preprocess=...)` overrides the
  per-FOV normalization on a single FOV without retraining. Ships a bounded op
  library — `apply_config`, `make_preprocessor`, `DEFAULT_CONFIG` (top-level
  exports) — and the `preproc-adapt` skill (`skills/preproc-adapt/`) for the
  composition-guided adaptation loop. `preprocess=None` is the unchanged
  default; `make_preprocessor(DEFAULT_CONFIG)` reproduces it bit-for-bit.
- **Archive-free inference.** `predict()` no longer requires the (multi-GB)
  TissueNet zarr archive: the marker / cell-type registry now ships as a small
  `vocab.json` snapshot, and `DCTConfig` falls back to it when no archive is
  found. `pip install deepcell-types` + `download_model()` is enough to run
  `predict()`. Pass `zarr_path=` / set `DEEPCELL_TYPES_ZARR_PATH` only if you
  need the archive (e.g. the tissue→cell-type mapping). Verified identical
  predictions with vs. without the archive on the paper checkpoint.
- Training pipeline (`deepcell_types.training`) is now shipped from this
  repository, gated behind the `[train]` install extra:
  ```bash
  pip install "deepcell-types[train]"
  ```
  Includes `TissueNetConfig`, `FullImageDataset`, `FOVGroupedSampler`,
  `FocalLoss`, `HierarchicalLoss`, `MPMetricsTracker`, and the new
  focused submodules `training/{archive,patch,metrics,baseline_features}.py`.
- End-to-end training scripts under `scripts/` (`train.py`,
  `pretrain.py`, `predict.py`).
- **Residual-MLP cell-type head (`ct_head_arch="resmlp"`).** A width-512,
  depth-4 residual-MLP head trained on the frozen backbone via
  `scripts/retrain_head.py` (the two-stage sampler-off recipe). Checkpoints
  record `config["ct_head_arch"]` and `predict()` auto-detects a `resmlp` head
  from the state dict. The legacy 3-layer MLP (`ct_head_arch="mlp"`) remains
  the default, so existing v0.1.0 checkpoints load unchanged — `resmlp`
  checkpoints have a different head shape and are NOT interchangeable with
  `mlp` checkpoints.
- Public top-level exports: `DCTConfig` and `PredictionResult` are now
  importable from `deepcell_types` directly.
- `predict(return_probabilities=True)` returns a `PredictionResult`
  dataclass with the full per-cell softmax matrix and cell indices.
- Four paper comparison baselines vendored under
  `deepcell_types.baselines` (`cellsighter`, `maps`, `nimbus`, `xgb`),
  invoked through the unified runner
  `python -m deepcell_types.baselines <name>`. Each ships a self-contained
  install extra (`baseline-xgboost`, `baseline-nimbus`, `baseline-maps`,
  `baseline-cellsighter`).

### Changed
- **Breaking (baselines default): all baselines unified onto the DCT sampler
  by default.** MAPS, CellSighter, and XGBoost previously defaulted to their
  own native class-balancing scheme; `--class_balance` now defaults to
  `dct`/`sqrt` (sqrt-inverse-frequency + 1000-count floor) across all three,
  matching the main model's sampler, so cross-method comparisons isolate the
  modeling choice rather than the balancing scheme. Each baseline's own
  native sampler remains available via `--class_balance` and is reported
  separately in the paper appendix.
- **Residual-MLP cell-type head is now the DEFAULT** (`ct_head_arch="resmlp"`)
  for fresh training, replacing the legacy 3-layer MLP. `scripts/train.py` gains
  `--ct_head_arch {resmlp,mlp}` (default `resmlp`), records it in the checkpoint
  config, and validates it on `--resume_path`. Inference is unaffected for
  existing v0.1.0 checkpoints: `predict()` auto-detects the head from the
  state_dict, so the released (legacy-MLP) checkpoint still loads unchanged.
- Default training learning rate raised `3e-4 → 1e-3` in `scripts/train.py`,
  matching the canonical backbone recipe.
- **Breaking (training default):** backbone training no longer applies per-class
  `FocalLoss` weights. The `WeightedRandomSampler` is now the sole rare-class
  balancer, so double-weighting is impossible. The `--no_class_weights` flag and
  the `compute_class_weights` helper are removed, and the checkpoint config no
  longer records `no_class_weights`. The focal term (`--focal_gamma`) is
  unchanged. This makes the no-flags default reproduce the released-checkpoint
  recipe (which was trained with `--no_class_weights`); it changes
  `scripts/train.py`'s no-flag default versus earlier 0.1.0 development builds.
  The stage-2 head retrain (`scripts/retrain_head.py`) already used plain
  `CrossEntropyLoss` and is unaffected.
- **Breaking:** the legacy `CellTypeCLIPModel` class is removed.
  Canonical checkpoints now read marker / cell-type metadata from a
  TissueNet zarr v3 archive at inference time; the active model class
  is `CellTypeAnnotator` in `deepcell_types/model.py`.
- **Breaking:** all `predict()` arguments after `mpp` are now keyword-only,
  preventing accidental transposition of the adjacent string arguments
  `model_name` / `device`.
- The inference bright-spot clip percentile (`DCTConfig.PERCENTILE_THRESHOLD`)
  is now `99.9`, matching the recipe the training archive's `preprocessed/raw`
  was built with (was `99.0`, a carryover from the original library packaging).
  This shifts ~5% of predicted labels; on a held-out test-split sample it
  reproduced the canonical predictions slightly better (92.5% vs 91.9% argmax
  agreement).
- `predict(device=...)` is the preferred spelling for the inference device;
  `device_num=...` remains accepted as a deprecated alias.
- `predict()` now raises a clear `FileNotFoundError` (pointing at
  `download_model()`) when the requested checkpoint is absent, instead of a
  bare error from `torch.load`.
- Checkpoints are now self-describing: `scripts/train.py` bundles `ct2idx`,
  `n_heads`, and `compat_marker0_zero`, and inference asserts the archive's
  cell-type / marker ordering matches the checkpoint (a permuted vocabulary
  previously passed the count-only check and silently mislabeled cells).
- Duplicate input channels that resolve to the same canonical marker are now
  de-duplicated (the per-marker scatter is last-write-wins).
- **All-zero channels are now masked at inference**, matching the training
  dataloader. A marker listed in `channel_names` but all-zero on a given FOV was
  previously fed to the model as a present zero token carrying a real marker
  embedding — an input the model was trained never to see; it is now dropped
  (with a warning), aligning the `predict()` channel handling with the
  evaluation path. This can change predictions for FOVs that contain all-zero
  marker channels.
- `predict()` now rejects non-finite (`NaN`/`inf`) `raw` with a clear
  `ValueError` instead of silently labelling every cell as a single class via a
  poisoned softmax.
- Inference now sizes its per-cell tensors to the number of channels actually
  present in the FOV instead of the global `MAX_NUM_CHANNELS`. Padding tokens are
  inert in the model, so predictions are unchanged; the per-channel ResNet and
  the (channel-quadratic) transformer simply skip the wasted work over padding.
- **Breaking:** `predict(num_workers=...)` default is now `0` (was
  `24`). The `IterableDataset` patch generator held the whole FOV in
  memory; 24 workers reliably OOM'd machines with <64 GB RAM.
- Post-hoc cell-type abstention is **opt-in**: `predict(ct_abstention_k=...)`
  defaults to `None`, returning the raw argmax label for every cell. When set to
  a float, cells whose top-class probability falls below an IQR fence on the
  field-of-view's confidence distribution are relabeled to the sentinel
  `"Unknown"` — `ct_abstention_k=0.2` reproduces the paper headline operating
  point, and the pre-abstention argmax label is available via `cell_types_raw`
  when `return_probabilities=True`. (The default is `None` rather than on-by-
  default so the plain `list[str]` return is never silently rewritten at a
  benchmark-tuned threshold.)
- `TissueNetConfig(zarr_path=...)` defaults to `None` (was a hard-coded
  filesystem path that only resolved in one environment). When `None`,
  falls back to the `DEEPCELL_TYPES_ZARR_PATH` environment variable.
- Weights & Biases experiment logging is removed. The training and baseline
  scripts no longer accept `--enable_wandb`, `wandb` is no longer a `[train]`
  dependency, and `log_epoch_metrics` / `log_confusion_matrix` now log to the
  standard Python logger and save confusion-matrix images locally instead of
  uploading them.
- `mp_macro_precision` / `mp_macro_recall` now use `np.nanmean` to
  exclude vacuous markers, matching `mp_macro_f1`. Training runs no
  longer log `NaN` for these metrics.
- Numerical stability: padding-channel positions in `MarkerEmbeddingLayer`
  and the channel-spatial fusion now produce exactly zero output, so the
  `proj.bias`/`spatial_feat` contamination cannot reach the transformer
  for masked tokens.
- `TissueNetConfig.get_marker_positivity()` and `.marker_positivity_labels`
  now share a single `LazyMarkerPositivityDict`. Previously the dual-cache
  aliasing silently lost entries.

### Fixed
- `combined_celltypes.yaml` is now shipped inside the wheel
  (`training/config/*.yaml`); previously the file was outside the
  package tree and absent after `pip install`, causing
  `_get_combined_celltype_mapping()` to silently return `{}`.
- `tifffile` is now declared in `[train]` (TIFF-based ingest tooling
  imported it at module load without declaring it as a dependency).
- Stale fallback paths in `training/config.py` pointed at a sibling
  repository that is not bundled with this package and silently returned
  `{}` — these have been removed; the fallback now uses the in-package
  `CONFIG_DIR` only.
- Three broad `except Exception` blocks in `_load_tissuenet_archive`
  have been narrowed; a >1% per-archive drop-rate now raises rather
  than silently dropping hundreds of datasets.
- Test infrastructure: `_InferenceResultBuffer` (formerly `PredLogger`)
  is now private to `deepcell_types.predict`, eliminating the name
  collision with the training-side `training.utils.PredLogger`.
- **NumPy 2.0 compatibility** in the inference path: the per-channel
  normalization used `np.ptp`, removed as a free function in NumPy 2.0,
  so a fresh `pip install` (which pulls NumPy 2.x) raised `AttributeError`
  on the first `predict()`. Replaced with `max - min`.
- `--resume_path` now validates `n_heads` and `n_celltypes` (in addition
  to `resnet_channels`/`d_model`) before restoring optimizer state. Neither
  is recoverable from tensor shapes, so a mismatch previously restored an
  incompatible architecture silently (or surfaced only as a cryptic
  `load_state_dict` error).
- Dropped the experimental `--no_weighted_sampler` training flag: the
  canonical recipe is the decoupled two-stage `retrain_head.py` (backbone
  with the weighted sampler on, head retrained sampler-off), and end-to-end
  sampler-off training erodes the backbone, so the toggle was a dead path.

### Migration notes
- Users on `from deepcell_types.model import CellTypeCLIPModel` must
  switch to the canonical pipeline: `from deepcell_types import predict,
  DCTConfig`. There is no shim — the CLIP architecture is gone.
- `from deepcell_types.predict import PredLogger` is gone; the new
  private name is `_InferenceResultBuffer`. End users should not have
  been importing it.
- The `[train]` extra is required for anything under
  `deepcell_types.training`. Inference-only installs (`pip install
  deepcell-types`) are unchanged.

[0.1.0]: https://github.com/vanvalenlab/deepcell-types/tree/master
