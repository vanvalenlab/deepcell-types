# Changelog

All notable changes to `deepcell-types` are listed here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

This release consolidates the training and inference pipelines into a
single repository and switches the inference path to canonical-only
checkpoints. **Breaking changes** are noted below.

### Added
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
- **Breaking:** `predict(num_workers=...)` default is now `0` (was
  `24`). The `IterableDataset` patch generator held the whole FOV in
  memory; 24 workers reliably OOM'd machines with <64 GB RAM.
- **Breaking:** post-hoc cell-type abstention is now **on by default**
  (`predict(ct_abstention_k=0.2)`). Cells whose top-class probability
  falls below an IQR fence on the field-of-view's confidence
  distribution are relabeled to the sentinel `"Unknown"` (the
  pre-abstention argmax label is available via `cell_types_raw` when
  `return_probabilities=True`). This changes the returned labels for the
  same inputs relative to the unfiltered argmax behaviour of prior
  releases. Pass `ct_abstention_k=0` to recover the raw argmax label for
  every cell.
- `TissueNetConfig(zarr_path=...)` defaults to `None` (was a hard-coded
  lab-internal `/data2/...` path). When `None`, falls back to the
  `DEEPCELL_TYPES_ZARR_PATH` environment variable.
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
- Stale `deepcelltypes-kit` fallback paths in `training/config.py`
  pointed at a sibling repo that does not exist in the monorepo and
  silently returned `{}` — these have been removed; the fallback now
  uses the in-package `CONFIG_DIR` only.
- Three broad `except Exception` blocks in `_load_tissuenet_archive`
  have been narrowed; a >1% per-archive drop-rate now raises rather
  than silently dropping hundreds of datasets.
- Test infrastructure: `_InferenceResultBuffer` (formerly `PredLogger`)
  is now private to `deepcell_types.predict`, eliminating the name
  collision with the training-side `training.utils.PredLogger`.

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
