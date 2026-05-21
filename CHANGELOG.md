# Changelog

All notable changes to `deepcell-types` are listed here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-11

This release consolidates the training and inference pipelines into a
single repository and switches the inference path to canonical-only
checkpoints. **Breaking changes** are noted below.

### Added
- Training pipeline (`deepcell_types.training`) is now shipped from this
  repository, gated behind the `[train]` install extra:
  ```bash
  pip install "deepcell-types[train]"
  ```
  Includes `TissueNetConfig`, `FullImageDataset`, `FOVGroupedSampler`,
  `FocalLoss`, `HierarchicalLoss`, `MPMetricsTracker`, and the new
  focused submodules `training/{archive,patch,metrics,baseline_features}.py`.
- End-to-end training scripts under `scripts/` (`train.py`,
  `pretrain.py`, `predict.py`, `benchmark_gold_standard.py`,
  `ingest_gold_to_zarr.py`).
- Public top-level exports: `DCTConfig` and `PredictionResult` are now
  importable from `deepcell_types` directly.
- `predict(return_probabilities=True)` returns a `PredictionResult`
  dataclass with the full per-cell softmax matrix and cell indices.
- Four baseline submodules (`baselines/{cellsighter,maps,nimbus,xgboost}/`)
  tracking the `main` branch of their respective forks.

### Changed
- **Breaking:** the legacy `CellTypeCLIPModel` class is removed.
  Canonical checkpoints now read marker / cell-type metadata from a
  TissueNet zarr v3 archive at inference time; the active model class
  is `CellTypeAnnotator` in `deepcell_types/model.py`.
- **Breaking:** `predict(tissue_exclude=...)` is renamed to
  `predict(tissue_filter=...)`. The old name was semantically inverted
  (it reads as "exclude this tissue" but actually "filter TO this
  tissue"). `tissue_exclude` is still accepted (keyword-only) for one
  release and emits a `DeprecationWarning`.
- **Breaking:** `predict(num_workers=...)` default is now `0` (was
  `24`). The `IterableDataset` patch generator held the whole FOV in
  memory; 24 workers reliably OOM'd machines with <64 GB RAM.
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
- `tifffile` is now declared in `[train]` (was imported at module load
  by `scripts/ingest_gold_to_zarr.py` without being a dependency).
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

[0.1.0]: https://github.com/vanvalenlab/deepcell-types/releases/tag/v0.1.0
