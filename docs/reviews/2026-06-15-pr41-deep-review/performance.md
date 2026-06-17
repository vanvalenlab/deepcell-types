# Performance Audit — deepcell-types v0.1.0 (PR #41)

(0 blockers, 3 highs, 4 mediums, 2 lows)

## HIGH: `_calculate_marker_positivity` rebuilds two O(C) dicts on every `__getitem__`
**Location:** `deepcell_types/training/dataset.py:545-547`
`row_lookup`/`col_lookup` rebuilt per cell (millions of times/worker) from shared DataFrames.
**Recommendation:** Build once per dataset at `__init__`.

## HIGH: `PatchDataset` worker sharding runs the full O(N_cells) generator in every worker, discarding (N-1)/N
**Location:** `deepcell_types/dataset.py:140-155`
Each worker runs `patch_generator` to completion (per-cell crop, distance transform, resize) and yields only its `patch_idx % num_workers` slice. At workers=8, ~87% of crop compute wasted.
**Recommendation:** Precompute `(centroid, cell_idx)` records in `__init__`; shard by list position `records[worker_id::num_workers]`.

## HIGH: `patch_generator` runs full-FOV rescale + per-channel percentile normalize synchronously / redundantly
**Location:** `deepcell_types/preprocessing.py:316-348`
With workers=0, multi-second CPU work blocks the GPU; with workers>0 every worker repeats it.
**Recommendation:** Move rescale+normalize into `PatchDataset.__init__`, store `self.raw_processed`.

## MEDIUM: `compute_sample_weights` two O(N_train) Python loops at construction
**Location:** `samplers.py:19-52` — vectorize with numpy.

## MEDIUM: `FOVGroupedSampler.__iter__` `torch.cat` + `.tolist()` materializes full-epoch index twice
**Location:** `samplers.py:138-144` — yield lazily from group boundaries.

## MEDIUM: `PredictionResult` holds the full (N, 51) softmax in RAM (~1.4 GB at 7M cells)
**Location:** `predict.py:402-408` — when `return_probabilities=False`, don't accumulate full probs.

## MEDIUM: `SequentialFOVGroupedSampler.__iter__` rebuilds the full group dict every epoch
**Location:** `samplers.py:191-201` — build `self._groups` once in `__init__`, re-shuffle keys only.

## LOW: `mask_marker_channels` per-sample Python loop (pretraining-only)
**Location:** `model.py:730-745`.

## LOW: `DataLoaderConfig` defaults `num_workers=16`, `persistent_workers=False` → spawn/reap every epoch for dataclass-API users
**Location:** `dataloader.py:307`.

## Strengths
FOV-grouped sampler + per-worker LRU numpy cache (correct cold-zarr answer, ~300×). `__getstate__` clears caches before worker pickling. `SequentialFOVGroupedSampler` avoids the shuffle=True I/O storm. `num_workers=0` predict default is a sound memory tradeoff. scatter_ sink-column fix. `prefetch_factor=4` on train path.
