# Canonical FOV splits

These JSON manifests are the canonical train / validation / test field-of-view
(FOV) splits used for all reported numbers in the paper. They are committed so
results are reproducible without access to the raw archive.

The split is constructed in two stages (seed 42; see `scripts/generate_splits.py`
for stage 1 and `scripts/split_val_for_test.py` for stage 2):

| File | `train` | `val` | `heldout` | Purpose |
|------|--------:|------:|----------:|---------|
| `fov_split_v10.json` | 1722 | 431 | — | Stage 1: stratified (modality, tissue) 0.8/0.2 hold-out |
| `fov_split_v10_valsubset.json` | 1722 | 302 | 129 | Training: `val` = model-selection / early-stopping set |
| `fov_split_v10_test.json` | 1722 | 129 | 302 | Final test: `val` = frozen 129-FOV test set (never used for selection) |

`v10` refers to the TissueNet archive schema version (`tissuenet-v10.zarr`); the
FOV keys are bound to that archive. The `heldout` key records the FOVs withheld
from each subset so strict split loading can distinguish intentional exclusions
from a stale manifest.
