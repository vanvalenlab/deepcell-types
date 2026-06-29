# Canonical FOV splits

These JSON manifests are the canonical train / validation / test field-of-view
(FOV) splits used for all reported numbers in the paper. They are committed so
results are reproducible without access to the raw archive.

The split is constructed in two stages (seed 42; see `scripts/generate_splits.py`
for stage 1 and `scripts/split_val_for_test.py` for stage 2):

| File | `train` | `val` | `heldout` | Archive | Purpose |
|------|--------:|------:|----------:|---------|---------|
| `fov_split.json` | 1722 | 431 | — | `9f8cc9d6` | Stage 1: stratified (modality, tissue) 0.8/0.2 hold-out |
| `fov_split_valsubset.json` | 1722 | 302 | 129 | `9f8cc9d6` | Training: `val` = model-selection / early-stopping set |
| `fov_split_test.json` | 1722 | 129 | 302 | `9f8cc9d6` | Final test: `val` = frozen 129-FOV test set (never used for selection) |
| `fov_split_test_current.json` | 1722 | 129 | 302 | `f5b6ed52` | Same frozen 129-FOV test set, rebound to the **current** released archive. **This is the default headline-evaluation split** (`scripts/evaluate_on_test.sh`). |

The "Archive" column is the archive fingerprint each manifest's FOV keys are
bound to. The first three were built against an earlier archive snapshot
(`9f8cc9d6`); `fov_split_test_current.json` rebinds the *same* frozen 129-FOV
test set to the current released `tissuenet-v10.zarr` (`f5b6ed52`) — its 129
test FOVs are identical to `fov_split_test.json`, so test results compare
directly. Use `fov_split_test_current.json` against the released archive.

The `heldout` key records the FOVs withheld from each subset so strict split
loading can distinguish intentional exclusions from a stale manifest.
