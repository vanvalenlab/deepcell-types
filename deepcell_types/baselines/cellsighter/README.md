# CellSighter baseline

Re-implementation of CellSighter (a CNN that classifies cell types directly from
multi-channel image patches) against the DeepCell Types data loader, marker
vocabulary, and cell-type labels.

- Reference paper: Amitay et al., *Nature Communications* 2023, DOI:
  [10.1038/s41467-023-40066-7](https://doi.org/10.1038/s41467-023-40066-7)
- Upstream code: <https://github.com/KerenLab/CellSighter>

Run with:

```bash
pip install -e ".[baseline-cellsighter]"
python -m deepcell_types.baselines cellsighter ...
```

## Shared interface (identical to DeepCell Types)

- Consumes the same 32×32 multi-channel patches, the same train/validation/test
  split, and is scored with the same hierarchical accuracy and macro/weighted
  metrics as the other baselines.
- For zero-shot evaluation, panels are aligned to the global marker vocabulary
  by scattering each dataset's channels to their global `marker2idx` positions;
  absent markers are zero-padded (`model.py:94-111`).

## Deviations / adaptations (recorded for reproducibility)

- **Backbone: torchvision ResNet-50** (`model.py:46-48`), default
  `model_size="resnet50"`.
- **Custom CIFAR-style stem sized to 32×32 patches**: the ImageNet stem
  (7×7 stride-2 conv + max-pool) collapses a 32×32 input to 1×1 by `layer4`, so
  it is replaced by a single **3×3 stride-1 conv with no max-pool**
  (`model.py:54-57`); the spatial path becomes 32→32→16→8→4→1. This matches the
  small-patch adaptation in the upstream CellSighter recipe.
- **Input channels = `NUM_MARKERS + 2`** — the globally aligned marker channels
  plus the cell mask and neighbor mask.
- **Trained from random initialization** (`pretrained=False`, `weights=None`,
  `model.py:36-47`), matching upstream (no ImageNet weights).
- **50 epochs, Adam, constant learning rate `1e-3`, no scheduler.** The upstream
  repo constructs an `ExponentialLR` scheduler but never calls
  `scheduler.step()`, so it trains at constant LR; we reproduce that exactly and
  do not step a scheduler (`run.py:389-393`).
- **Best epoch selected on a held-out inner-validation set by macro-accuracy**;
  validation runs every `val_every_n_epochs` (default 10) plus the final epoch,
  matching the upstream cadence.
  - **Deviation from upstream:** the original CellSighter selects the best epoch
    on the same set it reports. We instead carve a FOV-grouped inner-validation
    set (10%, via `create_dataloader(inner_val_ratio=0.1)`) out of the training
    FOVs and select on it, so the reported test set never drives checkpoint
    selection (selection-on-the-reported-set is leakage). This mirrors the
    XGBoost baseline's FOV-grouped early-stopping set. As a consequence the
    model now trains on ~90% of the training cells (the inner-val FOVs are held
    out). Selection still uses macro-accuracy; switching it to macro-F1 (the
    headline metric) is a separate, not-yet-applied change.
