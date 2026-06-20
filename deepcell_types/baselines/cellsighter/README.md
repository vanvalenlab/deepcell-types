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

- Consumes multi-channel image patches (`--crop_size`, default **60×60**) from
  the same train/validation/test split, and is scored with the same hierarchical
  accuracy and macro/weighted metrics as the other baselines.
- For zero-shot evaluation, panels are aligned to the global marker vocabulary
  by scattering each dataset's channels to their global `marker2idx` positions;
  absent markers are zero-padded (`model.py`, `convert_batch_for_cellsighter`).

## Deviations / adaptations (recorded for reproducibility)

- **Backbone: torchvision ResNet-50** (`model.py:46-48`), default
  `model_size="resnet50"`.
- **Original ImageNet ResNet-50 stem on 60×60 patches** (7×7 stride-2 conv +
  max-pool, `model.py`, `cifar_stem=False` default), matching upstream
  CellSighter's `crop_input_size: 60`. A `--cifar_stem` ablation flag swaps in a
  3×3 stride-1 / no-max-pool stem appropriate only for 32×32 crops.
- **Input channels = `NUM_MARKERS + 2`** — the globally aligned marker channels
  plus the cell mask and neighbor mask.
- **Trained from random initialization** (`pretrained=False`, `weights=None`,
  `model.py:36-47`), matching upstream (no ImageNet weights).
- **50 epochs, Adam, constant learning rate `1e-3`, no scheduler.** The upstream
  repo constructs an `ExponentialLR` scheduler but never calls
  `scheduler.step()`, so it trains at constant LR; we reproduce that exactly and
  do not step a scheduler (`run.py:389-393`).
- **Best epoch selected on a held-out inner-validation set by macro-F1**;
  validation runs every `val_every_n_epochs` (default 10) plus the final epoch.
  (Upstream CellSighter validates every epoch; we deviate to every-N for cost.)
  - **Deviation from upstream:** the original CellSighter selects the best epoch
    on the same set it reports. We instead carve a FOV-grouped inner-validation
    set (10%, via `create_dataloader(inner_val_ratio=0.1)`) out of the training
    FOVs and select on it, so the reported test set never drives checkpoint
    selection (selection-on-the-reported-set is leakage). This mirrors the
    XGBoost baseline's FOV-grouped early-stopping set. As a consequence the
    model now trains on ~90% of the training cells (the inner-val FOVs are held
    out). Selection uses macro-F1 — the headline metric, matching the main
    model (`scripts/train.py` selects on `val_macro_f1`) and the other
    baselines; upstream CellSighter instead selects on validation loss.
- **Class balancing — faithful equal-proportion (default).** `--class_balance
  equal` (default) reproduces the upstream training recipe: the train pool is
  first capped to `--size_data` cells/class (default 1000, matching the paper's
  `size_data` / `subsample_const_size`), then a `WeightedRandomSampler` draws
  with full-inverse-frequency weights `weight = total / count`
  (`deepcell_types/training/samplers.py:compute_sample_weights_equal`, matching upstream `define_sampler`
  with `sample_batch: true`). Two ablation schemes are selectable: `--class_balance
  sqrt` (the DCT-wide sqrt-inverse-frequency sampler with a 1000-count floor) and
  `--class_balance none` (uniform; also reachable via the deprecated
  `--no_weighted_sampler`). Remaining deviation: upstream's `hierarchy_match`
  balances at the lineage level, whereas balancing here is computed on the
  fine-grained standardized cell-type label.
