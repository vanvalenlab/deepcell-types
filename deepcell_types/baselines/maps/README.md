# MAPS baseline

Re-implementation of MAPS (an MLP cell-type classifier on per-cell mean
intensities) inside the DeepCell Types training framework.

- Reference paper: *Nature Communications* 2023, DOI:
  [10.1038/s41467-023-44188-w](https://doi.org/10.1038/s41467-023-44188-w)
- Upstream code: <https://github.com/mahmoodlab/MAPS>

Run with:

```bash
pip install -e ".[baseline-maps]"
python -m deepcell_types.baselines maps ...
```

## Shared interface (identical to DeepCell Types)

- Reads the same zarr-format Expanded TissueNet archive, the same universal
  marker vocabulary, the same train/validation/test split, and is scored with
  the same hierarchical accuracy and macro/weighted metrics.

## Upstream-derived choices and DCT adaptations

- **Input schema: `NUM_MARKERS + 1` = 279-dimensional** — per-cell mean
  intensities plus the per-cell **size** scalar appended as the last column
  (`run.py`, cell-size append). The cell-size feature is part of the canonical
  mahmoodlab/MAPS recipe (its `data_preprocessing/*.py` emits
  `N markers + cellSize`), not a DeepCell Types addition.
- **Feature normalization: DCT-safe `((x - mu) / sigma) / 255` by default;
  `--no_znorm` for `/255` only.** DeepCell Types marker means come from
  preprocessed `[0, 1]` images while `cellSize` is a raw pixel count, so the
  default adapter keeps train-set z-score normalization on before `/255` to
  control the relative feature scales. `/255`-only remains available via
  `--no_znorm` as an upstream-provenance ablation, and reported MAPS numbers
  should include the exact command/config.
- **Optimizer: Adam, constant learning rate `1e-3`, no scheduler**
  (`run.py`), matching the upstream default.
- **Model: four 512-wide hidden layers, dropout 0.25** (`model.py:42-56`),
  matching the upstream MAPS MLP (`networks.py:22-36`).
- **Up to 500 epochs with early stopping, best epoch selected on a held-out
  inner-validation set** (`--max_epochs 500`, `--min_epochs 250`, `--patience
  100`; epoch budget derived from canonical mahmoodlab/MAPS `trainer.py`).
  Training early-stops on inner-val loss once past `min_epochs`; the
  lowest-inner-val-loss checkpoint is kept and then evaluated once on the
  reported test set.
  - **Deviation from upstream:** canonical mahmoodlab/MAPS selects the best
    epoch on the same set it reports. We instead carve a FOV-grouped
    inner-validation set (10%, `GroupShuffleSplit`) out of the training FOVs and
    select on it, so the reported test set never drives checkpoint selection
    (selection-on-the-reported-set is leakage). This mirrors the XGBoost
    baseline's FOV-grouped early-stopping set. As a consequence the model now
    trains on ~90% of the training cells (the inner-val FOVs are held out).
