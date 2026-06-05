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

## Faithful-to-upstream choices (recorded for reproducibility)

- **Input schema: `NUM_MARKERS + 1` = 279-dimensional** — per-cell mean
  intensities plus the per-cell **size** scalar appended as the last column
  (`run.py:289`, `run.py:321-326`). The cell-size feature is part of the
  canonical mahmoodlab/MAPS recipe (its `data_preprocessing/*.py` emits
  `N markers + cellSize`), not a DeepCell Types addition.
- **Feature normalization `((x - μ) / σ) / 255`** (`run.py:86`). The `1/255`
  factor is part of the canonical mahmoodlab/MAPS pipeline and is required to
  reproduce the published accuracy; train statistics (`μ`, `σ`) are applied to
  the test set.
- **Optimizer: Adam, constant learning rate `1e-3`, no scheduler**
  (`run.py:407`), matching the upstream default.
- **Model: single hidden layer of width 512, dropout 0.25** (`model.py:28-29`),
  matching the upstream MAPS MLP.
- **50 epochs, best epoch selected on validation loss** (`run.py:216-219`,
  `run.py:409-463`). The full epoch count is run (no early stopping); the
  lowest-val-loss checkpoint is kept.
