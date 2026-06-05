# Nimbus baseline

Wraps the published Nimbus inference library and applies the pretrained Nimbus
model to the DeepCell Types evaluation harness. Unlike the XGBoost / MAPS /
CellSighter baselines (which predict **cell type**), Nimbus is a **marker
positivity** baseline: it predicts per-cell, per-marker positivity, which we
compare against the curated ground-truth marker-positivity labels.

- Reference paper: Rumberger et al., *Nature Methods* 2025, DOI:
  [10.1038/s41592-025-02826-9](https://doi.org/10.1038/s41592-025-02826-9)
- Upstream code: <https://github.com/angelolab/Nimbus-Inference>

Run with:

```bash
pip install -e ".[baseline-nimbus]"
python -m deepcell_types.baselines nimbus ...
```

> **Environment note:** `baseline-nimbus` pins `nimbus-inference==0.0.5`, which
> requires Python < 3.12. Use a Python 3.11 environment for this baseline.

## Shared interface (identical to DeepCell Types)

- Reads the same zarr-format Expanded TissueNet archive and the same curated
  ground-truth marker-positivity labels used to score the DeepCell Types
  marker-positivity head.

## Deviations / adaptations (recorded for reproducibility)

- **Inference only** — the pretrained Nimbus model is run as-is; no retraining
  on Expanded TissueNet.
- **Binary threshold 0.5** on the continuous Nimbus output to call positivity
  (`run.py:136`, `run.py:189`).
- **Ambiguous-coded ground truth is dropped** to match the canonical Nimbus
  evaluation: GT values of `0.5` or `2` are excluded and only strict `0/1`
  labels are scored (`run.py:157-173`).
