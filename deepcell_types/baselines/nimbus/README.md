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
  (`run.py:136`, `run.py:189`). A single uniform threshold is applied; this
  wrapper does not consume a per-marker `--mp_threshold_file`.
  - **Cross-model-fairness note:** for a like-for-like comparison the main model
    must be scored with the *same* fixed threshold (its default is also `0.5` —
    do **not** report it with `--learn_mp_thresholds`) on the *same* val FOVs.
    This wrapper has no `--split_file`; FOV-set parity with the main model is the
    caller's responsibility (`--keep_datasets`).
- **Ambiguous-coded ground truth is dropped** to match the canonical Nimbus
  evaluation: GT values of `0.5` or `2` are excluded and only strict `0/1`
  labels are scored (`run.py:157-173`).
- **Prediction resize uses bilinear, not nearest-neighbor.** When resizing the
  model output back to mask resolution for per-cell aggregation we use
  `cv2.INTER_LINEAR` (`run.py:573-577`); upstream `nimbus-inference==0.0.5` uses
  `cv2.INTER_NEAREST` (`utils.py:661`). The binary-mask and marker-image resizes
  match upstream (nearest / bilinear respectively).
- **mpp-based rescale parameterization.** We compute the scale factor from
  microns-per-pixel (`scale = img_mpp / target_mpp`, `run.py:397,522`) rather than
  upstream's magnification ratio (`model_magnification / dataset.magnification`,
  `utils.py:636`). These coincide when `mpp = 10 / magnification`; they can differ
  for archive FOVs whose stored `mpp` is not the reciprocal of the acquisition
  magnification.

Verified against the installed `nimbus-inference==0.0.5` wheel. The library's
core primitives are used directly, so they are faithful by construction: the
UNet's own `torch.sigmoid` output is used as-is with no double activation, and
boundary erosion calls upstream `prepare_binary_mask` (`run.py:329,489`).
Normalization reproduces upstream's cross-FOV averaged 99.9th-percentile
foreground statistics (`quantile=0.999`, `n_subset=10`).
