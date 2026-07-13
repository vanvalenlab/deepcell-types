# DeepCell Types

DeepCell Types is a generalized cell-phenotyping model for spatial proteomics.
It generalizes across datasets with varying marker panels by matching each
image's channels against a marker / cell-type registry that ships with the
package (`vocab.json`), so inference runs on any in-memory image with no extra
data download.

> **License notice.** Distributed under a *Modified Apache License, Version 2.0*
> with non-commercial / academic-only carve-outs (see the [LICENSE](LICENSE)
> file for the full text). For any other use, including commercial use, contact
> `vanvalenlab@gmail.com`.

## Installation

Install into a virtual environment (`venv`, `conda`/`mamba`, `uv`, `pixi` — your
choice):

```bash
python -m venv dct-env && source dct-env/bin/activate
pip install git+https://github.com/vanvalenlab/deepcell-types@master
```

## Download the model

Downloading the checkpoint requires a free access token — register at
[`users.deepcell.org`](https://users.deepcell.org) and export it (see
[`docs/site/API-key.md`](docs/site/API-key.md)):

```bash
export DEEPCELL_ACCESS_TOKEN=<your token>
```

```python
from deepcell_types.utils import download_model

# Downloads the latest checkpoint into ~/.deepcell/models and returns its path.
model_path = download_model()
```

## Running inference

Inference needs only the checkpoint and your image as an in-memory array — no
TissueNet archive required. `predict` resolves your channels against the packaged
`vocab.json` automatically:

```python
import torch
from deepcell_types import predict

# Default GPU if available, else CPU (same result, slower); use "cuda:1" etc. for a specific GPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# raw: numpy (C, H, W); mask: 2D label image; channel_names: list[str]; mpp: microns/pixel.
# model_name accepts the path from download_model() or a path to a .pt file.
labels = predict(raw, mask, channel_names, mpp, model_name=model_path, device=device)
```

See the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html)
for a complete walk-through.

## TissueNet zarr archive (optional)

Only needed to override the packaged registry (e.g. a custom marker panel) or for
[training](#training). When present, `predict` reads the registry from it instead
of `vocab.json` — pass `zarr_path=...` or set `DEEPCELL_TYPES_ZARR_PATH`.
Registered users can download a public archive from `https://users.deepcell.org`
(see [`docs/site/API-key.md`](docs/site/API-key.md)):

```bash
export DEEPCELL_TYPES_ZARR_PATH=/absolute/path/to/tissuenet.zarr
```

Before publishing an archive or checkpoint, validate that its marker order/size
matches the released embeddings — drift silently breaks inference:

```bash
scripts/check_release_archive.sh /path/to/tissuenet.zarr /path/to/svd_512.npz
```

It exits non-zero on any drift (logic unit-tested in CI via
`tests/test_archive_contract_validator.py`).

## Custom preprocessing (advanced)

When a FOV's predictions look implausible — usually a saturated or high-background
channel steering the calls — adapt that FOV's per-channel normalization. **Start
with the `preproc-adapt` skill** (`skills/preproc-adapt/`): an agent-driven loop
that diagnoses the offending channel/op and iterates the config for you.

It drives `predict`'s optional `preprocess` hook, which you can also build directly
from a bounded set of ops:

```python
from deepcell_types import predict, make_preprocessor

config = [
    {"op": "clip_percentile", "p": 99.9},
    {"op": "channel_drop", "names": ["NeuN"]},  # drop a confounding marker
    {"op": "min_max_normalize"},                # model sees [0, 1]
]
labels = predict(raw, mask, channel_names, mpp, model_name=...,
                 device=device, preprocess=make_preprocessor(config))
```

The hook receives the resampled in-vocabulary `(C, H, W)` array and must return a
`(C, H, W)` array in `[0, 1]`. With `preprocess=None` (default) the built-in p99.9
clip + min-max is used.

## Training

Install the `[train]` extra (adds `zarr`, `pandas`, `scikit-learn`,
`torchmetrics`, plotly, …):

```bash
pip install "deepcell-types[train] @ git+https://github.com/vanvalenlab/deepcell-types@master"
```

Entry points under `scripts/`:

- `train.py` — main training loop (stage 1: backbone, weighted sampler on).
- `retrain_head.py` — stage 2: freeze the backbone and retrain the residual-MLP
  head on the natural class distribution (sampler off). This decoupled recipe is
  the default and produces the best model; the resMLP head is auto-detected at
  inference.
- `pretrain.py` — masked-marker pretraining.
- `predict.py` — batched evaluation over a zarr archive.
- `evaluate_on_test.sh` — canonical evaluation on the held-out 129-FOV test split
  (`splits/fov_split_test_current.json`), where the two-stage resMLP recipe reaches
  **80.27 hierarchical macro-F1** (full-coverage, no abstention), ahead of a tuned
  XGBoost baseline (79.03) and the other paper baselines.

Training scripts read config from a TissueNet zarr v3 archive; pass `--zarr_dir`
or set `DEEPCELL_TYPES_ZARR_PATH`. The `deepcell_types.training` modules
(`TissueNetConfig`, `FullImageDataset`, `FocalLoss`, `HierarchicalLoss`) can be
imported directly for custom scripts.

## Baselines

All four paper comparison baselines live in `deepcell_types.baselines` and run via
`python -m deepcell_types.baselines <name>` (no submodules).

> **Fairness contract.** For an apples-to-apples comparison, run every baseline
> and the main model with the same class-balancing sampler and model-selection
> split: `--class_balance dct` (default) and
> `--val_split_file splits/fov_split_valsubset.json` (seed 42), then evaluate on
> `splits/fov_split_test_current.json`. These flags are opt-in — parity depends on
> passing them consistently to every method.

- **XGBoost** — on mean-marker-intensity features.
  ```bash
  pip install -e ".[baseline-xgboost]"
  python -m deepcell_types.baselines xgboost ...      # or: xgboost-tune
  ```
- **Nimbus** — UNet marker-positivity
  ([Rumberger et al., *Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02826-9)).
  Pins `nimbus-inference==0.0.5` (requires Python <3.12).
  ```bash
  pip install -e ".[baseline-nimbus]"
  python -m deepcell_types.baselines nimbus ...
  ```
- **MAPS** — MLP classifier
  ([*Nature Communications* 2023](https://doi.org/10.1038/s41467-023-44188-w)).
  ```bash
  pip install -e ".[baseline-maps]"
  python -m deepcell_types.baselines maps ...
  ```
- **CellSighter** — ResNet-50 multiplexed classifier
  ([Amitay et al., *Nature Communications* 2023](https://doi.org/10.1038/s41467-023-40066-7));
  pulls in `torchvision`.
  ```bash
  pip install -e ".[baseline-cellsighter]"
  python -m deepcell_types.baselines cellsighter ...
  ```

## Citation

```
@article{deepcelltypes,
  title={Generalized cell phenotyping for spatial proteomics with language-informed vision models},
  author={Wang, Xuefei and Dilip, Rohit and Iqbal, Ahamed Raffey and Bussi, Yuval and Brown, Caitlin and Pradhan, Elora and Jain, Yashvardhan and Yu, Kevin and Li, Shenyi and Abt, Martin and Borner, Katy and Keren, Leeat and Yue, Yisong and Barnowski, Ross and Van Valen, David},
  journal={bioRxiv},
  pages={2026--07},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  url={https://www.biorxiv.org/content/10.1101/2024.11.02.621624v4}
}
```
