# DeepCell Types

DeepCell Types is a generalized cell-phenotyping model for spatial
proteomics. It addresses generalization across datasets with varying
marker panels by matching each image's channels against a marker /
cell-type registry, which ships with the package as a `vocab.json`
snapshot — so inference runs on any in-memory image with no extra data
download.

> **License notice.** This project is distributed under a *Modified Apache
> License, Version 2.0* with non-commercial / academic-only carve-outs (see
> the [LICENSE](LICENSE) file for the full text). For any other use,
> including commercial use, contact `vanvalenlab@gmail.com`.

## Installation

As with all Python packages, users are encouraged to use some form
of virtual environment for package installation.
Popular options include `venv`/`virtualenv`, `conda`/`mamba`, `uv`,
or `pixi`.
Users are encouraged to use whatever environment management toolchain
they are most comfortable with.
For those unsure, the quickest way to start is to use the `venv` module,
part of the Python standard library:

```bash
# Create a new virtual environment
python -m venv dct-env
# Enter the virtual environment
source dct-env/bin/activate

# Once inside the environment, install deepcell-types
pip install git+https://github.com/vanvalenlab/deepcell-types@master
```

## Download the model
```python
from deepcell_types.utils import download_model
download_model()
```

## Running inference (no archive required)

Inference needs only the model checkpoint (`download_model()` above) and
your image as an in-memory array — **no TissueNet archive download is
required**. The marker / cell-type registry ships inside the package as a
`vocab.json` snapshot, and `predict` resolves your channels against it
automatically:

```python
from deepcell_types import predict

# raw: numpy (C, H, W); mask: 2D label image; channel_names: list[str]
labels = predict(raw, mask, channel_names, mpp, device="cuda:0")
```

For a complete example of the cell-type inference pipeline, check out
the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html).

## TissueNet zarr archive (optional)

The archive is **only** needed if you want to override the packaged
registry — e.g. to run against a custom marker panel — or for training (see
[Training](#training)). When present, `predict` reads the registry from it
instead of `vocab.json`; pass `zarr_path=...` directly or set the
`DEEPCELL_TYPES_ZARR_PATH` environment variable.

A registered user can download a public TissueNet zarr archive from
`https://users.deepcell.org`; see `docs/site/API-key.md` for the access
token flow. Place the resulting `.zarr` directory anywhere, then:

```bash
export DEEPCELL_TYPES_ZARR_PATH=/absolute/path/to/tissuenet.zarr
```

### Validating an archive before publishing

The archive's `all_standardized_channels` attribute *is* the model's
marker→index map, so it must match the order the released checkpoint and
`embeddings/svd_512.npz` were built with — reordering or resizing it silently
breaks inference. Before publishing an archive (or a checkpoint), run the
release gate against the real archive and the released embeddings:

```bash
scripts/check_release_archive.sh /path/to/tissuenet.zarr /path/to/svd_512.npz
```

It exits non-zero on any marker-order/size drift. (The check's logic is
unit-tested in CI via `tests/test_archive_contract_validator.py`; this script
runs it against the actual archive, which CI cannot access.)

## Custom preprocessing (advanced)

When a single FOV's predictions look biologically implausible — usually because
one channel is saturated or has heavy background and is steering the calls — the
fix is to adapt the per-channel normalization for that FOV.

**Start with the `preproc-adapt` skill** (`skills/preproc-adapt/`). It is the
recommended, first-and-foremost way to do this: an agent-driven loop that
inspects the panel, predicts, checks the cell-type composition against tissue
biology, diagnoses *which* channel/op is the problem, and iterates the config for
you — so you don't have to guess the ops by hand. Use it before hand-tuning.

Under the hood the skill drives `predict`'s optional `preprocess` hook, which you
can also call directly if you already know the fix. Build one declaratively from a
bounded set of ops:

```python
from deepcell_types import predict, make_preprocessor

config = [
    {"op": "clip_percentile", "p": 99.9},
    {"op": "channel_drop", "names": ["NeuN"]},  # drop a confounding marker
    {"op": "min_max_normalize"},                # model sees [0, 1]
]
labels = predict(raw, mask, channel_names, mpp, model_name=...,
                 device="cuda:0", preprocess=make_preprocessor(config))
```

The hook receives the resampled, in-vocabulary raw `(C, H, W)` array and the
resolved marker names, and must return a `(C, H, W)` array in `[0, 1]`. With
`preprocess=None` (default) the built-in p99.9 clip + min-max is used;
`make_preprocessor(DEFAULT_CONFIG)` reproduces that default exactly.

## Training

To retrain or fine-tune, install the `[train]` extra (pulls in `zarr`,
`pandas`, `scikit-learn`, `torchmetrics`, plotly, etc.):

```bash
pip install "deepcell-types[train] @ git+https://github.com/vanvalenlab/deepcell-types@master"
```

Training entry points live under `scripts/`:

- `scripts/train.py` — main training loop.
- `scripts/pretrain.py` — masked-marker pretraining.
- `scripts/predict.py` — batched evaluation over a zarr archive.

All training scripts read configuration from a TissueNet zarr v3 archive.
Pass `--zarr_dir` (training scripts) or set `DEEPCELL_TYPES_ZARR_PATH`
(picked up by `deepcell_types.predict`). The training-side
modules under `deepcell_types.training` (e.g. `TissueNetConfig`,
`FullImageDataset`, `FocalLoss`, `HierarchicalLoss`) are stable enough to
import directly for custom training scripts.

## Baselines

All four paper comparison baselines are folded into `deepcell_types.baselines`
and run via the unified runner `python -m deepcell_types.baselines <name>`.
No submodules are required.

- **XGBoost** — XGBoost on mean-marker-intensity features.

  ```bash
  pip install -e ".[baseline-xgboost]"
  python -m deepcell_types.baselines xgboost ...
  python -m deepcell_types.baselines xgboost-tune ...
  ```

- **Nimbus** — Nimbus UNet marker-positivity baseline
  ([Rumberger et al., *Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02826-9)).

  ```bash
  pip install -e ".[baseline-nimbus]"
  python -m deepcell_types.baselines nimbus ...
  ```

  > **Note:** `baseline-nimbus` pins `nimbus-inference==0.0.5`, which
  > requires Python <3.12. Use a Python 3.11 environment for this baseline.

- **MAPS** — MAPS MLP classifier
  ([*Nature Communications* 2023](https://doi.org/10.1038/s41467-023-44188-w)).

  ```bash
  pip install -e ".[baseline-maps]"
  python -m deepcell_types.baselines maps ...
  ```

- **CellSighter** — ResNet-50 multiplexed cell classifier
  ([Amitay et al., *Nature Communications* 2023](https://doi.org/10.1038/s41467-023-40066-7)).
  Pulls in `torchvision`.

  ```bash
  pip install -e ".[baseline-cellsighter]"
  python -m deepcell_types.baselines cellsighter ...
  ```

## Citation
```
@article{deepcelltypes,
  title={Generalized cell phenotyping for spatial proteomics with language-informed vision models},
  author={Wang, Xuefei and Dilip, Rohit and Bussi, Yuval and Brown, Caitlin and Pradhan, Elora and Jain, Yashvardhan and Yu, Kevin and Li, Shenyi and Abt, Martin and Borner, Katy and others},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
