# DeepCell Types

DeepCell Types is a generalized cell-phenotyping model for spatial
proteomics. It addresses generalization across datasets with varying
marker panels by reading the active marker / cell-type registry from a
TissueNet zarr archive at inference time.

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

## TissueNet zarr archive

Canonical checkpoints do not embed the marker / cell-type registry —
they read it from a **TissueNet zarr v3 archive** at inference time. You
must provide one before calling `predict`, either by passing
`zarr_path=...` directly or by setting the `DEEPCELL_TYPES_ZARR_PATH`
environment variable.

A registered user can download a public TissueNet zarr archive from
`https://users.deepcell.org`; see `docs/site/API-key.md` for the access
token flow. Place the resulting `.zarr` directory anywhere, then:

```bash
export DEEPCELL_TYPES_ZARR_PATH=/absolute/path/to/tissuenet.zarr
```

## Running

The `deepcell-types` cell-type inference functionality is provided via
a simple functional interface, `deepcell_types.predict`.

For a complete example of the cell-type inference pipeline, check out
the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html).

## Training

To retrain or fine-tune, install the `[train]` extra (pulls in `wandb`,
`zarr`, `pandas`, `scikit-learn`, `torchmetrics`, plotly, etc.):

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

Comparison baselines from the paper live as git submodules under
`baselines/`:

- `baselines/cellsighter/` — ResNet-50 multiplexed cell classifier
  ([Amitay et al., *Nature Communications* 2023](https://doi.org/10.1038/s41467-023-40066-7)).
- `baselines/maps/` — MAPS MLP classifier
  ([*Nature Communications* 2023](https://doi.org/10.1038/s41467-023-44188-w)).
- `baselines/nimbus/` — Nimbus UNet marker-positivity baseline
  ([Rumberger et al., *Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02683-6)).
- `baselines/xgboost/` — XGBoost on mean-marker-intensity features.

Each submodule has its own README and `pyproject.toml`. To populate
them after cloning, run `git submodule update --init --recursive`.

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
