# DeepCell Types

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics that addresses the challenge of generalization across diverse datasets with varying marker panels.

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

## Running

The `deepcell-types` cell-type inference functionality is provided via
a simple functional interface, `deepcell_types.predict`.

Canonical checkpoints read their marker and cell-type registry from a
TissueNet zarr archive at inference time. Pass the archive with
`zarr_path=...` or set `DEEPCELL_TYPES_ZARR_PATH` before calling
`deepcell_types.predict`.

For a complete example of the cell-type inference pipeline, check out
the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html).

## Training

To retrain or fine-tune, install the `[train]` extra (pulls in `wandb`,
`zarr`, `pandas`, `scikit-learn`, `torchvision`, `torchmetrics`, etc.):

```bash
pip install "deepcell-types[train] @ git+https://github.com/vanvalenlab/deepcell-types@master"
```

Training entry points live under `scripts/`:

- `scripts/train.py` — main training loop.
- `scripts/pretrain.py` — masked-marker pretraining.
- `scripts/predict.py` — batched evaluation over a zarr archive.
- `scripts/benchmark_gold_standard.py` — gold-standard benchmark suite.

All training scripts read configuration from a TissueNet zarr v3 archive.
Pass `--zarr_path` or set `DEEPCELL_TYPES_ZARR_PATH`. The training-side
modules under `deepcell_types.training` (e.g. `TissueNetConfig`,
`FullImageDataset`, `FocalLoss`, `HierarchicalLoss`) are stable enough to
import directly for custom training scripts.

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
