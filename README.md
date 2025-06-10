# DeepCell Types

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics that addresses the challenge of generalization across diverse datasets with varying marker panels. 


## Download the model
```python
from deepcell_types.utils import download_model
download_model()
```

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

## Running

The `deepcell-types` cell-type inference functionality is provided via
a simple functional interface, `deepcell_types.predict`.

For a complete example of the cell-type inference pipeline, check out
the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html).

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
