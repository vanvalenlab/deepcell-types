# DeepCell Types

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics that addresses the challenge of generalization across diverse datasets with varying marker panels. 


## Download the model
```python
from deepcell_types.utils import download_model
download_model()
```

## How to use

Clone the repo, create a virual environment, install the package by `pip install -e .`.
To run the demo: `python demo/inference.py`


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
