deepcell-types documentation
============================

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics
that addresses the challenge of generalization across diverse datasets with
varying marker panels.

For details on running the containerized workflow, see the [source repo][github].

For access to models and datasets, see {doc}`site/API-key`.

## Installation

The development version of `deepcell-types` can be installed with `pip`:

```bash
pip install git+https://github.com/vanvalenlab/deepcell-types@master
```

## Pre-trained models

A pre-trained model is required to run the cell-type inference pipeline.
Pre-trained models are available via the [users.deepcell.org][dc_org] portal
according to the [license terms][license].

The latest version can be downloaded like so:

```python
from deepcell_types.utils import download_model

download_model()  # No argument == latest released version
```

## Running

The {doc}`tutorial` demonstrates how to set up, run, and visualize the
outputs of the cell-type inference pipeline.

This tutorial can also be run locally with `jupyter`.

Start by cloning the sorce repository:

```bash
git clone https://github.com/vanvalenlab/deepcell-types.git && cd deepcell-types
```

Ensure that all necessary extra dependencies are installed in your virtual
environment (along with `deepcell-types` itself):

```bash
pip install -r docs/requirements.txt .
```

Launch a jupyter lab instance:

```bash
jupyter lab
```

Then open the tutorial at `docs/site/tutorial.md`.

```{note}
Depending on your `jupyterlab` version, you may need to right-click the tutorial.md
and select `Open with -> Jupytext notebook`.
```

```{toctree}
---
maxdepth: 1
hidden: true
---
site/tutorial
site/API-key
```

[github]: https://github.com/vanvalenlab/deepcell-types
[dc_org]: https://vanvalenlab.github.io/deepcell-types/site/API-key.html#
[license]: https://github.com/vanvalenlab/deepcell-types/blob/master/LICENSE
