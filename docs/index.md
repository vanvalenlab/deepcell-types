deepcell-types documentation
============================

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics
that addresses the challenge of generalization across diverse datasets with
varying marker panels.

For details on running the cell-type inference workflow, see the {doc}`site/tutorial`.

The source code can be found at the [github repo][github].

For access to pre-trained models and datasets, see {doc}`site/API-key`.

## Installation

The development version of `deepcell-types` can be installed with `pip`:

```bash
pip install git+https://github.com/vanvalenlab/deepcell-types@master
```

## Pre-trained models

A pre-trained model is required to run the cell-type inference pipeline.
Pre-trained models are available via the [users.deepcell.org][dc_org] portal
according to the [license terms][license].
See {doc}`site/API-key` for details.

The latest version can be downloaded like so:

```python
from deepcell_types.utils import download_model

download_model()  # No argument == latest released version
```

## Running

The {doc}`site/tutorial` demonstrates how to set up, run, and visualize the
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

## Limitations

1. **Maximum channel limit**

   The model currently supports a maximum of **75** channels per dataset. If you
   have a dataset with more than 75 marker channels, it will be necessary to
   remove channels to get below this limit. Note that (by default) nuclear and
   non-recognized channel names are automatically dropped.

2. **Recognized channels**

   The model maintains a registry of natively-supported channels at
   [`deepcell_types/dct_kit/config/master_channels.yaml`][master_channels_gh].
   If your data contains markers not found in this listing, they will be
   ignored at inference time.
   There are two ways to add support for additional channels:

   - To add a new alias for a marker name that is currently supported, add the
     alias to `deepcell_types/dct_kit/config/channel_mapping.yaml`. For example,
     if your data contains a channel named `FP3` representing the `FoxP3` marker,
     add the following line to `channel_mapping.yaml`:

         FP3: FoxP3

     Note that the target name (`FoxP3` in this example) must be one of the
     names already found in `master_channels.yaml`

   - Adding new markers to the model can be achieved by manually acquiring
     embeddings for additional channels via DeepSeek (model and prompt details
     can be found in the paper).

3. **Image preprocessing**

   The model requires users to preprocess input images to align with the
   distribution of our training data for optimal performance and generalization.
   See the paper for details.

[master_channels_gh]: https://github.com/vanvalenlab/deepcell-types/blob/master/deepcell_types/dct_kit/config/master_channels.yaml

```{toctree}
---
maxdepth: 1
hidden: true
---
site/tutorial
site/API-key
site/reference
```

[github]: https://github.com/vanvalenlab/deepcell-types
[dc_org]: https://vanvalenlab.github.io/deepcell-types/site/API-key.html#
[license]: https://github.com/vanvalenlab/deepcell-types/blob/master/LICENSE
