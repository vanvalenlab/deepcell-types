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

## TissueNet archive (optional)

`predict()` does **not** require the TissueNet archive: the marker / cell-type
registry it needs ships with the package as a small `vocab.json` snapshot, so
the base install above + `download_model()` is enough to run inference.

The (multi-GB) `tissuenet-v*.zarr` archive is only needed if you want the
tissue→cell-type mapping or are reproducing the training pipeline. It is
distributed as a `.zip` that must be extracted before use:

```python
from deepcell_types.utils import download_training_data

# Downloads the .zip and extracts it; returns the extraction directory
# containing the tissuenet-v*.zarr archive. (Large download — see API-key page.)
archive_dir = download_training_data(extract=True)
```

When you do have an archive, point `predict` at it with the `zarr_path=`
argument, or set it once for the session:

```bash
export DEEPCELL_TYPES_ZARR_PATH=/path/to/tissuenet-v10.zarr
```

An explicit `zarr_path=` that doesn't contain an archive raises
`FileNotFoundError`; the checkpoint and registry must agree on the marker and
cell-type ordering, otherwise loading fails early with a `ValueError`.

## Running

The {doc}`site/tutorial` demonstrates how to set up, run, and visualize the
outputs of the cell-type inference pipeline.

This tutorial can also be run locally with `jupyter`.

Start by cloning the source repository:

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

   The model supports a maximum of **80** channels per dataset. If your data has
   more channels than this limit, it will be necessary to remove channels to get
   below it. Note that (by default) nuclear and non-recognized channel names are
   automatically dropped.

2. **Recognized channels**

   By default the model reads its registry of natively-supported channels and
   cell types from the `vocab.json` snapshot shipped with the package, so
   inference needs no archive download. The recognized channels are the markers
   listed in that snapshot. You can override it by pointing inference at a
   TissueNet zarr archive (pass `zarr_path=...` or set the
   `DEEPCELL_TYPES_ZARR_PATH` environment variable), in which case the
   recognized channels are whatever the selected archive exposes in its root
   `attrs.all_standardized_channels`. Either way, markers not found in the
   active registry are ignored at inference time.

   To check whether your marker panel overlaps the registry before
   downloading a checkpoint, call `list_supported_markers()` (and
   `list_supported_cell_types()` for the recognized cell-type labels):

   ```python
   import deepcell_types

   deepcell_types.list_supported_markers()
   ```

   There are two ways to add support for additional channels:

   - To add a new alias for a marker name that is currently supported, add the
     alias to `deepcell_types/channel_mapping.yaml`. For example,
     if your data contains a channel named `FP3` representing the `FoxP3` marker,
     add the following line to `channel_mapping.yaml`:

         FP3: FoxP3

     Note that the target name (`FoxP3` in this example) must be one of the
     names already found in the selected checkpoint's channel registry.

   - Adding new markers to the model can be achieved by manually acquiring
     embeddings for additional channels via OpenAI's `text-embedding-3-large`
     model. The reference helper that generates these embeddings from a
     channel's plain-English name is `scripts/generate_openai_embeddings.py`
     (model and prompt details can be found in the paper).

   Checkpoint loading validates that the archive and checkpoint agree on marker
   count and cell-type count. If they do not, inference fails early with a
   `ValueError`.

3. **Image preprocessing**

   The model requires users to preprocess input images to align with the
   distribution of our training data for optimal performance and generalization.
   See the paper for details.

## Training

The repository also ships the training pipeline used to produce the canonical
checkpoints. Training-only code lives under `deepcell_types.training` and is
gated behind the `[train]` install extra so plain inference users don't pull
in `zarr`, `pandas`, `scikit-learn`, etc.

```bash
pip install "deepcell-types[train] @ git+https://github.com/vanvalenlab/deepcell-types@master"
```

The end-to-end training and evaluation scripts live under `scripts/`:

- `scripts/train.py` — main training entry point (DCTConfig + FullImageDataset
  + FocalLoss / HierarchicalLoss + OneCycleLR; supports the FOV-grouped
  sampler and the conditioned marker-positivity head).
- `scripts/predict.py` — batched predictions over a zarr archive, with
  optional hierarchy-aware evaluation.
- `scripts/pretrain.py` — masked-marker pretraining stage.

All training scripts read mappings and metadata from a TissueNet zarr v3
archive (`tissuenet-v10.zarr` is the current canonical release; v8 and v9
are also accepted). Pass the archive path with
`--zarr_dir` (training scripts) or via the `DEEPCELL_TYPES_ZARR_PATH`
environment variable (`deepcell_types.predict`).

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
[dc_org]: https://users.deepcell.org/login/
[license]: https://github.com/vanvalenlab/deepcell-types/blob/master/LICENSE
