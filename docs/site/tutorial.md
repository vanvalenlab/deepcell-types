---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial

The `deepcell-types` model predicts cell types from multiplexed spatial
proteomic images.
There are three main inputs to the cell type prediction pipeline:
 1. The multiplexed image in channels-first format
 2. A whole-cell segmentation mask for the image, and
 3. A mapping of the channel index to the marker expression name.

Each of these components will be covered in further detail in this tutorial.

## Example datasets

This tutorial will make use of the spatial proteomic data provided by the
[HuBMAP data portal][hubmap-data-portal].
Users are encouraged to explore the portal for data of interest.
For convenience, a subset of the publicly-available spatial proteomic data
has been converted to a remote [zarr archive][zarr].
The datasets in the zarr archive reflect the original HuBMAP indexing scheme
(i.e. `HBM###_????_###`, where `#` indicates a number nad `?` indicates an
upper-case alphabetical character).

Interacting with the zarr hubmap data mirror requires a few additional
dependencies:

```bash
pip install zarr\>2 s3fs rich
```

```{note}
The hubmap data mirror uses zarr format v3, thus requires `zarr>2` to be
installed.
```

### Exploring the archive

```{code-cell}
import zarr

z = zarr.open_group(
    store="s3://hubmap-mirror-demo/hubmap.zarr",
    mode="r",
    storage_options={
        "anon": False,  # TODO: Make true when archive made public
        "client_kwargs": dict(region_name="us-east-2"),
    },
)
keys = list(z.group_keys())
print(f"Number of datasets in the archive: {len(keys)}")
```

High-level structure of the data archive:

```{code-cell}
z.tree()
```


[hubmap-data-portal]: https://portal.hubmapconsortium.org/search/datasets
[zarr]: https://zarr.readthedocs.io/en/stable/
