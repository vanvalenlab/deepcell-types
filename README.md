# DeepCell Types

DeepCell Types is a novel approach to cell phenotyping for spatial proteomics that addresses the challenge of generalization across diverse datasets with varying marker panels. 


## How to use

Put your dataset under `/data` folder, following the format below:
```
example_dataset.zarr
    attrs: 
        - channel_names
        - file_names
    - file_name1 (group)
        attrs: mpp
        - raw (dataset, shape (#channels, X, Y))
        - mask (dataset, shape (1, X, Y))
        - cell_type_info (dataset, optional)
            - cell_index
            - cell_type
    - file_name2
        ...
```

For `masks`, background are labeled as 0, cell indices starts from 1. For `raw`, the shapes should be (#channels, X, Y). If cell type annotations are available, you can add the optional `cell_type_info` of shape (#cells). It can be created as follows:
```
cell_type_info = np.zeros(
    num_cells,
    dtype=[("cell_index", "i4"), ("cell_type", "U60")],
)
cell_type_info["cell_index"] = #YOUR_CELL_INDEX_LIST
cell_type_info["cell_type"] = #YOUR_CELL_TYPE_LIST
```

You can also provide two optional mapping files: `celltype_mapping.yaml` and `channel_mapping.yaml` that maps your cell types and marker channels to the standard lists. The standard lists can be found here in `deepcelltypes-kit/deepcelltype_kit/config/core_celltypes.yaml` and `deepcelltypes-kit/deepcelltype_kit/config/master_channel.yaml`. If there are no cell type annotations, simply list `Unknown: Unknown` in the `celltype_mapping.yaml`. If your cell types and channels already match the standard lists, you can skip this by setting the two arguments to `None`.


We provided two formats of example data. `data/example_data_with_labels.zarr` comes with cell type labels, which are saved in `cell_type_info`; `/data/example_data_without_labels.zarr` has `cell_type_info` dataset and will be automatically labeled as 'Unknown's.


Next, build the docker image by running:
```
docker build . --tag=$USER/deepcell-types:latest
```

Once the docker image has been built, you can run the `preprocess.py` script to turn your images into patches:
```
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --entrypoint python \
    -v $PWD:/workspace \
    $USER/deepcell-types:latest \
    /workspace/preprocess.py --data_name example_dataset.zarr
```

Next, you can run predictions on the patches and collect results:
```
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --gpus '"device=0"' \
    --entrypoint python \
    --shm-size 80G \
    -v $PWD:/workspace \
    $USER/deepcell-types:latest \
    /workspace/predict.py --patch_data_name example_dataset.patched.zarr --model_name model_specific_ct
```

## Citation

