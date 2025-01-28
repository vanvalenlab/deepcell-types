from pathlib import Path
import numpy as np
from tqdm import tqdm
import zarr
import click
from collections import defaultdict

from dct_kit.image_funcs import patch_generator
from dct_kit.config import DCTConfig

dct_config = DCTConfig()


@click.command()
@click.option("--data_name", type=str, default="")
def patchify(data_name, batch_size=2000):
    data_dir = Path("data")
    data_path = data_dir / data_name
    output_path = Path(data_path).with_suffix(".patched.zarr")

    input_group = zarr.open(data_path, mode="r")
    output_group = zarr.open(output_path, mode="w")

    file_names = input_group.attrs['file_names']
    print(file_names)

    channel_names = input_group.attrs["channel_names"]

    output_group.attrs["channel_names"] = channel_names

    # Create the output group
    num_chs = len(channel_names)
    patch_shape = (
        num_chs,
        dct_config.CROP_SIZE,
        dct_config.CROP_SIZE,
    )

    unique_cell_types = set()
    for file_name in file_names:
        if "cell_type_info" in input_group[file_name]:
            cell_type = input_group[file_name]["cell_type_info"]["cell_type"]
            unique_cell_types.update(cell_type)
        else:
            unique_cell_types.add("Unknown")

    for orig_ct in list(unique_cell_types):
        ct_group = output_group.create_group(orig_ct)
        ct_group.create_dataset(
            "raw",
            shape=(0, *patch_shape),
            chunks=(64, *patch_shape),
            dtype="float32",
        )
        ct_group.create_dataset(
            "mask",
            shape=(0, *(patch_shape[1:]), 2),
            chunks=(64, *(patch_shape[1:]), 2),
            dtype="float32",
        )
        ct_group.create_dataset(
            "cell_index", shape=(0,), chunks=(64,), dtype="int32"
        )
        ct_group.create_dataset(
            "file_name", shape=(0,), chunks=(64,), dtype="U100"
        )

    q_values = []        
    for file_name in file_names:
        raw = input_group[file_name]["raw"][:].astype(
            np.float32
        )  # convert to float32 to save space
        raw[raw==0] = np.nan
        q = np.nanquantile(raw, 0.99, axis=(1,2))
        q_values.append(q)
    
    q_values = np.array(q_values)
    final_q = np.nanmean(q_values, axis=0)


    # Loop through each file
    for file_name in tqdm(file_names):
        raw = input_group[file_name]["raw"][:].astype(
            np.float32
        )  # convert to float32 to save space
        mask = input_group[file_name]["mask"][:]
        mpp = input_group[file_name].attrs["mpp"]
        if "cell_type_info" in input_group[file_name]:
            cell_type = input_group[file_name]["cell_type_info"]["cell_type"][:]
            cell_index = input_group[file_name]["cell_type_info"]["cell_index"][:]
        else:
            cell_type = None
            cell_index = None
        

        batches = defaultdict(list) # store patches for each cell type
        for raw_patch, mask_patch, idx, orig_ct in patch_generator(
            raw, mask, mpp, dct_config=dct_config, final_q=final_q, cell_index=cell_index, cell_type=cell_type, 
        ):
            batches[orig_ct].append((raw_patch, mask_patch, idx, orig_ct))
            if len(batches[orig_ct]) == batch_size:
                ct_group = output_group[orig_ct]
                ct_group["raw"].append(np.stack([x[0] for x in batches[orig_ct]]))
                ct_group["mask"].append(np.stack([x[1] for x in batches[orig_ct]]))
                ct_group["cell_index"].append(np.array([x[2] for x in batches[orig_ct]]))
                ct_group["file_name"].append(np.array([file_name] * len(batches[orig_ct]), dtype=f"U80"))
                batches[orig_ct] = []

        # Append remaining patches in each batch
        for orig_ct, items in batches.items():
            if items:
                ct_group = output_group[orig_ct]
                ct_group["raw"].append(np.stack([x[0] for x in batches[orig_ct]]))
                ct_group["mask"].append(np.stack([x[1] for x in batches[orig_ct]]))
                ct_group["cell_index"].append(np.array([x[2] for x in batches[orig_ct]]))
                ct_group["file_name"].append(np.array([file_name] * len(batches[orig_ct]), dtype=f"U80"))
    
    return output_path


if __name__ == "__main__":
    patchify()