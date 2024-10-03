import numpy as np
import yaml
from pathlib import Path


def flatten_nested_dict(nested_dict):
    flattened = []
    for key, value in nested_dict.items():
        if value:
            flattened.append(key)
            flattened.extend(flatten_nested_dict(value))
        else:
            flattened.append(key)
    return list(sorted(set(flattened)))


def get_ct_ch_across_files(dataset_path, keyword=None):
    core_dict = {}
    core_channels = []
    for file in dataset_path.iterdir():
        if "npz.dvc" in file.name:
            meta_file = file

            if keyword is not None:
                if not file.name.startswith(keyword):
                    continue

            with open(meta_file) as f:
                meta_info = yaml.load(f, Loader=yaml.FullLoader)

            try:
                celltype_mapper = meta_info["meta"]["file_contents"]["cell_types"][
                    "mapper"
                ]
            except KeyError:
                print(f"cell type mapper not found in {file.name}")
                continue

            channels = [
                item["target"] for item in meta_info["meta"]["sample"]["channels"]
            ]

            if core_dict == {}:
                core_dict = celltype_mapper
            else:
                assert (
                    celltype_mapper == core_dict
                ), f"celltype mapper is not the same across all files, {file.name}"

            if core_channels == []:
                core_channels = channels
            else:
                assert (
                    core_channels == channels
                ), f"channels are not the same across all files, {file.name}"

    return core_dict, core_channels


def choose_channels(channel_names, channel_mapping):
    channel_mask = []
    channel_names_updated = []
    for ch in channel_names:
        if ch in channel_mapping["channels_kept"]:
            channel_mask.append(True)
            channel_names_updated.append(channel_mapping["channels_kept"][ch])
        elif ch in channel_mapping["channels_dropped"]:
            channel_mask.append(False)
        else:
            raise ValueError(f"Channel name {ch} not found in channel_mapping.yaml")
        
    channel_slices = np.where(channel_mask)[0]

    return channel_slices, channel_names_updated



def create_marker_positivity_mask(unique_cell_types, dataset_name, channel_list, padding_length, dct_config):
    marker_positivity_mask_dict = {}
    for orig_ct in unique_cell_types:
        ct = dct_config.celltype_mapping[dataset_name][orig_ct]
        positive_channels = dct_config.positivity_mapping.get(ct, [0])
        positive_channels_dataset_specific = []
        if dataset_name in dct_config.positivity_mapping_dataset_specific:
            tissue_marker_pos_dict = dct_config.positivity_mapping_dataset_specific[
                dataset_name
            ]
            if orig_ct in tissue_marker_pos_dict:
                positive_channels_dataset_specific = tissue_marker_pos_dict[orig_ct]

        marker_positivity = [
            True
            if ch in positive_channels or ch in positive_channels_dataset_specific
            else False
            for ch in channel_list
        ] + [False] * padding_length
        marker_positivity = np.array(marker_positivity, dtype=np.int32)
        marker_positivity_mask_dict[ct] = marker_positivity
    
    return marker_positivity_mask_dict



if __name__ == "__main__":
    pass