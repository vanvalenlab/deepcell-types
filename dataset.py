import torch
import zarr
import numpy as np
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(
        self,
        path,
        dct_config,
        celltype_mapping=None,
        channel_mapping=None,
        **kwargs,
    ):
        super(PatchDataset, self).__init__(**kwargs)

        self.max_channels = dct_config.MAX_NUM_CHANNELS

        self.paddings = -1.0
        self.marker2idx = dct_config.marker2idx
        self.ct2idx = dct_config.ct2idx
        if celltype_mapping is None:
            celltype_mapping = {ct_label: ct_label for ct_label in dct_config.ct2idx.keys()}
        celltype_mapping["Unknown"] = "Unknown"
        user_channel_mapping = channel_mapping
        channel_mapping = {ch: ch for ch in dct_config.master_channels}
        if user_channel_mapping:
            channel_mapping.update(user_channel_mapping)
        self.celltype_mapping = celltype_mapping
        self.channel_mapping = channel_mapping
        self.indices = [] # global indices
        zf = zarr.open(path, mode="r")
        self.zarr_file = zf
        
        for ct_label, ct_data in zf.groups():
            if ct_label not in celltype_mapping:
                continue
            ct_label_standard = celltype_mapping[ct_label]
            new_indices = [
                    (
                        ct_label,
                        ct_label_standard,
                        idx,
                        fov_name,
                        path.stem, # dataset name
                    )
                    for idx, fov_name in enumerate(ct_data["file_name"])
                ]
            self.indices.extend(new_indices)

        

    def _pad_images(self, sample):
        return np.pad(
            sample,
            ((0, self.max_channels - sample.shape[0]), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=self.paddings,
        )
    
    def _pad_marker_positivity(self, marker_positivity):
        return np.pad(
            marker_positivity,
            (0, self.max_channels - len(marker_positivity)),
            mode="constant",
            constant_values=0,
        )

    def _create_attn_mask(self, sample):
        # True = padding
        # https://pytorch.org/docs/stable/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention.forward
        mask = np.full((self.max_channels), True)
        mask[0 : sample.shape[0]] = False
        
        return mask

    def _combine_masks(self, raw, mask):
        mask = np.swapaxes(mask, 0, 2)  # (2, H, W)
        mask = np.expand_dims(mask, axis=0)  # (1, 2, H, W)
        raw_aug_mask = np.concatenate(
            [
                np.expand_dims(raw, axis=1),  # (C, 1, H, W)
                np.tile(mask, (raw.shape[0], 1, 1, 1)),  # (C, 2, H, W)
            ],
            axis=1,
        )  # (C, 3, H, W)
        return raw_aug_mask


    def _calcualte_marker_positivity(self, raw, mask, threshold=0.05):
        """Threshold on mean intensity to get marker positivity
        Input: 
            raw: (C, H, W)
            mask: (H, W)
        Output:
            marker_positivity: (C, )
        """
        area = np.sum(mask)
        if area == 0: # this should not happen! 
            mean_intensity = np.zeros(len(raw), dtype=np.float32)
            return mean_intensity
        
        sum_intensity = np.sum(raw * np.expand_dims(mask, axis=0), axis=(-1,-2))
        mean_intensity = np.divide(sum_intensity, area)
        
        marker_positivity = (mean_intensity > threshold).astype(np.float32)

        return marker_positivity


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ct_label, ct_label_standard, sample_index, fov_name, dataset_name = self.indices[
            idx
        ]
        raw = self.zarr_file[ct_label]["raw"][sample_index]  # (C, H, W)
        combined_mask = self.zarr_file[ct_label]["mask"][
            sample_index
        ]  # (H, W, 2), self and neighbor masks
        cell_index = self.zarr_file[ct_label]["cell_index"][sample_index]
        ch_names = self.zarr_file.attrs["channel_names"]
        ch_names_standard = [self.channel_mapping[ch_name] for ch_name in ch_names]
        ch_idx = torch.as_tensor(
            [self.marker2idx[ch_name] for ch_name in ch_names_standard]
            + [-1] * (self.max_channels - len(ch_names_standard))
        )  # (C_max, )
        if ct_label_standard == 'Unknown':
            ct_idx = -1
        else:
            ct_idx = self.ct2idx[ct_label_standard]

        sample = self._combine_masks(raw, combined_mask)  # (C, 3, H, W)
        mask = self._create_attn_mask(sample)  # (C_max,)
        sample = self._pad_images(sample)  # (C_max, 3, H, W)
        
        sample, ch_idx, mask = torch.as_tensor(sample), torch.as_tensor(ch_idx), torch.as_tensor(mask)

        return sample, ch_idx, mask, ct_idx, cell_index, fov_name
    
