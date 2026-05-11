import torch
from torch.utils.data import IterableDataset, get_worker_info

import numpy as np
import warnings
from scipy.ndimage import distance_transform_edt

from .dct_kit.image_funcs import patch_generator


class PatchDataset(IterableDataset):
    """
    Dataset for single-image patchified data.
    """

    def __init__(
        self,
        raw,
        mask,
        channel_names,
        mpp,
        dct_config,
        **kwargs,
    ):
        super(PatchDataset, self).__init__(**kwargs)

        if raw.ndim != 3:
            raise ValueError("raw must have shape (C, H, W).")
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D label image.")
        if raw.shape[0] != len(channel_names):
            raise ValueError(
                f"raw has {raw.shape[0]} channels, but {len(channel_names)} "
                "channel names were provided."
            )

        self.n_cells = int(np.count_nonzero(np.unique(mask.astype(np.int64))))

        # Model requires image and mask in single precision
        raw = raw.astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.dct_config = dct_config
        self.max_channels = dct_config.MAX_NUM_CHANNELS
        self.paddings = -1.0
        self.mpp = mpp
        self.marker2idx = dct_config.marker2idx
        self.channel_mapping = dct_config.channel_mapping

        channel_names_standard = []
        channel_masking = []
        for ch_name in channel_names:
            ch_name_standard = self.dct_config.resolve_channel_name(ch_name)
            if ch_name_standard is None or ch_name_standard not in self.marker2idx:
                channel_masking.append(True)
                warnings.warn(
                    f"Channel {ch_name} is not in the channel mapping. "
                    "This channel will be masked out."
                )
            else:
                channel_masking.append(False)
                channel_names_standard.append(ch_name_standard)

        if len(channel_names_standard) > self.max_channels:
            raise ValueError(
                f"{len(channel_names_standard)} mapped channels exceeds "
                f"MAX_NUM_CHANNELS={self.max_channels}."
            )

        ch_idx = torch.as_tensor(
            [self.marker2idx[ch_name] for ch_name in channel_names_standard]
            + [-1] * (self.max_channels - len(channel_names_standard))
        )  # (C_max, )
        self.channel_names_standard = channel_names_standard
        self.ch_idx = ch_idx
        self.raw = raw[~np.array(channel_masking), :, :]  # (C, H, W)
        if self.raw.shape[0] == 0:
            raise ValueError(
                "No input channels matched the DeepCell Types marker registry."
            )

    def _create_attn_mask(self, sample):
        # True = padding
        # https://pytorch.org/docs/stable/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention.forward
        mask = np.full((self.max_channels), True)
        mask[0 : sample.shape[0]] = False

        return mask

    @staticmethod
    def _distance_transform(self_mask):
        if self_mask.sum() == 0:
            return np.zeros_like(self_mask, dtype=np.float32)
        dist = distance_transform_edt(self_mask).astype(np.float32)
        max_dist = dist.max()
        if max_dist > 0:
            dist /= max_dist
        return dist

    def _create_canonical_sample(self, raw, mask):
        self_mask = mask[:, :, 0].astype(np.float32)
        neighbor_mask = mask[:, :, 1].astype(np.float32)
        spatial_context = np.stack(
            [self_mask, neighbor_mask, self._distance_transform(self_mask)],
            axis=0,
        ).astype(np.float32)

        raw_masked = raw * np.expand_dims(self_mask, axis=0)
        c, h, w = raw_masked.shape
        sample = np.full((self.max_channels, 1, h, w), self.paddings, dtype=np.float32)
        sample[:c, 0, :, :] = raw_masked
        attn_mask = self._create_attn_mask(raw)

        return sample, spatial_context, attn_mask

    def __iter__(self):
        """
        Patchify the raw and mask data into smaller patches
        """
        worker_info = get_worker_info()
        for patch_idx, (raw_patch, mask_patch, cell_index, _) in enumerate(
            patch_generator(self.raw, self.mask, self.mpp, dct_config=self.dct_config)
        ):
            if (
                worker_info is not None
                and patch_idx % worker_info.num_workers != worker_info.id
            ):
                continue

            sample, spatial_context, attn_mask = self._create_canonical_sample(
                raw_patch, mask_patch
            )
            yield (
                torch.as_tensor(sample),
                torch.as_tensor(spatial_context),
                torch.as_tensor(self.ch_idx),
                torch.as_tensor(attn_mask),
                cell_index,
            )

    def __len__(self):
        return self.n_cells
