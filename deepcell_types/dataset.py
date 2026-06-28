import torch
from torch.utils.data import IterableDataset, get_worker_info

import numpy as np
import warnings
from scipy.ndimage import distance_transform_edt

from .preprocessing import patch_generator


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
        preprocess=None,
        **kwargs,
    ):
        super(PatchDataset, self).__init__(**kwargs)

        self.preprocess = preprocess

        if raw.ndim != 3:
            raise ValueError("raw must have shape (C, H, W).")
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D label image.")
        if raw.shape[0] != len(channel_names):
            raise ValueError(
                f"raw has {raw.shape[0]} channels, but {len(channel_names)} "
                "channel names were provided."
            )
        if not (np.isfinite(mpp) and mpp > 0):
            raise ValueError(
                f"mpp must be a positive, finite resolution in microns/pixel; "
                f"got {mpp!r}."
            )
        # Reject non-finite input up front. A NaN/inf in ``raw`` propagates
        # through the default per-channel normalization into the model, makes
        # ``F.softmax`` return a uniform/NaN distribution, and ``np.argmax``
        # then labels EVERY cell as class 0 — confident, uniform, wrong, and
        # un-abstained (the IQR fence collapses on a constant max-softmax).
        # Fail loudly here instead. (The custom-``preprocess`` hook output is
        # checked separately in preprocessing.py; this guards the raw input,
        # which the default path does not otherwise validate.)
        if not np.isfinite(raw).all():
            raise ValueError(
                "raw contains non-finite values (NaN/inf); predict() requires a "
                "finite (C, H, W) array."
            )

        self.n_cells = int(np.count_nonzero(np.unique(mask.astype(np.int64))))

        # Model requires image and mask in single precision
        raw = raw.astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.dct_config = dct_config
        self.paddings = -1.0
        self.mpp = mpp
        self.marker2idx = dct_config.marker2idx
        self.channel_mapping = dct_config.channel_mapping

        # A channel is masked out (dropped) when it (a) does not resolve to a
        # registry marker, (b) duplicates a marker an earlier channel already
        # provided, or (c) is identically zero across the whole FOV. Case (c)
        # matches the training dataloader, which attention-masks all-zero
        # channels (acquisition gaps cover ~3.4% of valid channels per MIBI/IMC
        # FOV) so the model never attends to a constant-zero token carrying a
        # real marker embedding. Dropping the channel here is equivalent to that
        # attention mask: model.forward() makes padding/masked channels inert
        # for the CLS embedding (key-padding-masked in attention and zeroed in
        # the mean-intensity scatter), so a dropped channel and a kept-but-masked
        # channel produce the same cell-type logits.
        channel_all_zero = (raw.reshape(raw.shape[0], -1) == 0).all(axis=1)
        channel_names_standard = []
        channel_masking = []
        seen_markers = set()
        for i, ch_name in enumerate(channel_names):
            ch_name_standard = self.dct_config.resolve_channel_name(ch_name)
            if ch_name_standard is None or ch_name_standard not in self.marker2idx:
                channel_masking.append(True)
                warnings.warn(
                    f"Channel {ch_name} is not in the channel mapping. "
                    "This channel will be masked out."
                )
            elif ch_name_standard in seen_markers:
                # Two input channels resolving to the same canonical marker would
                # share a marker2idx index; downstream the per-marker scatter is
                # last-write-wins, so the duplicate must be dropped, not stacked.
                channel_masking.append(True)
                warnings.warn(
                    f"Channel {ch_name} resolves to marker {ch_name_standard!r}, "
                    "already provided by an earlier channel; the duplicate will "
                    "be masked out."
                )
            elif channel_all_zero[i]:
                # All-zero on this FOV: mask it out to match training, where the
                # model is trained to never attend to a constant-zero channel.
                channel_masking.append(True)
                warnings.warn(
                    f"Channel {ch_name} (marker {ch_name_standard!r}) is all-zero "
                    "across this FOV; it will be masked out to match training, "
                    "where all-zero channels are attention-masked."
                )
            else:
                seen_markers.add(ch_name_standard)
                channel_masking.append(False)
                channel_names_standard.append(ch_name_standard)

        if len(channel_names_standard) > dct_config.MAX_NUM_CHANNELS:
            raise ValueError(
                f"{len(channel_names_standard)} mapped channels exceeds "
                f"MAX_NUM_CHANNELS={dct_config.MAX_NUM_CHANNELS}."
            )
        # Pad only to the channels actually present in THIS FOV, not to the
        # global MAX_NUM_CHANNELS. Padding tokens are inert in model.forward()
        # (see the channel-masking note above), so sizing the per-channel ResNet
        # and the (channel-quadratic) transformer to the real channel count is
        # numerically identical to padding to MAX_NUM_CHANNELS while avoiding the
        # wasted work over padding. Verified by
        # tests/test_channel_padding_equivalence.py.
        self.max_channels = len(channel_names_standard)

        ch_idx = torch.as_tensor(
            [self.marker2idx[ch_name] for ch_name in channel_names_standard]
            + [-1] * (self.max_channels - len(channel_names_standard))
        )  # (C_max, )
        self.channel_names_standard = channel_names_standard
        self.ch_idx = ch_idx
        channel_mask_arr = np.array(channel_masking)
        if channel_mask_arr.any():
            self.raw = raw[~channel_mask_arr, :, :]  # (C, H, W) drop masked
        else:
            # No channels dropped: alias the float32 array instead of taking a
            # full (multi-GB) copy. Boolean-row indexing always copies even when
            # the mask is all-False, which doubled peak RAM on wide FOVs.
            self.raw = raw
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
        if self.raw is None:
            raise RuntimeError(
                "PatchDataset is single-pass: its source array is released "
                "after the first iteration to free memory. Construct a new "
                "PatchDataset to iterate again."
            )
        # Release the dataset's reference to the full-resolution source array
        # before iterating, so it can be freed as soon as patch_generator
        # rescales it (the rescaled copy is roughly half the size at these MPPs).
        # This is the single biggest sustained-RAM reduction on multi-GB FOVs.
        # Safe because the dataset is single-pass: DataLoader iterates it once,
        # and multiprocessing workers each receive their own unpickled copy.
        raw = self.raw
        self.raw = None
        gen = patch_generator(
            raw,
            self.mask,
            self.mpp,
            dct_config=self.dct_config,
            preprocess=self.preprocess,
            channel_names=self.channel_names_standard,
        )
        del raw  # only the generator's frame now pins the full-res source
        for patch_idx, (raw_patch, mask_patch, cell_index, _) in enumerate(gen):
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
