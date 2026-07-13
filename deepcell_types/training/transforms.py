"""Tensor augmentation transforms and the augmenting dataset wrapper.

Extracted from ``deepcell_types.training.dataset`` for modularity. These
symbols are re-exported from ``dataset`` for backward compatibility, so
existing ``from deepcell_types.training.dataset import AugmentedDataset``
imports keep working unchanged.

Contains the minimal tensor-transform primitives (``_Compose``,
``_RandomHorizontalFlip``, ``_RandomVerticalFlip``) used to avoid a hard
torchvision dependency, the channel-dropout regularizer (``DropOutChannels``),
and the ``AugmentedDataset`` wrapper that applies spatial + dropout transforms
to a wrapped dataset/Subset.
"""

import torch
from torch.utils.data import Dataset


class _Compose:
    """Minimal tensor transform composition to avoid a hard torchvision import."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class _RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-1,))
        return x


class _RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-2,))
        return x


class AugmentedDataset(Dataset):
    """Wraps a dataset (or Subset) with augmentation transforms."""

    def __init__(self, dataset, transform=None, dropout_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.dropout_transform = dropout_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item) != 12:
            raise ValueError(
                f"AugmentedDataset expects a 12-tuple from the wrapped "
                f"dataset (sample, spatial_context, ch_idx, attn_mask, "
                f"ct_idx, domain_idx, mp, mp_mask, cell_index, "
                f"dataset_name, fov_name, tissue_idx), "
                f"got {len(item)}-tuple."
            )
        (
            sample,
            spatial_context,
            ch_idx,
            attn_mask,
            ct_idx,
            domain_idx,
            mp,
            mp_mask,
            cell_index,
            dataset_name,
            fov_name,
            tissue_idx,
        ) = item

        if self.transform:
            # Apply spatial transform consistently
            C_max = sample.shape[0]
            H, W = sample.shape[2], sample.shape[3]
            combined = torch.cat(
                [
                    sample.view(C_max, H, W),
                    spatial_context,
                ],
                dim=0,
            )
            combined = self.transform(combined)
            sample = combined[:C_max].unsqueeze(1)
            spatial_context = combined[C_max:]

        if self.dropout_transform:
            sample, ch_idx, attn_mask, mp, mp_mask = self.dropout_transform(
                sample, ch_idx, attn_mask, mp, mp_mask
            )

        return (
            sample,
            spatial_context,
            ch_idx,
            attn_mask,
            ct_idx,
            domain_idx,
            mp,
            mp_mask,
            cell_index,
            dataset_name,
            fov_name,
            tissue_idx,
        )


class DropOutChannels:
    """Drop random VALID channels (not padding) for regularization."""

    def __init__(self, n=3):
        self.n = n

    def __call__(self, sample, ch_idx, mask, marker_positivity, mp_mask):
        # Find valid (non-padded) channel indices
        valid_indices = torch.where(~mask)[0]
        n_valid = len(valid_indices)

        # Proportional dropout: drop at most 30% of valid channels, capped at
        # self.n. Very small panels are already information-poor; do not drop
        # from them.
        if n_valid <= 3:
            return sample, ch_idx, mask, marker_positivity, mp_mask

        n_drop = min(self.n, int(n_valid * 0.3))

        if n_drop <= 0 or n_valid <= n_drop:
            return sample, ch_idx, mask, marker_positivity, mp_mask

        # Sample from valid channels only
        drop_positions = valid_indices[torch.randperm(n_valid)[:n_drop]]

        sample[drop_positions] = -1.0
        ch_idx[drop_positions] = -1
        mask[drop_positions] = True
        marker_positivity[drop_positions] = 0
        mp_mask[drop_positions] = False

        return sample, ch_idx, mask, marker_positivity, mp_mask
