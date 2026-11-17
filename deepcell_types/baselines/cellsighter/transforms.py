"""Faithful CellSighter train-time augmentation.

The original CellSighter (Amitay et al., Nat Commun 2023; github.com/KerenLab/
CellSighter, ``data/transform.py``) applies, per crop: Poisson intensity
resampling, random cell/environment mask dilation, random 0-360° rotation,
independent per-channel sub-pixel shifts (<=5px), and H/V flips (p=0.75).

These transforms operate on the SAME ``combined`` representation the DCT
augmentation pipeline uses (see ``training.transforms.AugmentedDataset``): a
single ``(C_max + n_context, H, W)`` tensor whose last ``n_context`` channels
are the spatial-context maps ``[self_mask, neighbor_mask, distance_transform]``
and whose leading ``C_max`` channels are the (faithful CellSighter: unmasked)
intensity channels, including padding channels that are discarded downstream.

Deviation from the original — Poisson resampling is intentionally OMITTED.
Poisson shot-noise augmentation models photon counts, but our archive's
``preprocessed/raw`` is per-FOV per-channel min-max normalized to [0, 1] (not
raw counts), so ``np.random.poisson`` on those values would collapse the signal
to near-binary noise. The geometric augmentations below are scale-agnostic and
carry the bulk of CellSighter's generalization benefit.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class _RandomRotation:
    """Rotate ALL channels by one shared random angle in [0, 360)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        angle = float(torch.empty(()).uniform_(0.0, 360.0))
        # bilinear keeps intensities smooth; masks become slightly soft, which
        # is acceptable (the original feeds masks as float channels too).
        return TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)


class _PerChannelShift:
    """Independently shift each intensity channel by up to ``max_shift`` px.

    Matches the original ``ShiftAugmentation`` (per-marker registration jitter):
    each intensity channel is shifted with probability ``p`` by a random integer
    offset in ``[-max_shift, max_shift]`` along H and W, with zero fill. The
    trailing ``n_context`` mask channels (segmentation) are left untouched.
    """

    def __init__(self, max_shift: int = 5, p: float = 0.5, n_context: int = 3):
        self.max_shift = max_shift
        self.p = p
        self.n_context = n_context

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[0] - self.n_context
        if C <= 0 or torch.rand(()) >= self.p:
            return x
        H, W = x.shape[-2], x.shape[-1]
        # Per-channel integer offsets; ~30% of channels actually shift.
        do = torch.rand(C) < 0.3
        dy = torch.randint(-self.max_shift, self.max_shift + 1, (C,)).float() * do
        dx = torch.randint(-self.max_shift, self.max_shift + 1, (C,)).float() * do
        if not bool(do.any()):
            return x
        # Vectorized per-channel shift via grid_sample (channels as batch).
        # nearest + align_corners gives an exact integer-pixel shift; zeros
        # padding makes it a true shift (not a wrap), matching the loop version.
        inten = x[:C].unsqueeze(1)  # (C, 1, H, W)
        ys = torch.linspace(-1.0, 1.0, H)
        xs = torch.linspace(-1.0, 1.0, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(C, 1, 1, 1)  # (C, H, W, 2)
        # shift content by (dy, dx) px -> sample at (y - dy, x - dx)
        grid[..., 0] -= (2.0 * dx / max(W - 1, 1)).view(C, 1, 1)
        grid[..., 1] -= (2.0 * dy / max(H - 1, 1)).view(C, 1, 1)
        shifted = F.grid_sample(
            inten, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )  # (C, 1, H, W)
        out = x.clone()
        out[:C] = shifted[:, 0]
        return out


class _MaskDilation:
    """Randomly dilate the self and neighbor mask channels (50% each).

    The self mask is channel ``-n_context`` and the neighbor mask is
    ``-n_context + 1`` in the combined tensor. Dilation uses a square structuring
    element of random odd-ish size in {2, 3, 5} via max-pooling, matching the
    original's ``cell_shape_aug`` / ``env_shape_aug``.
    """

    def __init__(self, p: float = 0.5, n_context: int = 3):
        self.p = p
        self.n_context = n_context

    @staticmethod
    def _dilate(plane: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        d = F.max_pool2d(
            plane[None, None].float(), kernel_size=k, stride=1, padding=pad
        )[0, 0]
        # max_pool2d with even k can grow the spatial size by 1; crop back.
        return d[: plane.shape[0], : plane.shape[1]]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self_idx = x.shape[0] - self.n_context
        neigh_idx = self_idx + 1
        for idx in (self_idx, neigh_idx):
            if torch.rand(()) < self.p:
                k = int([2, 3, 5][int(torch.randint(0, 3, ()))])
                x[idx] = self._dilate(x[idx], k)
        return x


def build_cellsighter_train_transform(
    max_shift: int = 5, n_context: int = 3, flip_p: float = 0.75
):
    """Compose the faithful CellSighter geometric augmentation.

    Returns a callable ``(C_max + n_context, H, W) -> (C_max + n_context, H, W)``
    usable as the ``train_transform`` of ``create_dataloader``. Order mirrors the
    original: mask dilation -> rotation -> per-channel shift -> flips.
    """
    # Local import to reuse the framework's flip primitives without a cycle.
    from deepcell_types.training.transforms import (
        _Compose,
        _RandomHorizontalFlip,
        _RandomVerticalFlip,
    )

    return _Compose(
        [
            _MaskDilation(p=0.5, n_context=n_context),
            _RandomRotation(p=1.0),
            _PerChannelShift(max_shift=max_shift, p=0.5, n_context=n_context),
            _RandomHorizontalFlip(p=flip_p),
            _RandomVerticalFlip(p=flip_p),
        ]
    )
