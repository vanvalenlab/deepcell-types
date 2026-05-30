"""
CellSighter model and batch conversion utilities.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-40066-7
- Code: https://github.com/KerenLab/CellSighter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from deepcell_types.training.utils import BatchData


class CellSighterModel(nn.Module):
    """
    CellSighter model based on ResNet-50.

    Modifies the first convolutional layer to accept (num_channels + 2) input channels
    (raw intensity channels + cell mask + neighbor mask), and the final FC layer for
    num_classes output.

    Reference: https://github.com/KerenLab/CellSighter
    """

    def __init__(self, input_channels: int, num_classes: int, pretrained: bool = False,
                 model_size: str = "resnet50"):
        """
        Initialize CellSighter model.

        Args:
            input_channels: Number of input channels (raw channels + 2 for masks)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights (default False
                        to match original CellSighter implementation)
            model_size: ResNet variant ('resnet18' or 'resnet50', default 'resnet50')
        """
        super().__init__()

        # Load ResNet backbone
        if model_size == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = torchvision.models.resnet18(weights=weights, num_classes=num_classes)
        else:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = torchvision.models.resnet50(weights=weights, num_classes=num_classes)

        # CIFAR-style stem: 3×3 stride-1 conv + no maxpool.
        # The standard ImageNet stem (7×7 stride-2 + maxpool) collapses 32×32
        # inputs to 1×1 by layer4, producing degenerate features.
        # With this stem: 32→32→32→16→8→4→1 (adaptive avg pool).
        self.model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, input_channels, H, W)

        Returns:
            Logits of shape (B, num_classes) during training,
            Softmax probabilities during evaluation
        """
        out = self.model(x)
        if not self.training:
            out = F.softmax(out, dim=-1)
        return out


def convert_batch_for_cellsighter(
    batch_data: BatchData,
    num_markers: int,
) -> torch.Tensor:
    """
    Convert batch from dataset format to CellSighter format with global channel alignment.

    Scatters the sequential per-dataset channels to their global marker2idx positions
    so that the same marker always occupies the same input channel across datasets.

    Args:
        batch_data: BatchData instance
        num_markers: Total number of unique markers (269)

    Returns:
        cellsighter_input: (B, num_markers+2, H, W) - [globally aligned channels, cell mask, neighbor mask]
    """
    # Extract raw intensity patches: (B, 75, H, W) - sequential local order
    raw_patches = batch_data.sample[:, :, 0, :, :]
    B, C_max, H, W = raw_patches.shape

    # Zero out padded channels first
    masks_expanded = batch_data.mask.unsqueeze(-1).unsqueeze(-1)  # (B, C_max, 1, 1)
    raw_patches = raw_patches.masked_fill(masks_expanded, 0.0)

    # Scatter to globally aligned positions using ch_idx
    # ch_idx: (B, 75), values are global marker indices (0-268) or -1 for padding
    ch_idx = batch_data.ch_idx  # (B, C_max)
    valid = ch_idx >= 0  # (B, C_max)

    global_patches = raw_patches.new_zeros(B, num_markers, H, W)
    ch_idx_clamped = ch_idx.clamp(min=0)  # clamp -1 to 0 for scatter (masked out anyway)
    ch_idx_expanded = ch_idx_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    valid_expanded = valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    global_patches.scatter_(1, ch_idx_expanded, raw_patches * valid_expanded.float())

    # Extract masks from spatial context
    cell_masks = batch_data.spatial_context[:, 0:1, :, :]  # (B, 1, H, W)
    neighbor_masks = batch_data.spatial_context[:, 1:2, :, :]  # (B, 1, H, W)

    # Concatenate: (B, num_markers+2, H, W)
    return torch.cat([global_patches, cell_masks, neighbor_masks], dim=1)
