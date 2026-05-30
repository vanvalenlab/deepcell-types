"""
MAPS MLP model for cell type classification.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-44188-w
- Code: https://github.com/mahmoodlab/MAPS
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPSModel(nn.Module):
    """
    MLP classifier from MAPS paper.

    4 FC layers with ReLU + Dropout, followed by final classifier.

    Reference: https://github.com/mahmoodlab/MAPS
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ):
        """
        Initialize MAPS model.

        Args:
            input_dim: Number of input features (marker channels)
            num_classes: Number of output classes (cell types)
            hidden_dim: Hidden layer dimension (default 512 per paper)
            dropout: Dropout rate (default 0.25, matching original experiment scripts)
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, input_dim)

        Returns:
            logits: Logits of shape (B, num_classes)
            probs: Softmax probabilities of shape (B, num_classes)

        Note: Matches original MAPS implementation which returns both logits and probs.
        """
        features = self.fc(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
