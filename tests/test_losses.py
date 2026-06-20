"""Tests for deepcell_types.training.losses.

R7 M4 — FocalLoss coverage:
    (a) reduction='none' shape
    (b) reduction='sum' matches sum of 'none'
    (c) ignore_index correctly excludes samples
    (d) gamma=0, alpha=None reduces to F.cross_entropy

R7 M5 — HierarchicalLoss coverage:
    (a) parent-superclass-wrong-leaf case: hierarchical loss is lower than flat CE
    (b) smoke test on project CELL_TYPE_HIERARCHY → finite scalar
"""

import pytest
import torch
import torch.nn.functional as F

from deepcell_types.training.losses import FocalLoss, HierarchicalLoss


# =============================================================================
# R7 M4 — FocalLoss coverage
# =============================================================================


class TestFocalLossReduction:
    def test_reduction_none_shape(self):
        """reduction='none' returns per-sample tensor of shape (N,)."""
        torch.manual_seed(0)
        N, C = 16, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        loss = loss_fn(logits, targets)

        assert loss.shape == (N,), f"expected ({N},), got {tuple(loss.shape)}"
        assert loss.ndim == 1

    def test_reduction_sum_equals_sum_of_none(self):
        """reduction='sum' returns a scalar equal to the sum of reduction='none'."""
        torch.manual_seed(0)
        N, C = 16, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        none_fn = FocalLoss(gamma=2.0, reduction="none")
        sum_fn = FocalLoss(gamma=2.0, reduction="sum")

        loss_none = none_fn(logits, targets)
        loss_sum = sum_fn(logits, targets)

        assert loss_sum.dim() == 0, "sum reduction should yield a scalar"
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6), (
            f"sum={loss_sum.item():.6f} != sum(none)={loss_none.sum().item():.6f}"
        )

    def test_ignore_index_excludes_samples(self):
        """ignore_index excludes samples at that label; remaining loss must match
        loss computed on a filtered input directly."""
        torch.manual_seed(0)
        N, C = 10, 4
        ignore = -100
        logits = torch.randn(N, C)
        # Use normal labels everywhere, then overwrite a couple to the ignore label.
        targets = torch.randint(0, C, (N,))
        targets[2] = ignore
        targets[7] = ignore

        fn_ignore = FocalLoss(gamma=2.0, reduction="sum", ignore_index=ignore)
        loss_with_ignore = fn_ignore(logits, targets)

        # Reference: drop ignored rows manually and compute the same loss.
        keep_mask = targets != ignore
        fn_ref = FocalLoss(gamma=2.0, reduction="sum", ignore_index=ignore)
        loss_ref = fn_ref(logits[keep_mask], targets[keep_mask])

        assert torch.allclose(loss_with_ignore, loss_ref, atol=1e-6), (
            f"ignore_index path diverges from filtered input: "
            f"{loss_with_ignore.item():.6f} vs {loss_ref.item():.6f}"
        )

    def test_ignore_index_all_ignored_returns_zero(self):
        """If every label is ignored, loss must collapse to 0."""
        N, C = 4, 3
        ignore = -100
        logits = torch.randn(N, C)
        targets = torch.full((N,), ignore, dtype=torch.long)

        fn = FocalLoss(gamma=2.0, reduction="mean", ignore_index=ignore)
        loss = fn(logits, targets)
        assert loss.item() == 0.0

    def test_gamma_zero_matches_cross_entropy(self):
        """FocalLoss(gamma=0, alpha=None) === F.cross_entropy."""
        torch.manual_seed(0)
        N, C = 32, 7
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        # reduction='mean'
        fn_focal_mean = FocalLoss(alpha=None, gamma=0.0, reduction="mean")
        loss_focal_mean = fn_focal_mean(logits, targets)
        loss_ce_mean = F.cross_entropy(logits, targets, reduction="mean")
        assert torch.allclose(loss_focal_mean, loss_ce_mean, atol=1e-5), (
            f"focal(mean)={loss_focal_mean.item():.6f} != ce(mean)={loss_ce_mean.item():.6f}"
        )

        # reduction='sum'
        fn_focal_sum = FocalLoss(alpha=None, gamma=0.0, reduction="sum")
        loss_focal_sum = fn_focal_sum(logits, targets)
        loss_ce_sum = F.cross_entropy(logits, targets, reduction="sum")
        assert torch.allclose(loss_focal_sum, loss_ce_sum, atol=1e-5)

        # reduction='none'
        fn_focal_none = FocalLoss(alpha=None, gamma=0.0, reduction="none")
        loss_focal_none = fn_focal_none(logits, targets)
        loss_ce_none = F.cross_entropy(logits, targets, reduction="none")
        assert torch.allclose(loss_focal_none, loss_ce_none, atol=1e-5)


# =============================================================================
# R7 M5 — HierarchicalLoss
# =============================================================================


@pytest.fixture
def tiny_hierarchy_yaml(tmp_path):
    """Build a tiny fine→coarse hierarchy YAML.

    6 fine classes bucketed into 3 coarse groups:
        CD4T, CD8T, Treg → Tcell
        Bcell, Plasma    → Bcell
        Macrophage       → Myeloid
    """
    import yaml
    mapping = {
        "CD4T": "Tcell",
        "CD8T": "Tcell",
        "Treg": "Tcell",
        "Bcell": "Bcell",
        "Plasma": "Bcell",
        "Macrophage": "Myeloid",
    }
    path = tmp_path / "tiny_hierarchy.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(mapping, f)
    return path


class TestHierarchicalLoss:
    def test_parent_superclass_wrong_leaf_is_lower_than_flat_ce(self, tiny_hierarchy_yaml):
        """If the prediction lands on the wrong leaf but right parent group,
        hierarchical loss < flat cross-entropy.

        Setup:
            ct2idx: CD4T=0, CD8T=1, Treg=2, Bcell=3, Plasma=4, Macrophage=5
            True class: CD4T (0) — a Tcell
            Predicted (one-hot-ish): CD8T (1) — also a Tcell (sibling)

        The coarse distribution puts all Tcell probability on "Tcell" (correct
        parent), so hierarchical NLL on the coarse label is small. Flat CE on
        the fine label is large (predicted wrong fine class).
        """
        torch.manual_seed(0)
        ct2idx = {
            "CD4T": 0,
            "CD8T": 1,
            "Treg": 2,
            "Bcell": 3,
            "Plasma": 4,
            "Macrophage": 5,
        }
        loss_fn = HierarchicalLoss(
            config_path=str(tiny_hierarchy_yaml),
            ct2idx=ct2idx,
            weight=1.0,  # use weight=1 to compare against unweighted CE
        )

        # Predicted CD8T with high confidence; true is CD4T (sibling in Tcell).
        logits = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([0])  # CD4T

        hierarchical_loss = loss_fn(logits, targets)
        flat_ce = F.cross_entropy(logits, targets, reduction="mean")

        assert hierarchical_loss.item() < flat_ce.item(), (
            f"Expected hierarchical_loss ({hierarchical_loss.item():.4f}) < "
            f"flat_ce ({flat_ce.item():.4f}) when prediction is wrong leaf but right parent"
        )

    def test_smoke_project_hierarchy(self):
        """Forward pass with the project's CELL_TYPE_HIERARCHY and
        combined_celltypes.yaml returns a finite scalar on random inputs."""
        # training.config pulls pandas (a [train] extra); only this test needs
        # it, so guard here rather than skipping the whole module.
        pytest.importorskip("pandas")
        from deepcell_types.training.config import CELL_TYPE_HIERARCHY, CONFIG_DIR  # noqa: F401

        yaml_path = CONFIG_DIR / "combined_celltypes.yaml"
        if not yaml_path.exists():
            pytest.skip(f"combined_celltypes.yaml not found at {yaml_path}")

        # Construct a minimal ct2idx that matches the YAML keys.
        import yaml
        with open(yaml_path) as f:
            mapping = yaml.safe_load(f)
        fine_names = sorted(mapping.keys())
        ct2idx = {name: i for i, name in enumerate(fine_names)}
        n_fine = len(ct2idx)

        loss_fn = HierarchicalLoss(
            config_path=str(yaml_path),
            ct2idx=ct2idx,
            weight=0.5,
        )

        torch.manual_seed(0)
        logits = torch.randn(8, n_fine)
        targets = torch.randint(0, n_fine, (8,))

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0, "loss must be scalar"
        assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
        # Weight=0.5 so loss should be strictly positive for random inputs.
        assert loss.item() > 0.0
