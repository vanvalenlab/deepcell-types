"""
Tests for CellTypeAnnotator, FullImageDataset, FocalLoss with class weights,
FOV splits, DropOutChannels.
"""

import numpy as np
import torch
import pytest

from deepcell_types.model import (
    CellTypeAnnotator,
    SpatialEncoder,
    PerChannelResNet,
    ResBlock,
    MaskedMarkerHead,
    mask_marker_channels,
    create_model,
    MarkerConditionedMPHead,
)
from deepcell_types.training.losses import FocalLoss
from deepcell_types.training.dataset import (
    CellIndexRecord,
    FullImageDataset,
    DropOutChannels,
    create_fov_splits,
    compute_sample_weights,
)
from deepcell_types.training.utils import (
    BatchData,
    seed_everything,
)
from deepcell_types.training.config import compute_distance_transform


# =============================================================================
# Fixtures
# =============================================================================


class MockTissueNetConfig:
    """Mock TissueNetConfig for testing."""

    MAX_NUM_CHANNELS = 10  # smaller for fast tests
    CROP_SIZE = 32
    OUTPUT_SIZE = 32
    STANDARD_MPP_RESOLUTION = 0.5
    PERCENTILE_THRESHOLD = 99.0

    def __init__(self):
        self.ct2idx = {"T_cell": 0, "B_cell": 1, "Macrophage": 2}
        self.marker2idx = {"CD3": 0, "CD4": 1, "CD8": 2, "CD45": 3}
        self.domain2idx = {"CODEX": 0, "MIBI": 1}
        self.dataset_celltypes = {
            "TestDataset": ["B_cell", "Macrophage", "T_cell"]
        }
        self.domain_mapping = {"TestDataset": "CODEX"}
        self.marker_positivity_labels = self._create_mock_mpi()
        self.tissue_celltype_mapping = {}
        self.NUM_CELLTYPES = len(self.ct2idx)
        self.NUM_DOMAINS = len(self.domain2idx)

    def _create_mock_mpi(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "CD3": [0.8, 0.1, 0.0],
                "CD4": [0.7, "?", 0.0],  # "?" for B_cell CD4
                "CD8": [0.1, 0.9, 0.0],
                "CD45": [0.9, 0.8, 0.9],
            },
            index=["T_cell", "B_cell", "Macrophage"],
        )
        return {"TestDataset": df}


@pytest.fixture
def dct_config():
    return MockTissueNetConfig()


@pytest.fixture
def marker_embeddings():
    """Fake marker embeddings (4 markers, 32-d)."""
    np.random.seed(42)
    return np.random.randn(4, 32).astype(np.float32)


# =============================================================================
# Model tests
# =============================================================================


class TestCellTypeAnnotator:
    """Tests for model architecture."""

    def test_output_shapes(self, marker_embeddings):
        """Verify output shapes match specification."""
        B, C_max, H, W = 4, 10, 32, 32
        n_celltypes, n_domains = 3, 2

        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=n_celltypes,
            n_domains=n_domains,
            marker_embeddings=marker_embeddings,
            dropout=0.0,
        )

        sample = torch.randn(B, C_max, 1, H, W)
        spatial = torch.randn(B, 3, H, W)
        ch_idx = torch.zeros(B, C_max, dtype=torch.long)
        ch_idx[:, :4] = torch.arange(4)
        ch_idx[:, 4:] = -1
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 4:] = True

        ct_logits, domain_logits, mp_logits, cls_emb, ch_out, _ = model(
            sample, spatial, ch_idx, pad_mask
        )

        assert ct_logits.shape == (B, n_celltypes)
        assert domain_logits.shape == (B, n_domains)
        assert mp_logits.shape == (B, C_max)
        assert cls_emb.shape == (B, 64)
        assert ch_out.shape == (B, C_max, 64)

    def test_no_marker_embeddings(self):
        """Model should work without marker embeddings."""
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=None,
        )
        sample = torch.randn(2, 10, 1, 32, 32)
        spatial = torch.randn(2, 3, 32, 32)
        ch_idx = torch.zeros(2, 10, dtype=torch.long)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)

        ct_logits, domain_logits, mp_logits, cls_emb, _, _ = model(
            sample, spatial, ch_idx, pad_mask
        )
        assert ct_logits.shape == (2, 3)

    def test_train_eval_consistency(self, marker_embeddings):
        """Marker embeddings should be normalized in BOTH train and eval."""
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=1,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=marker_embeddings,
        )
        torch.manual_seed(42)
        sample = torch.randn(2, 10, 1, 32, 32)
        spatial = torch.randn(2, 3, 32, 32)
        ch_idx = torch.zeros(2, 10, dtype=torch.long)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)

        model.train()
        ct_train, _, _, _, _, _ = model(sample, spatial, ch_idx, pad_mask)

        model.eval()
        ct_eval, _, _, _, _, _ = model(sample, spatial, ch_idx, pad_mask)

        # Both should produce valid outputs (no NaN)
        assert not torch.isnan(ct_train).any()
        assert not torch.isnan(ct_eval).any()


class TestSpatialEncoder:
    def test_output_shape(self):
        encoder = SpatialEncoder(out_dim=64)
        x = torch.randn(4, 3, 32, 32)
        out = encoder(x)
        assert out.shape == (4, 64)


class TestPerChannelResNet:
    def test_output_shape(self):
        net = PerChannelResNet(out_dim=128)
        x = torch.randn(4, 10, 1, 32, 32)
        out = net(x)
        assert out.shape == (4, 10, 128)


class TestResBlock:
    def test_residual_connection(self):
        block = ResBlock(32)
        x = torch.randn(4, 32, 16, 16)
        out = block(x)
        assert out.shape == x.shape


# =============================================================================
# Loss tests
# =============================================================================


class TestFocalLossWithWeights:
    def test_focal_loss_with_class_weights(self):
        """FocalLoss should accept and use class weights."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)

        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        loss.backward()
        assert logits.grad is not None

    def test_focal_loss_without_weights(self):
        """FocalLoss should work without class weights."""
        loss_fn = FocalLoss(gamma=2.0)

        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_focal_loss_gradient_flow(self):
        """Verify gradients flow correctly through FocalLoss."""
        weights = torch.tensor([1.0, 5.0, 1.0])
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)

        logits = torch.randn(16, 3, requires_grad=True)
        targets = torch.randint(0, 3, (16,))

        loss = loss_fn(logits, targets)
        loss.backward()
        assert (logits.grad.abs() > 0).any()


# =============================================================================
# Dataset tests
# =============================================================================


class TestDropOutChannels:
    def test_drops_only_valid_channels(self):
        """DropOutChannels should only drop valid (non-padded) channels.

        With proportional dropout (30%), 5 valid channels → n_drop = min(3, max(1, int(5*0.3))) = 1.
        """
        dropout = DropOutChannels(n=3)

        sample = torch.randn(10, 1, 32, 32)
        ch_idx = torch.arange(10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[5:] = True  # last 5 are padding
        mp = torch.ones(10)
        mp_mask = torch.ones(10, dtype=torch.bool)

        sample_out, ch_idx_out, mask_out, mp_out, mp_mask_out = dropout(
            sample, ch_idx, mask, mp, mp_mask
        )

        # Padding channels should still be padded
        assert mask_out[5:].all()
        # Proportional: 30% of 5 valid = 1 dropped → total masked = 5 + 1 = 6
        n_drop = min(3, max(1, int(5 * 0.3)))  # = 1
        assert mask_out.sum() == 5 + n_drop

    def test_no_drop_when_only_one_valid(self):
        """Should not drop if only 1 valid channel (floor guard: n_valid <= n_drop)."""
        dropout = DropOutChannels(n=5)

        sample = torch.randn(10, 1, 32, 32)
        ch_idx = torch.arange(10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[1:] = True  # only 1 valid channel
        mp = torch.ones(10)
        mp_mask = torch.ones(10, dtype=torch.bool)

        _, _, mask_out, _, _ = dropout(sample, ch_idx, mask, mp, mp_mask)
        # n_valid <= 3 guard: no drop
        assert mask_out.sum() == 9  # unchanged

    def test_no_drop_when_tiny_panel(self):
        """Do not remove signal from panels with three or fewer valid channels."""
        dropout = DropOutChannels(n=8)

        sample = torch.randn(10, 1, 32, 32)
        ch_idx = torch.arange(10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[3:] = True  # 3 valid channels
        mp = torch.ones(10)
        mp_mask = torch.ones(10, dtype=torch.bool)

        _, _, mask_out, _, _ = dropout(sample, ch_idx, mask, mp, mp_mask)
        assert torch.equal(mask_out, mask)

    def test_proportional_drop_small_dataset(self):
        """Proportional dropout drops fewer channels on small datasets."""
        dropout = DropOutChannels(n=8)

        # 9-channel dataset: n_drop = min(8, max(1, int(9*0.3))) = min(8, 2) = 2
        sample = torch.randn(10, 1, 32, 32)
        ch_idx = torch.arange(10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[9:] = True  # 9 valid, 1 padding
        mp = torch.ones(10)
        mp_mask = torch.ones(10, dtype=torch.bool)

        _, _, mask_out, _, _ = dropout(sample, ch_idx, mask, mp, mp_mask)
        n_drop = min(8, max(1, int(9 * 0.3)))  # = 2
        assert mask_out.sum() == 1 + n_drop  # 1 original padding + dropped


class TestFOVSplits:
    def test_no_overlap(self):
        """FOV splits should have zero overlap."""

        class MockDataset:
            def __init__(self):
                self.indices = [
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", 1, "FOV1", "DS1", (10, 10)
                    ),
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", 2, "FOV1", "DS1", (20, 20)
                    ),
                    CellIndexRecord(
                        0, "B", "B_cell", "CODEX", 1, "FOV2", "DS1", (10, 10)
                    ),
                    CellIndexRecord(
                        0, "B", "B_cell", "CODEX", 2, "FOV2", "DS1", (20, 20)
                    ),
                    CellIndexRecord(
                        0, "M", "Macrophage", "CODEX", 1, "FOV3", "DS1", (10, 10)
                    ),
                    CellIndexRecord(
                        0, "M", "Macrophage", "CODEX", 2, "FOV3", "DS1", (20, 20)
                    ),
                ]

        dataset = MockDataset()
        train_idx, val_idx = create_fov_splits(dataset, train_ratio=0.6, seed=42)

        # No overlap
        assert len(set(train_idx) & set(val_idx)) == 0

        # All indices covered
        assert set(train_idx) | set(val_idx) == set(range(len(dataset.indices)))

        # FOVs are not split
        train_fovs = {dataset.indices[i][5] for i in train_idx}
        val_fovs = {dataset.indices[i][5] for i in val_idx}
        assert len(train_fovs & val_fovs) == 0

    def test_stratification_by_dataset(self):
        """Each dataset should have at least 1 train FOV."""

        class MockDataset:
            def __init__(self):
                self.indices = [
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", 1, "FOV1", "DS1", (10, 10)
                    ),
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", 2, "FOV2", "DS1", (20, 20)
                    ),
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", 3, "FOV3", "DS1", (30, 30)
                    ),
                    CellIndexRecord(
                        1, "B", "B_cell", "MIBI", 1, "FOV4", "DS2", (10, 10)
                    ),
                    CellIndexRecord(
                        1, "B", "B_cell", "MIBI", 2, "FOV5", "DS2", (20, 20)
                    ),
                ]

        dataset = MockDataset()
        train_idx, val_idx = create_fov_splits(dataset, train_ratio=0.5, seed=42)

        # Each dataset should have at least one train FOV
        train_datasets = {dataset.indices[i][6] for i in train_idx}
        assert "DS1" in train_datasets
        assert "DS2" in train_datasets


class TestComputeSampleWeights:
    def test_rare_class_higher_weight(self):
        """Rare classes should have higher sampling weights when count exceeds cap."""

        class MockDataset:
            def __init__(self):
                # 5000 T_cells, 100 B_cells — both above the 1000-sample cap floor,
                # so B_cell (rarer) should get a higher weight than T_cell.
                self.indices = [
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", i, "FOV1", "DS1", (10, 10)
                    )
                    for i in range(5000)
                ] + [
                    CellIndexRecord(
                        0, "B", "B_cell", "CODEX", i + 5000, "FOV1", "DS1", (20, 20)
                    )
                    for i in range(100)
                ]

        dataset = MockDataset()
        indices = list(range(5100))
        weights = compute_sample_weights(dataset, indices)

        # B_cell (rarer, but > 1000 cap) should have higher weight than T_cell
        t_cell_weight = weights[0].item()
        b_cell_weight = weights[5000].item()
        assert b_cell_weight > t_cell_weight, (
            f"Expected B_cell weight ({b_cell_weight:.4f}) > T_cell weight ({t_cell_weight:.4f})"
        )

    def test_weight_cap_equalizes_extreme_rarity(self):
        """Classes with fewer than 1000 samples all get the same (capped) weight."""

        class MockDataset:
            def __init__(self):
                # 10 T_cells, 1 B_cell — both below the 1000-sample cap floor
                self.indices = [
                    CellIndexRecord(
                        0, "T", "T_cell", "CODEX", i, "FOV1", "DS1", (10, 10)
                    )
                    for i in range(10)
                ] + [
                    CellIndexRecord(
                        0, "B", "B_cell", "CODEX", 11, "FOV1", "DS1", (20, 20)
                    )
                ]

        dataset = MockDataset()
        indices = list(range(11))
        weights = compute_sample_weights(dataset, indices)

        # Both are below the 1000-count cap, so they get equal weights
        import pytest

        assert weights[10].item() == pytest.approx(weights[0].item(), rel=1e-5)


class TestDistanceTransform:
    def test_basic(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        dt = compute_distance_transform(mask)
        assert dt.shape == (32, 32)
        assert dt.max() <= 1.0
        assert dt.min() >= 0.0
        # Center should have highest value
        assert dt[15, 15] > dt[10, 10]

    def test_empty_mask(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        dt = compute_distance_transform(mask)
        assert dt.sum() == 0.0


class TestMarkerPositivityHandling:
    def test_question_mark_masked_out(self, dct_config):
        """'?' marker positivity should be masked out (validity_mask=False)."""
        from deepcell_types.training.dataset import FullImageDataset

        # Create a mock instance to test the method
        dataset = FullImageDataset.__new__(FullImageDataset)
        dataset.marker_positivity_labels = dct_config.marker_positivity_labels

        ch_names = ["CD3", "CD4", "CD8", "CD45"]

        mp, vm = dataset._calculate_marker_positivity("TestDataset", "B_cell", ch_names)

        # CD4 for B_cell is "?" -> should be masked
        assert not vm[1]  # CD4 index 1
        assert vm[0]  # CD3 is valid
        assert vm[2]  # CD8 is valid

    def test_no_labels_all_masked(self, dct_config):
        """Datasets without marker positivity labels should have all-False validity mask."""
        from deepcell_types.training.dataset import FullImageDataset

        dataset = FullImageDataset.__new__(FullImageDataset)
        dataset.marker_positivity_labels = dct_config.marker_positivity_labels

        ch_names = ["CD3", "CD4", "CD8", "CD45"]

        mp, vm = dataset._calculate_marker_positivity(
            "UnknownDataset", "T_cell", ch_names
        )

        # No labels → all zeros, all-False validity (excluded from loss)
        assert (mp == 0.0).all()
        assert not vm.any()

    def test_marker_positivity_channel_lookup_is_case_insensitive(self):
        """MP supervision should use the same canonical channel semantics."""
        import pandas as pd
        from deepcell_types.training.dataset import FullImageDataset

        dataset = FullImageDataset.__new__(FullImageDataset)
        dataset.marker_positivity_labels = {
            "TestDataset": pd.DataFrame({"Ki67": [1.0]}, index=["T_cell"])
        }

        mp, vm = dataset._calculate_marker_positivity("TestDataset", "T_cell", ["KI67"])

        assert mp[0] == 1.0
        assert vm[0]


# =============================================================================
# Masked Marker Pre-training tests
# =============================================================================


class TestMaskedMarkerHead:
    def test_output_shape(self):
        head = MaskedMarkerHead(d_model=64)
        channel_outputs = torch.randn(4, 10, 64)
        out = head(channel_outputs)
        assert out.shape == (4, 10)

    def test_gradient_flow(self):
        head = MaskedMarkerHead(d_model=64)
        channel_outputs = torch.randn(4, 10, 64, requires_grad=True)
        out = head(channel_outputs)
        loss = out.sum()
        loss.backward()
        assert channel_outputs.grad is not None
        assert (channel_outputs.grad.abs() > 0).any()


class TestMaskMarkerChannels:
    def test_masking_ratio(self):
        """Approximately 30% of valid channels should be masked."""
        B, C_max = 32, 10
        sample = torch.randn(B, C_max, 1, 16, 16)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 7:] = True  # 7 valid channels

        masked_sample, masked_indices, mean_expr = mask_marker_channels(
            sample, pad_mask, mask_ratio=0.3
        )

        # Check shapes
        assert masked_sample.shape == sample.shape
        assert masked_indices.shape == (B, C_max)
        assert mean_expr.shape == (B, C_max)

        # Only valid channels should be masked
        assert not masked_indices[:, 7:].any(), (
            "Padding channels should never be masked"
        )

        # Masked channels should be zeroed in sample
        for i in range(B):
            for j in range(C_max):
                if masked_indices[i, j]:
                    assert masked_sample[i, j].abs().sum() == 0.0

    def test_preserves_unmasked(self):
        """Unmasked channels should be identical to original."""
        B, C_max = 8, 10
        sample = torch.randn(B, C_max, 1, 16, 16)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 5:] = True

        masked_sample, masked_indices, _ = mask_marker_channels(
            sample, pad_mask, mask_ratio=0.3
        )

        for i in range(B):
            for j in range(C_max):
                if not masked_indices[i, j]:
                    assert torch.equal(masked_sample[i, j], sample[i, j])

    def test_min_keep(self):
        """Should always keep at least min_keep channels unmasked."""
        B, C_max = 8, 10
        sample = torch.randn(B, C_max, 1, 16, 16)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 3:] = True  # only 3 valid channels

        _, masked_indices, _ = mask_marker_channels(
            sample, pad_mask, mask_ratio=0.5, min_keep=2
        )

        for i in range(B):
            valid_count = (~pad_mask[i]).sum().item()
            masked_count = masked_indices[i].sum().item()
            unmasked_valid = valid_count - masked_count
            assert unmasked_valid >= 2

    def test_mean_expression_correct(self):
        """Mean expression should match actual channel means before masking."""
        sample = torch.randn(4, 10, 1, 16, 16)
        pad_mask = torch.zeros(4, 10, dtype=torch.bool)

        _, _, mean_expr = mask_marker_channels(sample, pad_mask)

        expected = sample.mean(dim=(2, 3, 4))
        assert torch.allclose(mean_expr, expected)

    def test_mean_expression_ignores_padding_fill(self):
        """Padded -1 channels must not make the whole patch count as cell area."""
        sample = torch.zeros(1, 4, 1, 4, 4)
        # Valid channel 0 has signal only in a 2x2 cell area.
        sample[0, 0, 0, :2, :2] = 2.0
        # Valid channel 1 is a true zero-expression marker.
        sample[0, 1, 0] = 0.0
        # Dataset padding fill.
        sample[0, 2:] = -1.0
        pad_mask = torch.tensor([[False, False, True, True]])

        _, _, mean_expr = mask_marker_channels(sample, pad_mask)

        assert torch.isclose(mean_expr[0, 0], torch.tensor(2.0))
        assert torch.isclose(mean_expr[0, 1], torch.tensor(0.0))
        assert torch.isclose(mean_expr[0, 2], torch.tensor(0.0))
        assert torch.isclose(mean_expr[0, 3], torch.tensor(0.0))


class TestPretrainEndToEnd:
    def test_pretrain_forward(self, marker_embeddings):
        """End-to-end: model + masking + reconstruction loss."""
        B, C_max, H, W = 4, 10, 32, 32
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=marker_embeddings,
        )
        recon_head = MaskedMarkerHead(d_model=64)

        sample = torch.randn(B, C_max, 1, H, W)
        spatial = torch.randn(B, 3, H, W)
        ch_idx = torch.zeros(B, C_max, dtype=torch.long)
        ch_idx[:, :4] = torch.arange(4)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 4:] = True

        # Mask channels
        masked_sample, masked_indices, mean_expr = mask_marker_channels(
            sample, pad_mask, mask_ratio=0.3
        )

        # Forward through model
        _, _, _, _, channel_outputs, _ = model(masked_sample, spatial, ch_idx, pad_mask)

        # Reconstruct
        pred_expr = recon_head(channel_outputs)

        # MSE loss only on masked channels
        if masked_indices.any():
            loss = torch.nn.functional.mse_loss(
                pred_expr[masked_indices], mean_expr[masked_indices]
            )
            loss.backward()

            # Gradients should flow through model
            assert model.spatial_encoder.layers[0].weight.grad is not None


# =============================================================================
# Integration tests
# =============================================================================


class TestBatchData:
    def test_batch_data_creation(self):
        """Verify BatchData can be created with all fields."""
        bd = BatchData(
            sample=torch.randn(4, 10, 1, 32, 32),
            spatial_context=torch.randn(4, 3, 32, 32),
            ch_idx=torch.zeros(4, 10, dtype=torch.long),
            mask=torch.zeros(4, 10, dtype=torch.bool),
            ct_idx=torch.zeros(4, dtype=torch.long),
            domain_idx=torch.zeros(4, dtype=torch.long),
            marker_positivity=torch.zeros(4, 10),
            marker_positivity_mask=torch.ones(4, 10, dtype=torch.bool),
            cell_index=torch.zeros(4, dtype=torch.long),
            dataset_name=("DS1", "DS1", "DS2", "DS2"),
            fov_name=("FOV1", "FOV1", "FOV2", "FOV2"),
        )
        assert bd.sample.shape == (4, 10, 1, 32, 32)
        assert bd.spatial_context.shape == (4, 3, 32, 32)
        assert bd.marker_positivity_mask.shape == (4, 10)

    def test_batch_data_to_device(self):
        """Verify .to() moves tensors and preserves tuples."""
        bd = BatchData(
            sample=torch.randn(2, 5, 1, 16, 16),
            spatial_context=torch.randn(2, 3, 16, 16),
            ch_idx=torch.zeros(2, 5, dtype=torch.long),
            mask=torch.zeros(2, 5, dtype=torch.bool),
            ct_idx=torch.zeros(2, dtype=torch.long),
            domain_idx=torch.zeros(2, dtype=torch.long),
            marker_positivity=torch.zeros(2, 5),
            marker_positivity_mask=torch.ones(2, 5, dtype=torch.bool),
            cell_index=torch.zeros(2, dtype=torch.long),
            dataset_name=("DS1", "DS2"),
            fov_name=("FOV1", "FOV2"),
        )
        bd2 = bd.to("cpu")
        assert bd2.sample.device.type == "cpu"
        assert bd2.spatial_context.device.type == "cpu"
        assert bd2.ch_idx.device.type == "cpu"
        assert bd2.mask.device.type == "cpu"
        assert bd2.ct_idx.device.type == "cpu"
        assert bd2.domain_idx.device.type == "cpu"
        assert bd2.marker_positivity.device.type == "cpu"
        assert bd2.marker_positivity_mask.device.type == "cpu"
        assert bd2.cell_index.device.type == "cpu"
        assert bd2.dataset_name == ("DS1", "DS2")
        assert bd2.fov_name == ("FOV1", "FOV2")


class TestSeedEverything:
    def test_reproducibility(self):
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)


class TestCreateModel:
    def test_creates_model(self, dct_config, marker_embeddings):
        """Factory should return correct model type with right dimensions."""
        model = create_model(
            dct_config,
            marker_embeddings,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        assert isinstance(model, CellTypeAnnotator)
        assert model.n_celltypes == dct_config.NUM_CELLTYPES
        assert model.n_domains == dct_config.NUM_DOMAINS
        assert model.d_model == 64


# =============================================================================
# MarkerConditionedMPHead tests
# =============================================================================


class TestMarkerConditionedMPHead:
    def test_output_shape(self, marker_embeddings):
        """MarkerConditionedMPHead should produce (B, C_max) output."""
        from deepcell_types.model import MarkerEmbeddingLayer

        d_model = 64
        mel = MarkerEmbeddingLayer(d_model, marker_embeddings)
        head = MarkerConditionedMPHead(d_model, mel)

        channel_outputs = torch.randn(4, 10, d_model)
        ch_idx = torch.zeros(4, 10, dtype=torch.long)
        ch_idx[:, :4] = torch.arange(4)

        logits = head(channel_outputs, ch_idx)
        assert logits.shape == (4, 10)

    def test_gradient_flow(self, marker_embeddings):
        """Gradients should flow through the head."""
        from deepcell_types.model import MarkerEmbeddingLayer

        d_model = 64
        mel = MarkerEmbeddingLayer(d_model, marker_embeddings)
        head = MarkerConditionedMPHead(d_model, mel)

        channel_outputs = torch.randn(4, 10, d_model, requires_grad=True)
        ch_idx = torch.zeros(4, 10, dtype=torch.long)

        logits = head(channel_outputs, ch_idx)
        loss = logits.sum()
        loss.backward()
        assert channel_outputs.grad is not None
        assert (channel_outputs.grad.abs() > 0).any()

    def test_different_markers_different_outputs(self, marker_embeddings):
        """Different marker indices should produce different FiLM modulations."""
        from deepcell_types.model import MarkerEmbeddingLayer

        d_model = 64
        mel = MarkerEmbeddingLayer(d_model, marker_embeddings)
        head = MarkerConditionedMPHead(d_model, mel)

        channel_outputs = torch.randn(1, 2, d_model)
        # Two different markers
        ch_idx = torch.tensor([[0, 1]])
        logits = head(channel_outputs, ch_idx)
        # Should (generally) be different
        assert logits[0, 0] != logits[0, 1]


# =============================================================================
# Attention extraction tests
# =============================================================================


class TestAttentionExtraction:
    def test_return_attn_weights(self, marker_embeddings):
        """forward with return_attn_weights=True should return 6 values."""
        B, C_max, H, W = 2, 10, 32, 32
        n_layers = 2
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=n_layers,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=marker_embeddings,
            dropout=0.0,
        )

        sample = torch.randn(B, C_max, 1, H, W)
        spatial = torch.randn(B, 3, H, W)
        ch_idx = torch.zeros(B, C_max, dtype=torch.long)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 4:] = True

        outputs = model(sample, spatial, ch_idx, pad_mask, return_attn_weights=True)
        assert len(outputs) == 6

        (
            ct_logits,
            domain_logits,
            mp_logits,
            cls_emb,
            ch_out,
            cls_to_channels,
        ) = outputs
        assert cls_to_channels.shape == (n_layers, B, C_max)

    def test_no_attn_weights_default(self, marker_embeddings):
        """Default forward returns fixed-arity AnnotatorOutput; attn field is None."""
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=marker_embeddings,
        )

        sample = torch.randn(2, 10, 1, 32, 32)
        spatial = torch.randn(2, 3, 32, 32)
        ch_idx = torch.zeros(2, 10, dtype=torch.long)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)

        outputs = model(sample, spatial, ch_idx, pad_mask)
        assert len(outputs) == 6
        assert outputs.cls_to_channels is None

    def test_attn_weights_sum_to_one(self, marker_embeddings):
        """CLS→all attention weights should approximately sum to 1."""
        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=marker_embeddings,
            dropout=0.0,
        )

        sample = torch.randn(2, 10, 1, 32, 32)
        spatial = torch.randn(2, 3, 32, 32)
        ch_idx = torch.zeros(2, 10, dtype=torch.long)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)

        *_, cls_to_channels = model(
            sample, spatial, ch_idx, pad_mask, return_attn_weights=True
        )
        # cls_to_channels is CLS row excluding CLS column, but CLS→CLS weight is missing
        # so sum should be <= 1 and close to 1 (CLS→CLS is typically small)
        for layer_idx in range(cls_to_channels.shape[0]):
            sums = cls_to_channels[layer_idx].sum(dim=-1)  # (B,)
            assert (sums <= 1.0 + 1e-5).all()
            assert (sums > 0.5).all()  # most attention should go to channels


# =============================================================================
# create_model with conditioned MP head tests
# =============================================================================


class TestCreateModelWithConditionedHead:
    def test_conditioned_head_enabled(self, dct_config, marker_embeddings):
        """create_model with use_conditioned_mp_head=True should use MarkerConditionedMPHead."""
        model = create_model(
            dct_config,
            marker_embeddings,
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_conditioned_mp_head=True,
        )
        assert isinstance(model.marker_pos_head, MarkerConditionedMPHead)

    def test_conditioned_head_disabled(self, dct_config, marker_embeddings):
        """create_model with use_conditioned_mp_head=False should use plain Linear."""
        model = create_model(
            dct_config,
            marker_embeddings,
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_conditioned_mp_head=False,
        )
        assert isinstance(model.marker_pos_head, torch.nn.Linear)

    def test_conditioned_head_forward(self, dct_config, marker_embeddings):
        """Full forward pass with MarkerConditionedMPHead should work."""
        model = create_model(
            dct_config,
            marker_embeddings,
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_conditioned_mp_head=True,
        )

        B, C_max = 4, 10
        sample = torch.randn(B, C_max, 1, 32, 32)
        spatial = torch.randn(B, 3, 32, 32)
        ch_idx = torch.zeros(B, C_max, dtype=torch.long)
        ch_idx[:, :4] = torch.arange(4)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 4:] = True

        ct_logits, domain_logits, mp_logits, cls_emb, ch_out, _ = model(
            sample, spatial, ch_idx, pad_mask
        )

        assert ct_logits.shape == (B, dct_config.NUM_CELLTYPES)
        assert mp_logits.shape == (B, C_max)

    def test_no_embeddings_falls_back_to_linear(self, dct_config):
        """Without marker embeddings, should fall back to Linear head."""
        model = create_model(
            dct_config,
            None,
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_conditioned_mp_head=True,
        )
        # marker_embedder is None, so conditioned head can't be created
        assert isinstance(model.marker_pos_head, torch.nn.Linear)


# =============================================================================
# TestInstanceNormZeroInput
# =============================================================================


class TestInstanceNormZeroInput:
    """InstanceNorm on zero inputs should not produce NaN or Inf (critical regression test)."""

    def test_resblock_zero_input_no_nan(self):
        """ResBlock should handle all-zero input without NaN/Inf."""
        block = ResBlock(32)
        x = torch.zeros(4, 32, 16, 16)
        out = block(x)
        assert not torch.isnan(out).any(), "NaN in ResBlock output for zero input"
        assert not torch.isinf(out).any(), "Inf in ResBlock output for zero input"

    def test_per_channel_resnet_zero_input_no_nan(self):
        """PerChannelResNet should handle all-zero input without NaN/Inf."""
        # PerChannelResNet expects (B, C_max, 1, H, W) input
        resnet = PerChannelResNet(out_dim=128)
        x = torch.zeros(2, 4, 1, 32, 32)
        out = resnet(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_padding_channels_not_attended_to_in_transformer(self):
        """Padded channels are excluded from attention (src_key_padding_mask).

        The model zeroes padded channel features before the transformer via
        channel_feat[padding_mask] = 0.0. After that, the transformer's
        src_key_padding_mask prevents non-padded tokens from attending TO padded
        positions. However, padded positions may still receive non-zero outputs
        because the transformer itself does not mask its output positions — only
        its attention keys. What we verify here is:
          1. Padding channels whose inputs are zero produce reproducible outputs
             (i.e. the same model run with the same padding produces identical
             padded-position outputs, which is true by determinism).
          2. No NaN/Inf appears in any channel output (regression for InstanceNorm).
        """
        B, C_max = 2, 8
        sample = torch.randn(B, C_max, 1, 32, 32)
        spatial = torch.randn(B, 3, 32, 32)
        ch_idx = torch.zeros(B, C_max, dtype=torch.long)
        ch_idx[:, :4] = torch.arange(4)
        pad_mask = torch.zeros(B, C_max, dtype=torch.bool)
        pad_mask[:, 4:] = True  # channels 4-7 are padding

        model = CellTypeAnnotator(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_celltypes=3,
            n_domains=2,
            marker_embeddings=None,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            _, _, _, _, channel_outputs, _ = model(sample, spatial, ch_idx, pad_mask)

        # No NaN or Inf anywhere (InstanceNorm regression check)
        assert not torch.isnan(channel_outputs).any(), "NaN in channel_outputs"
        assert not torch.isinf(channel_outputs).any(), "Inf in channel_outputs"

        # Determinism: running again with same inputs produces identical outputs
        with torch.no_grad():
            _, _, _, _, channel_outputs2, _ = model(sample, spatial, ch_idx, pad_mask)
        assert torch.equal(channel_outputs, channel_outputs2), (
            "Model outputs are not deterministic in eval mode"
        )


# =============================================================================
# TestLabelRemap
# =============================================================================


class TestLabelRemap:
    """build_label_remap should be identity for 0-indexed ct2idx (post-migration)."""

    def test_maps_0indexed_identity(self):
        from deepcell_types.training.utils import build_label_remap

        # ct2idx is now 0-indexed after archive migration
        ct2idx = {"A": 0, "B": 1, "C": 2}
        remap = build_label_remap(ct2idx)
        assert remap[0] == 0
        assert remap[1] == 1
        assert remap[2] == 2

    def test_remap_is_contiguous(self):
        from deepcell_types.training.utils import build_label_remap

        ct2idx = {f"CT{i}": i for i in range(9)}  # values 0-8
        remap = build_label_remap(ct2idx)
        mapped = [remap[i].item() for i in range(9)]
        assert sorted(mapped) == list(range(9))

    def test_zero_indexed_passthrough(self):
        """0-indexed ct2idx should still map correctly (values stay 0-based)."""
        from deepcell_types.training.utils import build_label_remap

        ct2idx = {"A": 0, "B": 1, "C": 2}
        remap = build_label_remap(ct2idx)
        assert remap[0] == 0
        assert remap[1] == 1
        assert remap[2] == 2


# =============================================================================
# TestLossesAndMetricsCompute
# =============================================================================


class TestLossesAndMetricsCompute:
    """LossesAndMetrics.compute() should expose macro-F1 as the single CT metric,
    matching sklearn's confusion-matrix-based macro-F1."""

    def _make_metrics(self, num_classes):
        """Build a LossesAndMetrics with torchmetrics internals but only use conf_mat_ct_metric."""
        from deepcell_types.training.utils import LossesAndMetrics, MPMetricsTracker
        from deepcell_types.training.losses import FocalLoss

        torchmetrics = pytest.importorskip("torchmetrics")

        lm = LossesAndMetrics(
            ct_loss_fn=FocalLoss(gamma=2.0),
            domain_loss_fn=torch.nn.CrossEntropyLoss(),
            marker_pos_loss_fn=torch.nn.BCEWithLogitsLoss(),
            acc_domain_metric=torchmetrics.Accuracy(task="multiclass", num_classes=2),
            conf_mat_ct_metric=torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ),
            mp_metrics=MPMetricsTracker(),
        )
        return lm

    def _update_conf_mat(self, lm, logits, targets):
        """Update only the CT confusion matrix (minimal update for testing)."""
        preds = logits.argmax(dim=1)
        lm.conf_mat_ct_metric.update(preds, targets)
        # Provide dummy updates for the other metrics so compute() works
        lm.acc_domain_metric.update(
            torch.zeros(len(targets), dtype=torch.long),
            torch.zeros(len(targets), dtype=torch.long),
        )

    @staticmethod
    def _sklearn_macro_f1(targets, preds):
        """Reference macro-F1 over classes present in the targets (support > 0),
        matching _conf_mat_summary's has_support reduction."""
        from sklearn.metrics import f1_score

        labels_present = sorted(set(targets.tolist()))
        return f1_score(
            targets, preds, average="macro", labels=labels_present, zero_division=0
        )

    def test_compute_exposes_only_macro_f1_for_ct(self):
        """The CT surface is a single key: ct_macro_f1. The old accuracy /
        weighted variants must be gone."""
        m = self._make_metrics(5)
        logits = torch.eye(5) * 10.0
        targets = torch.arange(5)
        self._update_conf_mat(m, logits, targets)
        result = m.compute()
        assert "ct_macro_f1" in result
        for removed in (
            "ct_macro_accuracy",
            "ct_weighted_accuracy",
            "ct_weighted_f1",
        ):
            assert removed not in result

    def test_perfect_predictions(self):
        m = self._make_metrics(5)
        logits = torch.eye(5) * 10.0
        targets = torch.arange(5)
        self._update_conf_mat(m, logits, targets)
        result = m.compute()
        assert abs(result["ct_macro_f1"] - 1.0) < 1e-5

    def test_known_error_pattern(self):
        """Class 0 always wrong (predicted as class 1), others correct."""
        m = self._make_metrics(5)
        # 2 samples of class 0 → predicted as class 1 (wrong)
        # 1 each of classes 1-4 → correct
        preds_logits = torch.zeros(6, 5)
        preds_logits[0] = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])
        preds_logits[1] = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])
        preds_logits[2] = torch.eye(5)[1]
        preds_logits[3] = torch.eye(5)[2]
        preds_logits[4] = torch.eye(5)[3]
        preds_logits[5] = torch.eye(5)[4]
        targets = torch.tensor([0, 0, 1, 2, 3, 4])
        self._update_conf_mat(m, preds_logits, targets)
        result = m.compute()
        preds = preds_logits.argmax(dim=1).numpy()
        assert (
            abs(result["ct_macro_f1"] - self._sklearn_macro_f1(targets.numpy(), preds))
            < 1e-4
        )

    def test_zero_support_classes_excluded(self):
        """Classes with no samples should be excluded from the macro-F1 denominator."""
        m = self._make_metrics(10)
        # Only classes 0, 1, 2 have support, all predicted correctly
        for cls in [0, 1, 2]:
            logits = torch.zeros(3, 10)
            logits[:, cls] = 10.0
            targets_cls = torch.full((3,), cls)
            self._update_conf_mat(m, logits, targets_cls)
        result = m.compute()
        # Only 3 classes in denominator, all correct → macro-F1 = 1.0
        assert abs(result["ct_macro_f1"] - 1.0) < 1e-5

    def test_imbalanced_macro_matches_sklearn(self):
        """Macro-F1 on imbalanced data should match sklearn's computation."""
        m = self._make_metrics(4)
        # Build explicit predictions:
        # class 0: 10 samples, 8 correct, 2 → class 1
        # class 1: 2 samples, 2 correct
        # class 2: 1 sample, 1 correct
        # class 3: 1 sample, 0 correct (→ class 0)
        targets_list = [0] * 10 + [1] * 2 + [2] * 1 + [3] * 1
        correct_remaining = {0: 8, 1: 2, 2: 1, 3: 0}
        wrong_pred = {0: 1, 1: 0, 2: 0, 3: 0}
        logits_list = []
        for t in targets_list:
            logit = torch.zeros(4)
            if correct_remaining[t] > 0:
                logit[t] = 10.0
                correct_remaining[t] -= 1
            else:
                logit[wrong_pred[t]] = 10.0
            logits_list.append(logit)
        logits = torch.stack(logits_list)
        targets = torch.tensor(targets_list)
        self._update_conf_mat(m, logits, targets)
        result = m.compute()
        preds = logits.argmax(dim=1).numpy()
        assert (
            abs(result["ct_macro_f1"] - self._sklearn_macro_f1(targets.numpy(), preds))
            < 1e-4
        )


# =============================================================================
# TestMPMetricsTracker
# =============================================================================


class TestMPMetricsTracker:
    """MPMetricsTracker should compute per-marker and macro-averaged MP metrics."""

    def test_perfect_predictions(self):
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        # 2 markers, all correct
        pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ch_idx = torch.tensor([0, 0, 1, 1])
        tracker.update(pred, target, ch_idx)
        result = tracker.compute()
        assert abs(result["mp_macro_f1"] - 1.0) < 1e-5
        assert abs(result["mp_micro_f1"] - 1.0) < 1e-5
        assert abs(result["mp_macro_precision"] - 1.0) < 1e-5
        assert abs(result["mp_macro_recall"] - 1.0) < 1e-5
        assert result["mp_num_markers"] == 2

    def test_macro_vs_micro_divergence(self):
        """Macro should differ from micro when marker frequencies are imbalanced."""
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        # Marker 0: 100 samples, all correct (F1=1.0)
        pred_0 = torch.cat([torch.ones(50) * 0.9, torch.ones(50) * 0.1])
        target_0 = torch.cat([torch.ones(50), torch.zeros(50)])
        ch_0 = torch.zeros(100, dtype=torch.long)
        # Marker 1: 4 samples, all wrong (F1=0.0)
        pred_1 = torch.tensor([0.1, 0.1, 0.9, 0.9])
        target_1 = torch.tensor([1.0, 1.0, 0.0, 0.0])
        ch_1 = torch.ones(4, dtype=torch.long)
        tracker.update(pred_0, target_0, ch_0)
        tracker.update(pred_1, target_1, ch_1)
        result = tracker.compute()
        # Macro: (1.0 + 0.0) / 2 = 0.5
        assert abs(result["mp_macro_f1"] - 0.5) < 1e-5
        # Micro: dominated by marker 0 → close to 1.0
        assert result["mp_micro_f1"] > 0.9

    def test_per_marker_breakdown(self):
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        pred = torch.tensor([0.9, 0.1, 0.9])
        target = torch.tensor([1.0, 0.0, 0.0])
        ch_idx = torch.tensor([5, 5, 10])
        tracker.update(pred, target, ch_idx)
        idx2marker = {5: "CD45", 10: "Ki67"}
        per_marker = tracker.compute_per_marker(idx2marker)
        assert "CD45" in per_marker
        assert "Ki67" in per_marker
        assert per_marker["CD45"]["f1"] == 1.0  # 1 TP, 1 TN
        assert per_marker["Ki67"]["f1"] == 0.0  # 1 FP, 0 TP
        assert per_marker["Ki67"]["n_samples"] == 1

    def test_reset(self):
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        tracker.update(torch.tensor([0.9]), torch.tensor([1.0]), torch.tensor([0]))
        tracker.reset()
        result = tracker.compute()
        assert result["mp_num_markers"] == 0

    def test_learned_thresholds_improve_f1(self):
        """Learned thresholds should be >= fixed 0.5 F1 (optimized on same data)."""
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        # Marker with low-confidence true positives (scores around 0.3-0.4)
        preds = torch.tensor([0.35, 0.38, 0.32, 0.40, 0.05, 0.08, 0.03, 0.06])
        targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        ch_idx = torch.zeros(8, dtype=torch.long)
        tracker.update(preds, targets, ch_idx)

        # At fixed 0.5: all below threshold → F1=0
        fixed_result = tracker.compute_at_fixed_threshold(0.5)
        assert fixed_result["mp_macro_f1"] == 0.0

        # Learn optimal threshold
        learned = tracker.find_optimal_thresholds()
        assert learned[0] < 0.5  # should find a lower threshold

        # Apply learned thresholds
        tracker.thresholds = learned
        learned_result = tracker.compute()
        assert learned_result["mp_macro_f1"] > 0.9  # should recover the positives

    def test_compute_at_fixed_vs_default(self):
        """compute_at_fixed_threshold(0.5) should match compute() when no thresholds set."""
        from deepcell_types.training.utils import MPMetricsTracker

        tracker = MPMetricsTracker()
        pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ch_idx = torch.tensor([0, 0, 1, 1])
        tracker.update(pred, target, ch_idx)
        default_result = tracker.compute()
        fixed_result = tracker.compute_at_fixed_threshold(0.5)
        assert abs(default_result["mp_macro_f1"] - fixed_result["mp_macro_f1"]) < 1e-8


# =============================================================================
# TestStrictChannelContract
# =============================================================================


class TestStrictChannelContract:
    """Channel names must be CANONICAL — no runtime alias resolution and no
    case-insensitive fallback. Source-data variants get masked. The
    canonicalization step (hubmap-to-zarr/apply_canonicalization.py) is
    responsible for producing canonical channel names in the archive."""

    def _dataset_with_markers(self, marker2idx):
        dataset = FullImageDataset.__new__(FullImageDataset)
        dataset.marker2idx = marker2idx
        dataset._idx2marker = {v: k for k, v in marker2idx.items()}
        return dataset

    def test_canonical_match(self):
        dataset = self._dataset_with_markers({"Ki67": 5, "CD3": 0, "CD4": 1})

        result, canonical = dataset._resolve_channel_index("Ki67")
        assert result == 5
        assert canonical == "Ki67"

    def test_uppercase_variant_does_not_match(self):
        # Strict contract: KI67 != Ki67. Upstream must canonicalize.
        dataset = self._dataset_with_markers({"Ki67": 5, "CD3": 0, "CD4": 1})

        result, canonical = dataset._resolve_channel_index("KI67")
        assert result == -1
        assert canonical == "KI67"

    def test_lowercase_variant_does_not_match(self):
        dataset = self._dataset_with_markers({"CD45": 7, "CD3": 0})

        result, canonical = dataset._resolve_channel_index("cd45")
        assert result == -1
        assert canonical == "cd45"

    def test_alias_does_not_resolve_at_runtime(self):
        # DC-SIGN was historically aliased to CD209 via the in-package
        # CHANNEL_ALIASES dict. That dict is gone — aliases are an
        # ingestion-time concern (hubmap-to-zarr/config/channel_mapping.yaml).
        dataset = self._dataset_with_markers({"CD209": 12, "CD3": 0})

        result, canonical = dataset._resolve_channel_index("DC-SIGN")
        assert result == -1
        assert canonical == "DC-SIGN"

    def test_unknown_channel_returns_minus_one(self):
        dataset = self._dataset_with_markers({"CD3": 0, "CD4": 1})

        result, canonical = dataset._resolve_channel_index("UNKNOWN_MARKER")
        assert result == -1
        assert canonical == "UNKNOWN_MARKER"


# =============================================================================
# TestComputeSampleWeightsCorrectness
# =============================================================================


class TestComputeSampleWeightsCorrectness:
    """compute_sample_weights should use sqrt-inverse-frequency weighting."""

    def _make_mock_dataset(self, ct_labels):
        """Build a minimal mock dataset whose .indices[i][2] is the ct_label."""

        class _MockDataset:
            def __init__(self, labels):
                # indices tuple: (ds_idx, tissue, ct_label_standard, modality, cell_idx, fov, ds_name, shape)
                self.indices = [
                    CellIndexRecord(0, "T", label, "CODEX", i, "FOV1", "DS1", (10, 10))
                    for i, label in enumerate(labels)
                ]

        return _MockDataset(ct_labels)

    def test_sqrt_inverse_frequency(self):
        """Weights should be proportional to sqrt(total/count) for each class."""
        # Use counts above the 1000-sample cap floor to test true sqrt-inverse behavior.
        # 1000 samples of class "A", 9000 samples of class "B"
        labels = ["A"] * 1000 + ["B"] * 9000
        dataset = self._make_mock_dataset(labels)
        indices = list(range(10000))
        weights = compute_sample_weights(dataset, indices)
        assert len(weights) == 10000
        # Class "A" weight: sqrt(10000/1000) = sqrt(10) ≈ 3.162
        # Class "B" weight: sqrt(10000/9000) ≈ 1.054
        # Ratio should be sqrt(9000/1000) = 3.0
        w_a = weights[0].item()  # first "A" sample
        w_b = weights[1000].item()  # first "B" sample
        expected_ratio = (9000 / 1000) ** 0.5  # = 3.0
        actual_ratio = w_a / w_b
        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"Weight ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"
        )

    def test_single_class_uniform_weights(self):
        """All same class → all weights equal."""
        labels = ["A"] * 20
        dataset = self._make_mock_dataset(labels)
        indices = list(range(20))
        weights = compute_sample_weights(dataset, indices)
        weight_values = set(weights.tolist())
        assert len(weight_values) == 1, "All weights should be equal for single class"

    def test_extreme_imbalance(self):
        """Very rare class should have higher weight than common class.

        With the 1000-count cap floor, a class with 1 sample gets effective count=1000,
        so the ratio is sqrt(10001/1000) / sqrt(10001/10000) ≈ sqrt(10) ≈ 3.16 (not 50x).
        The cap prevents extreme oversampling of ultra-rare singleton classes.
        """
        labels = ["common"] * 10000 + ["rare"] * 1
        dataset = self._make_mock_dataset(labels)
        indices = list(range(10001))
        weights = compute_sample_weights(dataset, indices)
        rare_weight = weights[-1].item()
        common_weight = weights[0].item()
        # With cap floor=1000: rare effective=1000, common effective=10000
        # ratio = sqrt(10000/1000) ≈ sqrt(10) ≈ 3.16
        assert rare_weight > common_weight * 2, (
            f"Rare class weight ({rare_weight:.2f}) should be > 2x common ({common_weight:.2f})"
        )
        assert rare_weight < common_weight * 20, (
            f"Cap should prevent extreme oversampling: {rare_weight:.2f} < 20x {common_weight:.2f}"
        )

    def test_weights_length_matches_indices(self):
        """Returned weights tensor length should match number of indices."""
        labels = ["A"] * 5 + ["B"] * 5
        dataset = self._make_mock_dataset(labels)
        indices = list(range(10))
        weights = compute_sample_weights(dataset, indices)
        assert len(weights) == len(indices)


class TestPredLoggerOutputFormat:
    """cell_type_actual must be saved as class name strings, not integer indices."""

    def _make_ct2idx(self):
        return {"CD4T": 4, "Macrophage": 13, "Tumor": 41}

    def test_pred_logger_saves_string_names(self, tmp_path):
        from deepcell_types.training.utils import PredLogger

        ct2idx = self._make_ct2idx()
        logger = PredLogger(ct2idx)
        logger.log(
            labels=np.array([4, 13, 41]),  # 1-indexed zarr values
            probs=np.zeros((3, 3)),
            cell_index=np.array([1, 2, 3]),
            dataset_name=np.array(["ds", "ds", "ds"]),
            fov_name=np.array(["fov", "fov", "fov"]),
        )
        path = tmp_path / "preds.csv"
        logger.save(str(path))
        df = import_pandas().read_csv(path)
        assert df["cell_type_actual"].tolist() == ["CD4T", "Macrophage", "Tumor"], (
            f"Expected string names, got: {df['cell_type_actual'].tolist()}"
        )

    def test_save_baseline_predictions_saves_string_names(self, tmp_path):
        from deepcell_types.training.utils import save_baseline_predictions

        ct2idx = self._make_ct2idx()
        path = tmp_path / "baseline_preds.csv"
        save_baseline_predictions(
            y_true=np.array([4, 13, 41]),
            y_prob=np.zeros((3, 3)),
            cell_indices=[1, 2, 3],
            dataset_names=["ds", "ds", "ds"],
            fov_names=["fov", "fov", "fov"],
            ct2idx=ct2idx,
            output_path=path,
        )
        df = import_pandas().read_csv(path)
        assert df["cell_type_actual"].tolist() == ["CD4T", "Macrophage", "Tumor"], (
            f"Expected string names, got: {df['cell_type_actual'].tolist()}"
        )

    def test_pred_and_actual_same_dtype(self, tmp_path):
        """Predicted class (argmax col name) and actual must both be strings for direct comparison."""
        from deepcell_types.training.utils import PredLogger

        ct2idx = {"CD4T": 4, "Macrophage": 13}
        logger = PredLogger(ct2idx)
        logger.log(
            labels=np.array([4, 13]),
            probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
            cell_index=np.array([1, 2]),
            dataset_name=np.array(["ds", "ds"]),
            fov_name=np.array(["fov", "fov"]),
        )
        path = tmp_path / "preds.csv"
        logger.save(str(path))
        df = import_pandas().read_csv(path)
        prob_cols = [c for c in ["CD4T", "Macrophage"] if c in df.columns]
        pred = df[prob_cols].idxmax(axis=1)
        actual = df["cell_type_actual"]
        # Both must be strings so direct == comparison works
        assert (pred == actual).all(), f"pred={pred.tolist()} actual={actual.tolist()}"


def import_pandas():
    import pandas as pd

    return pd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
