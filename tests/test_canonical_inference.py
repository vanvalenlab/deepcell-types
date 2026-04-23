import numpy as np
import torch

from deepcell_types.annotator_model import create_model
from deepcell_types.dataset import PatchDataset
from deepcell_types.dct_kit.config import DCTConfig
from deepcell_types.model import CellTypeCLIPModel
from deepcell_types.predict import LEGACY_EMBEDDING_DIM, predict


def test_canonical_config_matches_latest_design_contract():
    config = DCTConfig(profile="canonical")

    assert config.MAX_NUM_CHANNELS == 80
    assert config.CROP_SIZE == 32
    assert config.NUM_DOMAINS == 8
    assert len(config.ct2idx) == 51
    assert len(config.marker2idx) == 269
    assert config.ct2idx["Tumor"] == 50
    assert config.marker2idx["CD45"] >= 0


def test_patch_dataset_can_emit_canonical_factored_batch():
    config = DCTConfig(profile="canonical")
    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    dataset = PatchDataset(
        raw,
        mask,
        ["CD45"],
        0.5,
        config,
        output_mode="canonical",
    )
    sample, spatial_context, ch_idx, attn_mask, cell_index = next(iter(dataset))

    assert sample.shape == (80, 1, 32, 32)
    assert spatial_context.shape == (3, 32, 32)
    assert ch_idx.shape == (80,)
    assert attn_mask.shape == (80,)
    assert attn_mask[0].item() is False
    assert attn_mask[1:].all().item() is True
    assert cell_index == 1


def test_patch_dataset_default_preserves_legacy_batch_shape():
    config = DCTConfig()
    raw = np.ones((1, 80, 80), dtype=np.float32)
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[20:60, 20:60] = 1

    dataset = PatchDataset(raw, mask, ["CD45"], 0.5, config)
    sample, ch_idx, attn_mask, cell_index = next(iter(dataset))

    assert sample.shape == (75, 3, 64, 64)
    assert ch_idx.shape == (75,)
    assert attn_mask.shape == (75,)
    assert cell_index == 1


def test_predict_accepts_canonical_checkpoint_path(tmp_path):
    config = DCTConfig(profile="canonical")
    marker_embeddings = np.zeros((len(config.marker2idx), 8), dtype=np.float32)
    model = create_model(
        config,
        marker_embeddings,
        d_model=32,
        n_heads=8,
        n_layers=1,
        resnet_base_channels=4,
    )
    checkpoint_path = tmp_path / "canonical.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    raw = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[12:28, 12:28] = 1

    cell_types = predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        str(checkpoint_path),
        "cpu",
        batch_size=1,
        num_workers=0,
    )

    assert len(cell_types) == 1
    assert cell_types[0] in config.ct2idx


def test_predict_still_accepts_legacy_checkpoint_path(tmp_path):
    config = DCTConfig(profile="legacy")
    ct_embeddings = np.zeros(
        (len(config.ct2idx), LEGACY_EMBEDDING_DIM), dtype=np.float32
    )
    marker_embeddings = np.zeros(
        (len(config.marker2idx), LEGACY_EMBEDDING_DIM), dtype=np.float32
    )
    model = CellTypeCLIPModel(
        n_filters=256,
        n_heads=4,
        n_celltypes=len(config.ct2idx),
        n_domains=config.NUM_DOMAINS,
        marker_embeddings=marker_embeddings,
        embedding_dim=LEGACY_EMBEDDING_DIM,
        ct_embeddings=ct_embeddings,
        img_feature_extractor="conv",
    )
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), checkpoint_path)

    raw = np.ones((1, 80, 80), dtype=np.float32)
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[20:60, 20:60] = 1

    cell_types = predict(
        raw,
        mask,
        ["CD45"],
        0.5,
        str(checkpoint_path),
        "cpu",
        batch_size=1,
        num_workers=0,
    )

    assert len(cell_types) == 1
    assert cell_types[0] in config.ct2idx
