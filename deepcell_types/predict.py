import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .annotator_model import create_model as create_annotator_model
from .dataset import PatchDataset
from .dct_kit.config import DCTConfig
from .model import CellTypeCLIPModel


LEGACY_EMBEDDING_MODEL = "deepseek-r1-70b-llama-distill-q4_K_M"
LEGACY_EMBEDDING_DIM = 8192


class PredLogger:
    def __init__(self, dct_config):
        self.dct_config = dct_config
        self.probs = []
        self.cell_index = []

    def log(self, probs, cell_index):
        self.probs.append(probs)
        self.cell_index.append(cell_index)

    def get_result(self):
        idx2ct = {v: k for k, v in self.dct_config.ct2idx.items()}
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        order = np.argsort(cell_index, kind="stable")
        probs = probs[order]
        cell_index = cell_index[order]

        top_probs = np.max(probs, axis=1)
        cell_type_int_pred = np.argmax(probs, axis=1)
        cell_type_str_pred = [idx2ct[i] for i in cell_type_int_pred]

        return cell_type_str_pred, top_probs, cell_index


def _torch_load_weights(path, device):
    """Load a checkpoint with `weights_only=True`.

    The kwarg landed in PyTorch 1.13. Deserializing without it on an
    untrusted checkpoint allows arbitrary code execution at load time via
    pickle. We require a torch new enough to support it; the fallback only
    fires for the literal `TypeError` from `torch.load` itself rejecting
    the kwarg, and emits a loud warning. Older torch installs should be
    upgraded rather than silently bypassing the safety check.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as e:
        warnings.warn(
            f"torch.load(weights_only=True) unsupported in this torch "
            f"version ({torch.__version__}); falling back to unsafe pickle "
            f"load. Upgrade torch to >=1.13 to enable the safety check. "
            f"Error: {e}",
            stacklevel=2,
        )
        return torch.load(path, map_location=device)


def _model_path(model_name):
    candidate = Path(model_name).expanduser()
    if (
        candidate.exists()
        or candidate.suffix in {".pt", ".pth"}
        or candidate.parent != Path(".")
    ):
        return candidate
    return Path.home() / ".deepcell" / "models" / f"{model_name}.pt"


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _is_canonical_checkpoint(state_dict):
    return (
        isinstance(state_dict, dict)
        and "channel_encoder.stem.0.weight" in state_dict
        and "ct_head.6.weight" in state_dict
    )


def _count_transformer_layers(state_dict):
    indices = set()
    prefix = "transformer_layers."
    for key in state_dict:
        if key.startswith(prefix):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.add(int(parts[1]))
    return max(indices) + 1 if indices else 4


def _infer_spatial_pool_size(state_dict):
    fusion_in = state_dict["fusion.weight"].shape[1]
    channel_dim = state_dict["channel_encoder.proj.weight"].shape[0]
    spatial_dim = fusion_in - channel_dim
    if spatial_dim <= 0:
        return 1
    pool_area = max(1, spatial_dim // 64)
    return int(round(pool_area**0.5))


def _infer_n_domains(state_dict):
    return state_dict["domain_head.8.weight"].shape[0]


def _build_canonical_model(checkpoint, dct_config, device):
    state_dict = _state_dict(checkpoint)
    marker_weight = state_dict["marker_embedder.embed_layer.weight"]
    n_markers = marker_weight.shape[0] - 1
    embedding_dim = marker_weight.shape[1]
    n_celltypes = state_dict["ct_head.6.weight"].shape[0]

    if n_markers != len(dct_config.marker2idx):
        raise ValueError(
            f"Checkpoint expects {n_markers} markers, but the canonical config "
            f"has {len(dct_config.marker2idx)}."
        )
    if n_celltypes != len(dct_config.ct2idx):
        raise ValueError(
            f"Checkpoint expects {n_celltypes} cell types, but the canonical "
            f"config has {len(dct_config.ct2idx)}."
        )

    marker_embeddings = np.zeros((n_markers, embedding_dim), dtype=np.float32)
    lora_rank = (
        state_dict["marker_embedder.lora_A.weight"].shape[0]
        if "marker_embedder.lora_A.weight" in state_dict
        else 0
    )
    use_conditioned_mp_head = "marker_pos_head.film_scale.weight" in state_dict
    has_tumor_head = any(key.startswith("tumor_head.") for key in state_dict)

    model = create_annotator_model(
        dct_config,
        marker_embeddings,
        d_model=state_dict["cls_token"].shape[-1],
        n_layers=_count_transformer_layers(state_dict),
        n_domains=_infer_n_domains(state_dict),
        resnet_base_channels=state_dict["channel_encoder.stem.0.weight"].shape[0],
        lora_rank=lora_rank,
        spatial_pool_size=_infer_spatial_pool_size(state_dict),
        tumor_head=has_tumor_head,
        use_conditioned_mp_head=use_conditioned_mp_head,
    )
    model.load_state_dict(state_dict)
    return model.to(device)


def _load_legacy_embeddings(dct_config):
    ct2embedding = dct_config.get_celltype_embedding(
        embedding_model_name=LEGACY_EMBEDDING_MODEL
    )
    ct_embeddings = np.zeros(
        (len(dct_config.ct2idx), LEGACY_EMBEDDING_DIM), dtype=np.float32
    )
    for cell_type, embedding in ct2embedding.items():
        if cell_type in dct_config.ct2idx:
            ct_embeddings[dct_config.ct2idx[cell_type]] = embedding

    marker2embedding = dct_config.get_channel_embedding(
        embedding_model_name=LEGACY_EMBEDDING_MODEL
    )
    marker_embeddings = np.zeros(
        (len(marker2embedding), LEGACY_EMBEDDING_DIM), dtype=np.float32
    )
    for marker, embedding in marker2embedding.items():
        if marker in dct_config.marker2idx:
            marker_embeddings[dct_config.marker2idx[marker]] = embedding

    return ct_embeddings, marker_embeddings


def _build_legacy_model(checkpoint, dct_config, device):
    ct_embeddings, marker_embeddings = _load_legacy_embeddings(dct_config)
    model = CellTypeCLIPModel(
        n_filters=256,
        n_heads=4,
        n_celltypes=len(dct_config.ct2idx),
        n_domains=dct_config.NUM_DOMAINS,
        marker_embeddings=marker_embeddings,
        embedding_dim=LEGACY_EMBEDDING_DIM,
        ct_embeddings=ct_embeddings,
        img_feature_extractor="conv",
    )
    model.load_state_dict(_state_dict(checkpoint))
    return model.to(device)


def _excluded_celltype_indices(dct_config, tissue, batch_size):
    if tissue is None:
        return None
    tct = dct_config.get_tct_mapping()
    if tissue not in tct:
        raise ValueError(f"Unknown tissue_exclude={tissue!r}")
    allowed = {
        dct_config.ct2idx[name] for name in tct[tissue] if name in dct_config.ct2idx
    }
    return [
        [idx for idx in range(len(dct_config.ct2idx)) if idx not in allowed]
        for _ in range(batch_size)
    ]


def predict(
    raw,
    mask,
    channel_names,
    mpp,
    model_name,
    device_num,
    batch_size=256,
    num_workers=24,
    tissue_exclude=None,
    zarr_path=None,
):
    """Run the cell-type prediction pipeline.

    Given a spatial proteomics image `raw`, a corresponding segmentation `mask`,
    and a list of markers (`channel_names`) corresponding to the channels of `raw`,
    predict the cell type associated with each index in `mask`.

    Parameters
    ----------
    raw : A spatial proteomic image as an `numpy.ndarray` with shape ``(C, W, H)``.
        A 2D multiplexed image in channel-first format. The image will be converted
        internally to ``dtype=np.float32``.
    mask : 2D label image
        Segmentation mask of `raw` as a 2D label image with shape ``(W, H)``.
    channel_names : list of str
        A list of channel markers. Must have the same length as the number of channels
        in `raw` and be given in the same order as the channels in `raw`.
    mpp : float
        The image resolution in microns-per-pixel. Improves prediction performance by
        removing scale variability.
    model_name : str
        Name of the pre-trained model to use for inference. Models are searched for
        at ``Path.home() / ".deepcell/models"``. A filesystem path to a ``.pt`` file
        is also accepted.
    device_num : `torch.device` or `str`
        Which device to run inference on. For example, ``"cpu"`` or ``"cuda"``.
        To specify a specific GPU on multi-GPU systems, try ``"cuda:<device_num>``,
        e.g. ``"cuda:0"``.
    batch_size : int, default=256
        Batch size to be used for inference. Larger `batch_size` will increase
        performance by increasing VRAM usage. Default value of 256 is conservative
        and should be appropriate for systems with <16GB VRAM.
    num_workers : int, default=24
        Number of threads to use for loading data. Increasing `num_workers` may result
        in large increases in CPU memory footprint. Only recommended for systems with
        ``>64 GB`` RAM.
    tissue_exclude : str, optional, default=None
        If provided, limit the cell type prediction to only those categories known to
        be associated with the specified tissue type.
    zarr_path : str or pathlib.Path, optional, default=None
        Canonical checkpoints now read marker and cell type metadata from a TissueNet
        zarr archive. Pass the archive path here or set the
        ``DEEPCELL_TYPES_ZARR_PATH`` environment variable.

    Returns
    -------
    list of str
        A list whose ``len`` is equal to the number of unique cell indices in `mask`,
        ordered by ascending cell index.
    """
    device = torch.device(device_num)
    checkpoint = _torch_load_weights(_model_path(model_name), device)
    state_dict = _state_dict(checkpoint)
    canonical = _is_canonical_checkpoint(state_dict)

    dct_config = DCTConfig(
        profile="canonical" if canonical else "legacy",
        zarr_path=zarr_path if canonical else None,
    )
    model = (
        _build_canonical_model(checkpoint, dct_config, device)
        if canonical
        else _build_legacy_model(checkpoint, dct_config, device)
    )

    pred_logger = PredLogger(dct_config)
    dataset = PatchDataset(
        raw,
        mask,
        channel_names,
        mpp,
        dct_config,
        output_mode="canonical" if canonical else "legacy",
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()
    with torch.no_grad():
        if canonical:
            for sample, spatial_context, ch_idx, attn_mask, cell_index in tqdm(
                data_loader, desc="(inference)"
            ):
                ct_exclude = _excluded_celltype_indices(
                    dct_config, tissue_exclude, len(sample)
                )
                ct_logits, *_ = model(
                    sample.to(device),
                    spatial_context.to(device),
                    ch_idx.to(device),
                    attn_mask.to(device),
                    ct_exclude=ct_exclude,
                )
                probs = F.softmax(ct_logits, dim=-1)
                pred_logger.log(
                    probs=probs.cpu().detach().numpy(),
                    cell_index=cell_index.detach().cpu().numpy(),
                )
        else:
            for sample, ch_idx, attn_mask, cell_index in tqdm(
                data_loader, desc="(inference)"
            ):
                ct_exclude = _excluded_celltype_indices(
                    dct_config, tissue_exclude, len(sample)
                )
                _, _, _, _, probs, _ = model(
                    sample.to(device),
                    ch_idx.to(device),
                    attn_mask.to(device),
                    ct_exclude=ct_exclude,
                )
                pred_logger.log(
                    probs=probs.cpu().detach().numpy(),
                    cell_index=cell_index.detach().cpu().numpy(),
                )

    return pred_logger.get_result()[0]
