import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .abstention import ABSTENTION_LABEL, compute_iqr_fence
from .config import DCTConfig

__all__ = ["predict", "PredictionResult"]

# NOTE: torch (and the torch-importing .model / .dataset modules) are imported
# lazily inside the functions that use them, NOT at module scope, so that
# `import deepcell_types` stays torch-free until predict() is actually called
# (torch is a ~1.4s import). Keep it that way — do not add a top-level
# `import torch` here.

def _names_ordered_by_index(mapping):
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def validate_checkpoint_vocabulary(checkpoint, ct2idx, marker2idx):
    """Reject a checkpoint/vocabulary pairing whose ORDERING would mislabel cells.

    The count check in :func:`_build_model` only catches a wrong *number* of
    cell types or markers; a permuted mapping of the right size passes it and
    then silently mislabels every prediction. This guards ordering for both the
    cell-type and marker sides, and is shared by the Python API
    (:func:`_build_model`) and the evaluation CLI (``scripts/predict.py``) so
    they cannot drift apart.

    - The checkpoint must bundle its own ``ct2idx``, and it must match ``ct2idx``
      (name -> index mapping, order-independent).
    - If the checkpoint bundles ``canonical_channels`` it must match the marker
      ordering (compared by numeric marker index, not dict insertion order).

    Raises ``ValueError`` on any mismatch.
    """
    ckpt_ct2idx = checkpoint.get("ct2idx") if isinstance(checkpoint, dict) else None
    if ckpt_ct2idx is None:
        raise ValueError(
            "Checkpoint does not bundle a ct2idx, so the cell-type ordering "
            "cannot be verified and a permuted vocabulary would silently "
            "mislabel every cell. Use a checkpoint that bundles its own ct2idx "
            "(scripts/train.py and retrain_head.py record it)."
        )
    if dict(ckpt_ct2idx) != dict(ct2idx):
        raise ValueError(
            "Checkpoint ct2idx ordering does not match the inference "
            "vocabulary's. The model would mislabel cell types; use the "
            "archive the checkpoint was trained against."
        )

    ckpt_channels = (
        checkpoint.get("canonical_channels") if isinstance(checkpoint, dict) else None
    )
    marker_order = _names_ordered_by_index(marker2idx)
    if ckpt_channels is not None and list(ckpt_channels) != marker_order:
        raise ValueError(
            "Checkpoint marker ordering (canonical_channels) does not match the "
            "inference vocabulary's marker2idx ordering."
        )


@dataclass(frozen=True)
class PredictionResult:
    """Structured ``predict()`` output when ``return_probabilities=True``.

    cell_types : list[str]
        Predicted cell-type name for each unique cell index in ``mask``,
        ordered by ascending cell index. When the opt-in IQR-fence abstention
        is enabled (``ct_abstention_k`` set to a float), cells it flags carry
        the sentinel ``"Unknown"`` here and their original argmax label is in
        ``cell_types_raw``; with abstention off (the default) this equals
        ``cell_types_raw``.
    probabilities : np.ndarray, shape (n_cells, n_celltypes)
        Per-cell softmax probabilities across all cell-type classes (same
        order as ``cell_types``). Column ``i`` corresponds to cell type
        ``i`` in ``dct_config.ct2idx``.
    cell_indices : np.ndarray, shape (n_cells,)
        The unique mask indices, in the same order as ``cell_types``.
    abstained : np.ndarray, shape (n_cells,), dtype=bool
        ``True`` for cells whose max-softmax fell below the IQR fence (= the
        ones rewritten to ``"Unknown"`` in ``cell_types``). All-``False`` when
        ``ct_abstention_k`` is disabled or when the FOV has fewer than 4 cells
        (IQR is undefined on a tiny sample).
    cell_types_raw : list[str]
        Pre-abstention argmax label for every cell — useful when callers
        want to inspect what abstained cells would have been classified as.
    """

    cell_types: List[str]
    probabilities: np.ndarray
    cell_indices: np.ndarray
    abstained: np.ndarray
    cell_types_raw: List[str]


class _InferenceResultBuffer:
    """Private accumulator for per-batch inference outputs.

    Not part of the public API — see training/utils.py:PredLogger for the
    training-side logger with a different (5-field, CSV-saving) interface.
    """

    def __init__(self, dct_config):
        self.dct_config = dct_config
        self.probs = []
        self.cell_index = []

    def log(self, probs, cell_index):
        self.probs.append(probs)
        self.cell_index.append(cell_index)

    def get_result(self):
        """Return (cell_type_str_pred, top_probs, cell_index, probs).

        ``probs`` is the full per-cell softmax matrix in cell_index order; the
        first three elements stay as before for back-compat with callers.
        """
        idx2ct = {v: k for k, v in self.dct_config.ct2idx.items()}
        n_celltypes = len(self.dct_config.ct2idx)
        if not self.probs:
            # No cells (e.g. an all-background mask): return empty, well-typed
            # arrays rather than letting np.concatenate([]) raise.
            return (
                [],
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, n_celltypes), dtype=np.float32),
            )
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        order = np.argsort(cell_index, kind="stable")
        probs = probs[order]
        cell_index = cell_index[order]

        top_probs = np.max(probs, axis=1)
        cell_type_int_pred = np.argmax(probs, axis=1)
        cell_type_str_pred = [idx2ct[i] for i in cell_type_int_pred]

        return cell_type_str_pred, top_probs, cell_index, probs


def _torch_load_weights(path, device):
    """Load a checkpoint with `weights_only=True`.

    The kwarg landed in PyTorch 1.13. Deserializing without it on an
    untrusted checkpoint allows arbitrary code execution at load time via
    pickle. We require a torch new enough to support it; the fallback only
    fires for the literal `TypeError` from `torch.load` itself rejecting
    the kwarg, and emits a loud warning. Older torch installs should be
    upgraded rather than silently bypassing the safety check.
    """
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as e:
        # Only fall back for the specific "kwarg unsupported" TypeError; a
        # TypeError raised while unpickling a malformed checkpoint must not be
        # silently retried with the safety check disabled.
        if "weights_only" not in str(e):
            raise
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


def _infer_ct_head_params(state_dict):
    """Infer residual-MLP CT-head construction params from a checkpoint state dict."""
    block_ids = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("ct_head.blocks.") and key.endswith(".0.weight")
    }
    return {
        "ct_head_width": int(state_dict["ct_head.inp.0.weight"].shape[0]),
        "ct_head_depth": len(block_ids),
        "n_celltypes": int(state_dict["ct_head.out.weight"].shape[0]),
    }


def _build_model(checkpoint, dct_config, device):
    from .model import create_model

    state_dict = _state_dict(checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    try:
        marker_weight = state_dict["marker_embedder.embed_layer.weight"]
        n_markers = marker_weight.shape[0] - 1
        embedding_dim = marker_weight.shape[1]
        ct_head_params = _infer_ct_head_params(state_dict)
        n_celltypes = ct_head_params["n_celltypes"]
        d_model = state_dict["cls_token"].shape[-1]
        resnet_base_channels = state_dict["channel_encoder.stem.0.weight"].shape[0]
    except KeyError as e:
        raise ValueError(
            f"Checkpoint is missing the expected key {e}; this does not look like "
            "a deepcell-types CellTypeAnnotator checkpoint."
        ) from e

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

    # Require the inference vocabulary to agree on ORDERING, not just counts
    # (a permuted mapping of the right size passes the count checks above but
    # silently mislabels every prediction). Validated against the checkpoint's
    # own bundled ct2idx / canonical_channels. Shared with scripts/predict.py.
    validate_checkpoint_vocabulary(checkpoint, dct_config.ct2idx, dct_config.marker2idx)

    marker_embeddings = np.zeros((n_markers, embedding_dim), dtype=np.float32)
    use_conditioned_mp_head = "marker_pos_head.film_scale.weight" in state_dict

    # n_heads and compat_marker0_zero are NOT recoverable from tensor shapes
    # (MultiheadAttention params are head-count-independent; compat_marker0_zero
    # is a pure-Python numerics flag), so they must be recorded in the checkpoint
    # config. scripts/train.py and retrain_head.py bundle both; a wrong value
    # loads with silently wrong numerics, so require them rather than guess.
    missing_arch_keys = [
        k for k in ("n_heads", "compat_marker0_zero") if k not in config
    ]
    if missing_arch_keys:
        raise ValueError(
            f"Checkpoint config is missing {', '.join(missing_arch_keys)}. These "
            "are not recoverable from the weights and a wrong value loads with "
            "silently wrong numerics; scripts/train.py records them in every "
            "checkpoint."
        )
    n_heads = int(config["n_heads"])
    compat_marker0_zero = bool(config["compat_marker0_zero"])

    model = create_model(
        dct_config,
        marker_embeddings,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=_count_transformer_layers(state_dict),
        n_domains=_infer_n_domains(state_dict),
        resnet_base_channels=resnet_base_channels,
        spatial_pool_size=_infer_spatial_pool_size(state_dict),
        use_conditioned_mp_head=use_conditioned_mp_head,
        compat_marker0_zero=compat_marker0_zero,
        ct_head_width=ct_head_params["ct_head_width"],
        ct_head_depth=ct_head_params["ct_head_depth"],
    )
    model.load_state_dict(state_dict)
    return model.to(device)


def predict(
    raw,
    mask,
    channel_names,
    mpp,
    *,
    model_name,
    device=None,
    device_num=None,
    batch_size=256,
    num_workers=0,
    zarr_path=None,
    return_probabilities=False,
    ct_abstention_k=None,
    preprocess=None,
) -> "list[str] | PredictionResult":
    """Run the cell-type prediction pipeline.

    Given a spatial proteomics image `raw`, a corresponding segmentation `mask`,
    and a list of markers (`channel_names`) corresponding to the channels of `raw`,
    predict the cell type associated with each index in `mask`.

    Parameters
    ----------
    raw : A spatial proteomic image as an `numpy.ndarray` with shape ``(C, H, W)``.
        A 2D multiplexed image in channel-first format. The image will be converted
        internally to ``dtype=np.float32``.
    mask : 2D label image
        Segmentation mask of `raw` as a 2D label image with shape ``(H, W)``.
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
    device : `torch.device` or `str`
        Which device to run inference on, e.g. ``"cpu"``, ``"cuda"``, or
        ``"cuda:0"`` to select a specific GPU. All arguments after `mpp` are
        keyword-only.
    device_num : `torch.device` or `str`, optional
        Deprecated alias for `device`, retained for backward compatibility. If
        both are given, `device` takes precedence.
    batch_size : int, default=256
        Batch size to be used for inference. Larger `batch_size` will increase
        performance by increasing VRAM usage. Default value of 256 is conservative
        and should be appropriate for systems with <16GB VRAM.
    num_workers : int, default=0
        Number of DataLoader worker processes. Default ``0`` runs the patch
        generator in-process (safe on all machines). ``PatchDataset`` is an
        ``IterableDataset`` that holds the full FOV in memory, so each worker
        is an extra copy AND re-runs the per-FOV preprocessing — only raise
        this on machines with abundant RAM and CPU.
    zarr_path : str or pathlib.Path, optional, default=None
        Optional path to a TissueNet zarr archive to read the marker / cell-type
        registry from (or set ``DEEPCELL_TYPES_ZARR_PATH``). When omitted, the
        registry is read from the vocabulary snapshot shipped with the package,
        so inference does not require the archive.
    return_probabilities : bool, default=False
        If False (default, back-compat), returns a list of cell-type names.
        If True, returns a :class:`PredictionResult` with the full per-cell
        softmax probability matrix and the cell indices.
    ct_abstention_k : float or None, default=None
        IQR-fence post-hoc abstention multiplier. Abstention is **opt-in**:
        the default ``None`` returns the raw argmax cell-type label for every
        cell and never relabels to ``"Unknown"``. When set to a float, the
        fence is ``Q1 - k*IQR`` on the per-FOV cell-wise max-softmax
        distribution and cells below it are relabelled to ``"Unknown"``
        (``k=0.2`` reproduces the paper headline operating point; pass
        ``return_probabilities=True`` to recover the pre-abstention labels and
        the ``abstained`` mask). Has no effect on FOVs with fewer than 4 cells
        (the IQR is undefined).
    preprocess : callable, optional, default=None
        Custom per-FOV preprocessing hook. Called as
        ``preprocess(raw, channel_names) -> raw`` where ``raw`` is a
        ``(C, H, W)`` float32 array already resampled to the model's target
        MPP and restricted to in-vocabulary channels, and ``channel_names``
        are the resolved standard marker names aligned to ``raw``. Must return
        a ``(C, H, W)`` array in ``[0, 1]``. When ``None`` (default), the
        built-in per-channel p99.9 clip + min-max normalization is used. Build
        one declaratively with :func:`deepcell_types.make_preprocessor`.
        Note: :func:`deepcell_types.preprocess_fov` is NOT a valid hook — it has
        a different signature (takes ``mask`` / keyword-only ``native_mpp``) and
        returns a :class:`PreprocessedFov`, not a bare ``(C, H, W)`` array.

    Returns
    -------
    list of str
        (default) Predicted cell-type name for each unique cell index in
        ``mask``, ordered by ascending cell index. Cells flagged by the
        IQR-fence abstention carry the sentinel ``"Unknown"`` — but only
        when ``ct_abstention_k`` is set; with the default ``None`` every
        cell carries its raw argmax label.
    PredictionResult
        (when ``return_probabilities=True``) Full per-cell probabilities,
        cell indices, predicted names, and an ``abstained`` boolean
        mask. See :class:`PredictionResult`.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from .dataset import PatchDataset

    if device_num is not None:
        warnings.warn(
            "predict(device_num=...) is deprecated; use device=... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if device is None:
        device = device_num
    if device is None:
        raise TypeError(
            "predict() requires a device, e.g. device='cpu' or device='cuda:0'."
        )
    device = torch.device(device)

    model_file = _model_path(model_name)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model {model_name!r} not found at {model_file}. Download it with "
            "deepcell_types.utils.download_model(version=...), or pass a filesystem "
            "path to a .pt file as model_name."
        )
    checkpoint = _torch_load_weights(model_file, device)

    dct_config = DCTConfig(zarr_path=zarr_path)
    model = _build_model(checkpoint, dct_config, device)

    pred_logger = _InferenceResultBuffer(dct_config)
    dataset = PatchDataset(
        raw, mask, channel_names, mpp, dct_config, preprocess=preprocess
    )
    # Unique cell ids in the caller's mask (ascending; background 0 excluded).
    # Used below to keep the returned labels aligned to every input cell even if
    # a small cell is lost when the mask is downscaled to the model's MPP.
    input_cell_ids = np.unique(np.asarray(mask).astype(np.int64))
    input_cell_ids = input_cell_ids[input_cell_ids != 0]
    # The dataset now holds its own working copies (a float32 raw and a float32
    # mask), so drop the caller-supplied references. When the caller passes the
    # arrays inline (no other reference), this frees the full-resolution source
    # for the duration of the inference loop instead of pinning it to the end.
    del raw, mask
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()
    with torch.no_grad():
        for sample, spatial_context, ch_idx, attn_mask, cell_index in tqdm(
            data_loader, desc="(inference)"
        ):
            ct_logits, *_ = model(
                sample.to(device),
                spatial_context.to(device),
                ch_idx.to(device),
                attn_mask.to(device),
            )
            probs = F.softmax(ct_logits, dim=-1)
            pred_logger.log(
                probs=probs.cpu().detach().numpy(),
                cell_index=cell_index.detach().cpu().numpy(),
            )

    cell_types_raw, _top_probs, cell_indices, full_probs = pred_logger.get_result()

    # A custom model / preprocess hook can emit non-finite logits; softmax then
    # yields NaN and np.argmax silently labels the cell class 0. Fail loudly
    # rather than return confident garbage. (compute_iqr_fence drops non-finite
    # values, so abstention would not catch this on its own.)
    if full_probs.size and not np.isfinite(full_probs).all():
        raise ValueError(
            "Model produced non-finite class probabilities (NaN/inf); the "
            "checkpoint or a custom model returned invalid logits."
        )

    # IQR-fence abstention on the FOV's max-softmax distribution. Abstention is
    # bucketed per FOV, and this API processes a single FOV, so the whole input
    # is one bucket and no grouping column is needed — `compute_iqr_fence`
    # returns None when n_cells < 4, in which case no cells are abstained.
    abstained = np.zeros(len(cell_types_raw), dtype=bool)
    cell_types = list(cell_types_raw)
    if ct_abstention_k is not None and ct_abstention_k > 0 and len(_top_probs) >= 4:
        fence = compute_iqr_fence(_top_probs, float(ct_abstention_k))
        if fence is not None:
            abstained = _top_probs < fence
            cell_types = [
                ABSTENTION_LABEL if was_abstained else ct
                for ct, was_abstained in zip(cell_types_raw, abstained)
            ]

    # Reinstate any cell that vanished when the mask was downscaled to the
    # model's target MPP (native mpp < 0.5 makes small cells disappear under
    # nearest-neighbor resampling). Without this, the default list[str] return is
    # shorter than the caller's cell set and a positional zip against
    # np.unique(mask) silently misaligns every downstream label. Dropped cells
    # get the "Unknown" sentinel and a zero-probability row; abstained stays
    # False for them (they were dropped, not abstained). No-op — and byte-for-
    # byte identical to prior behavior — when nothing was dropped (the common
    # case: native mpp >= the model target).
    missing_ids = np.setdiff1d(input_cell_ids, np.asarray(cell_indices, dtype=np.int64))
    if missing_ids.size:
        n_missing = int(missing_ids.size)
        warnings.warn(
            f"{n_missing} of {input_cell_ids.size} cells vanished when the mask was "
            f"resampled from mpp={mpp} to the model target "
            f"{dct_config.STANDARD_MPP_RESOLUTION} µm/px (small cells disappear under "
            f"nearest-neighbor downscaling). They are labeled {ABSTENTION_LABEL!r} "
            f"with a zero-probability row so the output stays aligned to the input "
            f"cells.",
            stacklevel=2,
        )
        n_classes = full_probs.shape[1]
        merged_ids = np.concatenate(
            [np.asarray(cell_indices, dtype=np.int64), missing_ids]
        )
        order = np.argsort(merged_ids, kind="stable")
        cell_indices = merged_ids[order]
        full_probs = np.concatenate(
            [full_probs, np.zeros((n_missing, n_classes), dtype=full_probs.dtype)]
        )[order]
        abstained = np.concatenate([abstained, np.zeros(n_missing, dtype=bool)])[order]
        cell_types = list(
            np.asarray(cell_types + [ABSTENTION_LABEL] * n_missing, dtype=object)[order]
        )
        cell_types_raw = list(
            np.asarray(
                list(cell_types_raw) + [ABSTENTION_LABEL] * n_missing, dtype=object
            )[order]
        )

    if return_probabilities:
        return PredictionResult(
            cell_types=cell_types,
            probabilities=full_probs,
            cell_indices=cell_indices,
            abstained=abstained,
            cell_types_raw=list(cell_types_raw),
        )
    return cell_types
