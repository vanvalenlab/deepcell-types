import logging
import hashlib
import json
import os
import pickle
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _file_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _atomic_pickle_dump(obj, path: Path, *, protocol: int = 4) -> None:
    """Write a pickle cache by replacing the final path atomically."""
    tmp_path = None
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        pickle.dump(obj, tmp, protocol=protocol)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _atomic_np_savez(path: Path, **arrays) -> None:
    """Write an npz cache by replacing the final path atomically."""
    tmp_path = None
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def load_matching_state_dict(model, state_dict):
    """Load the entries of ``state_dict`` whose key exists in ``model`` with a
    matching tensor shape; return the number of tensors copied.

    Used to warm-start a model from a checkpoint that may carry extra or
    mismatched keys (e.g. a pretraining reconstruction head, optimizer /
    scheduler state, or an older architecture).
    """
    model_state = model.state_dict()
    loaded = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    return loaded


def _feature_cache_metadata(
    zarr_dir: str,
    dct_config,
    dataset_keys: list,
    split_file: str | None = None,
) -> dict:
    from deepcell_types.training.config import archive_array_fingerprint

    zarr_path = Path(zarr_dir).expanduser()
    try:
        zarr_path = zarr_path.resolve()
    except OSError:
        pass

    return {
        # v6 adds per-dataset ``present_markers`` so callers can distinguish
        # ``marker absent in this dataset`` from ``marker present but mean
        # intensity is 0.0`` and substitute their own missing-value sentinel
        # (e.g. NaN for XGBoost) without re-extracting from zarr.
        "cache_version": 6,
        "zarr_dir": str(zarr_path),
        "dataset_keys_hash": _stable_hash(sorted(dataset_keys)),
        "marker2idx_hash": _stable_hash(dct_config.marker2idx),
        "ct2idx_hash": _stable_hash(dct_config.ct2idx),
        "split_file_hash": _file_hash(split_file),
        "archive_fingerprint": archive_array_fingerprint(zarr_path, dataset_keys),
    }


def _cache_metadata_mismatches(saved: dict | None, expected: dict) -> list[str]:
    if not saved:
        return ["missing metadata"]
    return [key for key, value in expected.items() if saved.get(key) != value]


def _format_examples(values, limit: int = 5) -> str:
    examples = sorted(values)[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(str(v) for v in examples) + suffix


@dataclass
class BatchData:
    """Standardized batch format for all training/inference scripts.

    Fields (factored representation):
        sample: (B, C_max, 1, H, W) - raw intensity * self_mask per channel
        spatial_context: (B, 3, H, W) - [self_mask, neighbor_mask, distance_transform]
        ch_idx: (B, C_max) - channel indices
        mask: (B, C_max) - padding mask (True = padding)
        ct_idx: (B,) - cell type indices
        domain_idx: (B,) - domain (modality) indices
        marker_positivity: (B, C_max) - marker positivity labels
        marker_positivity_mask: (B, C_max) - mask for "?" labels (True = valid, compute loss)
        cell_index: (B,) - cell index in FOV
        dataset_name: tuple of str - dataset names
        fov_name: tuple of str - FOV names
        tissue_idx: (B,) - tissue indices (index 0 = ``__null__``); defaults to
            zeros so older datasets that don't ship a tissue lookup still load.
    """

    sample: torch.Tensor
    spatial_context: torch.Tensor
    ch_idx: torch.Tensor
    mask: torch.Tensor
    ct_idx: torch.Tensor
    domain_idx: torch.Tensor
    marker_positivity: torch.Tensor
    marker_positivity_mask: torch.Tensor
    cell_index: torch.Tensor
    dataset_name: Any
    fov_name: Any
    tissue_idx: Optional[torch.Tensor] = None

    def to(self, device):
        """Move all tensor fields to device, pass through non-tensor fields."""
        return BatchData(
            sample=self.sample.to(device),
            spatial_context=self.spatial_context.to(device),
            ch_idx=self.ch_idx.to(device),
            mask=self.mask.to(device),
            ct_idx=self.ct_idx.to(device),
            domain_idx=self.domain_idx.to(device),
            marker_positivity=self.marker_positivity.to(device),
            marker_positivity_mask=self.marker_positivity_mask.to(device),
            cell_index=self.cell_index.to(device),
            dataset_name=self.dataset_name,
            fov_name=self.fov_name,
            tissue_idx=self.tissue_idx.to(device)
            if self.tissue_idx is not None
            else None,
        )


class PredLogger:
    def __init__(self, ct2idx):
        self.ct2idx = ct2idx
        self.labels = []
        self.probs = []
        self.cell_index = []
        self.dataset_name = []
        self.fov_name = []

    def log(self, labels, probs, cell_index, dataset_name, fov_name):
        self.labels.append(labels)
        self.probs.append(probs)
        self.cell_index.append(cell_index)
        self.dataset_name.append(dataset_name)
        self.fov_name.append(fov_name)

    def to_dataframe(self):
        """Assemble the accumulated predictions into a DataFrame.

        Columns: one per cell-type class (softmax probability, ordered by
        ``ct2idx`` value), then ``cell_type_actual``, ``cell_index``,
        ``dataset_name``, ``fov_name``.
        """
        columns = sorted(self.ct2idx, key=self.ct2idx.get)
        idx2ct = {v: k for k, v in self.ct2idx.items()}
        labels = np.concatenate(self.labels)
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        dataset_name = np.concatenate(self.dataset_name)
        fov_name = np.concatenate(self.fov_name)
        df = pd.DataFrame(probs, columns=columns)
        df["cell_type_actual"] = [idx2ct[label] for label in labels]
        df["cell_index"] = cell_index
        df["dataset_name"] = dataset_name
        df["fov_name"] = fov_name
        return df

    @staticmethod
    def write_csv_atomic(df, path_name):
        """Atomically write ``df`` to ``path_name`` as CSV.

        A disk-full or SIGTERM mid-write would otherwise leave a truncated CSV
        that pandas reads silently, producing wrong abstention numbers in
        downstream analysis. Write to a sibling tempfile, fsync, then replace.
        """
        final_path = Path(path_name)
        tmp_path: Optional[Path] = None
        with tempfile.NamedTemporaryFile(
            "w",
            dir=final_path.parent,
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            df.to_csv(tmp, index=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            tmp_path.replace(final_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    def save(self, path_name):
        self.write_csv_atomic(self.to_dataframe(), path_name)


def log_epoch_metrics(epoch_metrics, prefix, wandb_run=None):
    """Log epoch-level metrics to wandb.

    Args:
        epoch_metrics: Dict of metric name -> value
        prefix: "train", "val", or "test"
        wandb_run: Optional wandb run object. If None, imports wandb and logs directly.
    """
    # Only network/IO failures from wandb should be swallowed. Logic errors
    # (AttributeError/KeyError/TypeError) must propagate — they indicate a
    # bug in the caller, not an expected runtime condition.
    try:
        import wandb
        import wandb.errors as _wandb_errors
    except ImportError as exc:
        logger.warning(
            "log_epoch_metrics: wandb import failed (prefix=%s): %s", prefix, exc
        )
        return

    for metric_name, metric_value in epoch_metrics.items():
        try:
            wandb.log({f"{prefix}/{metric_name}_epoch": metric_value})
        except (_wandb_errors.CommError, OSError) as exc:
            logger.warning(
                "log_epoch_metrics failed for prefix=%s metric=%s: %s",
                prefix,
                metric_name,
                exc,
            )


def log_confusion_matrix(
    metric, prefix, class_names, metric_name="confusion_matrix", tmp_dir="./tmp_images"
):
    """Log confusion matrix to wandb.

    Args:
        metric: torchmetrics confusion matrix metric
        prefix: "train", "val", or "test"
        class_names: List of class names for axis labels
        metric_name: Name for the wandb log entry
        tmp_dir: Directory for temporary image files
    """
    # Compute outside try/except so torchmetrics / numpy errors propagate loudly.
    conf_mat = metric.compute().cpu().numpy()
    conf_mat_norm = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-8)

    try:
        import wandb
        import wandb.errors as _wandb_errors
        import plotly.express as px
    except ImportError as exc:
        logger.warning(
            "log_confusion_matrix: required import failed (prefix=%s): %s",
            prefix,
            exc,
        )
        return

    side = 1500
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_path = tmp_dir / f"{metric_name}.png"
    norm_path = tmp_dir / f"{metric_name}_norm.png"

    # Only wandb network/IO and plotly image-writer errors are swallowed.
    # Logic errors (AttributeError, KeyError, TypeError) propagate.
    try:
        fig = px.imshow(
            conf_mat,
            x=class_names,
            y=class_names,
            labels=dict(x="Predicted", y="Actual"),
            width=side,
            height=side,
        )
        fig.write_image(base_path)
        wandb.log({metric_name: wandb.Image(str(base_path))})

        fig_norm = px.imshow(
            conf_mat_norm,
            x=class_names,
            y=class_names,
            labels=dict(x="Predicted", y="Actual"),
            width=side,
            height=side,
        )
        fig_norm.write_image(norm_path)
        wandb.log({f"{metric_name}_normalized": wandb.Image(str(norm_path))})
    except (_wandb_errors.CommError, OSError, RuntimeError) as exc:
        logger.warning(
            "log_confusion_matrix failed for prefix=%s metric=%s: %s",
            prefix,
            metric_name,
            exc,
        )


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Seed python, numpy, torch, and cuda RNGs for reproducibility.

    What this guarantees:
        - ``random``, ``numpy.random``, and ``torch`` (CPU + all CUDA devices)
          are seeded in the calling process.
        - cuDNN is placed in deterministic mode (``cudnn.deterministic=True``,
          ``cudnn.benchmark=False``).

    What this does NOT guarantee:
        - DataLoader worker reproducibility. Each worker process has its own
          ``random`` / ``numpy.random`` state that this function cannot reach.
          Pair a ``torch.Generator`` from :func:`make_generator` with
          :func:`worker_init_fn` on the DataLoader so worker-side augmentations
          are reproducible.
        - Bit-exact determinism. Many CUDA kernels (e.g. scatter_add, atomic
          ops inside transformer attention, some convolutions) are
          non-deterministic even with ``cudnn.deterministic=True``. Enabling
          full bit-determinism via ``torch.use_deterministic_algorithms(True)``
          is intentionally not done here: it costs ~15-25% throughput on this
          model without closing the non-determinism gap (augmentations are
          stochastic at the dataset level, so training is only reproducible
          when DataLoader workers are seeded via ``worker_init_fn``).

    Args:
        seed: Seed to use for all RNGs.
        deterministic: When True, also sets ``CUBLAS_WORKSPACE_CONFIG`` so that
            cuBLAS reductions are deterministic. Off by default because it
            slightly reduces throughput on cuBLAS-heavy layers (e.g. the
            transformer MLPs).
    """
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Required for deterministic cuBLAS on CUDA >= 10.2. See
        # https://pytorch.org/docs/stable/notes/randomness.html
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int):
    """DataLoader ``worker_init_fn`` that seeds RNGs inside each worker.

    PyTorch already derives a per-worker seed (``torch.initial_seed()``) from
    the DataLoader's ``generator``; this helper propagates that seed to the
    ``random`` and ``numpy.random`` module-level RNGs used by augmentations
    and dataset code. Without this, two runs with the same
    ``seed_everything(42)`` can differ by ~0.1-0.3pp macro accuracy because
    worker-side augmentation RNGs are not seeded.

    Usage::

        gen = make_generator(seed=42)
        loader = DataLoader(..., generator=gen, worker_init_fn=worker_init_fn)
    """
    import random

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded to ``seed``.

    Pair with :func:`worker_init_fn` on the DataLoader so that worker
    processes inherit a deterministic sub-seed from this generator::

        gen = make_generator(seed=42)
        loader = DataLoader(..., generator=gen, worker_init_fn=worker_init_fn)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


# Backward-compat re-exports: these were defined here pre-split.
# Canonical homes are now metrics.py and baseline_features.py.
from .metrics import (  # noqa: F401, E402
    adjust_conf_mat_hierarchy,
    summarize_mp_per_marker,
    MPMetricsTracker,
    LossesAndMetrics,
    build_label_remap,
)

# Lazy re-exports from baseline_features. A direct ``from .baseline_features
# import ...`` here would create a circular import when baseline_features is
# imported first (it imports private helpers from this module): utils.py
# would still be mid-load when baseline_features tries to come back through
# this line, and the names below wouldn't yet be defined. The module-level
# __getattr__ defers the lookup until the attribute is actually accessed,
# by which point both modules have finished initializing.
_BASELINE_FEATURES_REEXPORTS = {
    "_extract_all_dataset_features",
    "compute_baseline_metrics",
    "save_baseline_predictions",
    "extract_features_from_zarr",
}


def __getattr__(name):  # noqa: E402  (intentional module-level definition)
    if name in _BASELINE_FEATURES_REEXPORTS:
        from . import baseline_features

        return getattr(baseline_features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
