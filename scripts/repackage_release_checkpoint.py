"""Back-fill a pre-bundling checkpoint so it is self-describing.

Older release checkpoints (e.g. the ``2026-06-15`` resMLP asset) were saved
with only ``{"model", "config"}`` and no bundled vocabulary. Current
``predict.py`` / ``scripts/predict.py`` run ``validate_checkpoint_vocabulary``,
which requires the checkpoint to carry its own ``ct2idx`` (and, when present,
``canonical_channels``) so a permuted vocabulary cannot silently mislabel every
cell. ``scripts/train.py`` and ``scripts/retrain_head.py`` already write these
keys for new checkpoints; this script back-fills them for a checkpoint that
predates the convention, without touching the model weights.

The vocabulary is sourced from :class:`~deepcell_types.config.DCTConfig` -- the
packaged canonical ``vocab.json`` by default, or an explicit ``--zarr_path``
archive. The checkpoint's ``config.n_celltypes`` must match the vocabulary
size, which guards against pairing a checkpoint with a mismatched vocabulary
(the count check that ``load_state_dict`` would otherwise only catch at load
time). Re-packaging changes the file's checksum, so the ``_model_registry``
entry in ``deepcell_types/utils/__init__.py`` must be updated to the new md5
and the new asset re-uploaded before ``download_model`` will serve it.

Usage:
    python -m scripts.repackage_release_checkpoint \\
        deepcell-types_2026-06-15_resmlp.pt \\
        -o deepcell-types_2026-06-15_resmlp_bundled.pt
"""

import argparse
import hashlib
from pathlib import Path

import torch

from deepcell_types.config import DCTConfig


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def bundle_vocabulary(ckpt, cfg):
    """Back-fill ``ct2idx`` + ``canonical_channels`` into a checkpoint dict.

    Mutates and returns ``ckpt``. ``cfg`` is a :class:`DCTConfig` supplying the
    vocabulary. Raises ``ValueError`` if ``ckpt`` is not a checkpoint dict, if
    it already bundles ``ct2idx``, or if its ``config.n_celltypes`` does not
    match the vocabulary size (a mismatched pairing would mislabel cells).
    """
    if not (isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt):
        raise ValueError("Expected a checkpoint dict with 'model' and 'config' keys.")
    if "ct2idx" in ckpt:
        raise ValueError("Checkpoint already bundles ct2idx; nothing to do.")
    n_ckpt = ckpt["config"].get("n_celltypes")
    if n_ckpt != len(cfg.ct2idx):
        raise ValueError(
            f"config.n_celltypes ({n_ckpt}) != vocabulary size "
            f"({len(cfg.ct2idx)}); refusing to bundle a mismatched vocabulary."
        )
    ckpt["ct2idx"] = dict(cfg.ct2idx)
    ckpt["canonical_channels"] = list(cfg.marker2idx.keys())
    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the input checkpoint (.pt) to re-package.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to write the re-packaged checkpoint.",
    )
    parser.add_argument(
        "--zarr_path",
        default=None,
        help=(
            "Optional archive to source the vocabulary from. Defaults to the "
            "packaged vocab.json (correct for canonically-trained checkpoints)."
        ),
    )
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(
            "Checkpoint could not be loaded in restricted weights-only mode. "
            "Do not repackage an untrusted legacy pickle checkpoint; convert it "
            "in an isolated environment first."
        ) from exc
    cfg = DCTConfig(zarr_path=args.zarr_path)
    try:
        bundle_vocabulary(ckpt, cfg)
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)

    print(f"in : {args.checkpoint}  md5={_md5(args.checkpoint)}")
    print(f"out: {args.output}  md5={_md5(args.output)}")
    print(
        f"bundled {len(ckpt['ct2idx'])} cell types, "
        f"{len(ckpt['canonical_channels'])} markers"
    )


if __name__ == "__main__":
    main()
