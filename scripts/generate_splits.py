"""Generate and save a canonical FOV split file for reproducible experiments.

This is **stage 1** of the two-stage canonical split. It stratifies all
labeled FOVs by (modality, tissue) and holds out FOVs proportionally within
each bucket at ``train_ratio`` (default 0.8), producing a ``train`` / ``val``
manifest. Stage 2 (``scripts/split_val_for_test.py``) then carves the held-out
``val`` FOVs into a model-selection validation subset and a frozen test set.

The canonical manifests used in the paper are committed under ``splits/`` and
do not need to be regenerated:

    splits/fov_split.json            -- stage 1 output (1722 train / 431 val)
    splits/fov_split_valsubset.json  -- stage 2: val=302 (model selection)
    splits/fov_split_test.json       -- stage 2: val=129 (frozen test set)

Usage (to reproduce stage 1 from a local archive):
    # Stratified (default — the canonical recipe):
    python -m scripts.generate_splits --output splits/fov_split.json \\
        --stratify_by modality,tissue
    # then run stage 2:
    python -m scripts.split_val_for_test \\
        --input splits/fov_split.json --output_prefix splits/fov_split

    # Unstratified global random shuffle (kept for benchmark continuity):
    python -m scripts.generate_splits --output splits/fov_split_unstratified.json \\
        --stratify_by ""

The default stratifies by (modality, tissue). Single-FOV strata are
forced to train (cannot evaluate held-out FOVs from a one-FOV bucket).
Empty `--stratify_by ""` reproduces the older global random shuffle.
"""

import os
import click
from pathlib import Path

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import FullImageDataset, save_fov_splits


DATA_DIR = Path(
    os.environ.get("DEEPCELL_TYPES_ZARR_PATH") or os.environ.get("DATA_DIR", "")
)


@click.command()
@click.option(
    "--zarr_dir",
    type=str,
    required=True,
    help="Path to the TissueNet zarr archive. Or set $DATA_DIR.",
    default=str(DATA_DIR / "tissuenet.zarr") if str(DATA_DIR) else None,
)
@click.option(
    "--output", type=str, required=True, help="Output path for the split JSON file"
)
@click.option("--seed", type=int, default=42)
@click.option("--train_ratio", type=float, default=0.8)
@click.option("--skip_datasets", type=str, multiple=True, default=[])
@click.option("--keep_datasets", type=str, multiple=True, default=[])
@click.option(
    "--stratify_by",
    type=str,
    default="modality,tissue",
    help=(
        "Comma-separated stratification keys. Empty string disables stratification "
        "(legacy global shuffle, used by v9 splits). Supported keys: modality, tissue."
    ),
)
def main(
    zarr_dir,
    output,
    seed,
    train_ratio,
    skip_datasets,
    keep_datasets,
    stratify_by,
):
    """Generate a canonical FOV split file."""
    dct_config = TissueNetConfig(zarr_dir)

    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None
    stratify_keys = tuple(k.strip() for k in stratify_by.split(",") if k.strip())

    dataset = FullImageDataset(
        zarr_dir,
        dct_config=dct_config,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
    )

    train_indices, val_indices = save_fov_splits(
        dataset,
        output,
        train_ratio=train_ratio,
        seed=seed,
        stratify_by=stratify_keys,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")


if __name__ == "__main__":
    main()
