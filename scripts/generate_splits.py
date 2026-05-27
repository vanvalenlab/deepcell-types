"""Generate and save a canonical FOV split file for reproducible experiments.

Usage:
    # Stratified (default — recommended for new experiments):
    python -m scripts.generate_splits --output splits/fov_split.json \\
        --stratify_by modality,tissue

    # Unstratified global random shuffle (kept for benchmark continuity):
    python -m scripts.generate_splits --output splits/fov_split_unstratified.json \\
        --stratify_by ""

    # Custom seed / ratio:
    python -m scripts.generate_splits --output splits/custom.json \\
        --seed 42 --train_ratio 0.8

The default stratifies by (modality, tissue). Single-FOV strata are
forced to train (cannot evaluate held-out FOVs from a one-FOV bucket).
Empty `--stratify_by ""` reproduces the older global random shuffle.
"""

import os
import click
from pathlib import Path

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import FullImageDataset, save_fov_splits


DATA_DIR = Path(os.environ.get("DATA_DIR", ""))


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
    "--min_channels",
    type=int,
    default=0,
    help="Min model-visible marker channels per dataset (default 0 = no filter)",
)
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
    zarr_dir, output, seed, train_ratio, skip_datasets, keep_datasets, min_channels,
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
        min_channels=min_channels,
    )

    train_indices, val_indices = save_fov_splits(
        dataset, output, train_ratio=train_ratio, seed=seed,
        stratify_by=stratify_keys,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")


if __name__ == "__main__":
    main()
