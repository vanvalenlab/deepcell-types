"""Registry + CLI option-snapshot tests for the round-2 baselines (maps, cellsighter).

Frozen option snapshots of the per-baseline click commands (re-frozen after the
wandb logging option was removed across all baselines). cellsighter's command
imports torchvision (via cellsighter.model), so its tests importorskip it.
"""

import click
import pytest

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


MAPS_OPTS = {
    "model_name",
    "device_num",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "split_file",
    "features_cache",
    "batch_size",
    "dropout",
    "hidden_dim",
    "learning_rate",
    "max_epochs",
    "seed",
}
# Re-frozen for the faithful CellSighter reimplementation (feat/faithful-cellsighter):
# added crop_size, mask_self, cifar_stem, test_split_file, allow_split_mismatch,
# and seed. These expose the paper-faithful training path (unmasked neighbor
# intensities, 60x60 crops, ImageNet stem) plus its self-masked/CIFAR ablations,
# a held-out test-split eval hook, and per-member seeding for ensembling.
CELLSIGHTER_OPTS = {
    "model_name",
    "device_num",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "split_file",
    "test_split_file",
    "split_mode",
    "batch_size",
    "epochs",
    "learning_rate",
    "model_size",
    "crop_size",
    "mask_self",
    "cifar_stem",
    "allow_split_mismatch",
    "seed",
    "no_amp",
    "no_compile",
    "pretrained",
    "val_every_n_epochs",
    # Ablation / faithfulness knobs added by the faithful-CellSighter work.
    "max_samples_per_epoch",
    "num_workers",
    "per_modality_norm",
    # Class-balancing scheme (faithful equal-proportion default + ablations).
    "class_balance",
    "size_data",
    "no_weighted_sampler",
}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def test_registry_has_maps():
    assert REGISTRY["maps"] == "deepcell_types.baselines.maps.run:main"


def test_maps_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "maps")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == MAPS_OPTS


def test_registry_has_cellsighter():
    assert REGISTRY["cellsighter"] == "deepcell_types.baselines.cellsighter.run:main"


def test_cellsighter_subcommand_options_frozen():
    pytest.importorskip("torchvision")
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "cellsighter")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == CELLSIGHTER_OPTS
