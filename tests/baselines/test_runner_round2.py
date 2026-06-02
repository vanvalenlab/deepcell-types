"""Registry + CLI option-snapshot tests for the round-2 baselines (maps, cellsighter).

Frozen option snapshots are verbatim from the original per-baseline click commands
(maps @ 85fa3229; cellsighter @ cebc391). cellsighter's command imports torchvision
(via cellsighter.model), so its tests importorskip it.
"""

import click
import pytest

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


MAPS_OPTS = {
    "model_name",
    "device_num",
    "enable_wandb",
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
CELLSIGHTER_OPTS = {
    "model_name",
    "device_num",
    "enable_wandb",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "split_file",
    "split_mode",
    "batch_size",
    "epochs",
    "learning_rate",
    "model_size",
    "no_amp",
    "no_compile",
    "pretrained",
    "val_every_n_epochs",
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
