"""Registry and CLI option-snapshot tests for MAPS and CellSighter.

CellSighter's command imports torchvision through its model, so those tests
skip when the optional dependency is unavailable.
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
    # Canonical external-val selection (feat/baselines-canonical-val-selection).
    "val_split_file",
    "batch_size",
    "dropout",
    "hidden_dim",
    "learning_rate",
    "max_epochs",
    "min_epochs",
    "patience",
    "seed",
    "znorm",
    # Class-balancing scheme (shared DCT sampler default + faithful ablation).
    "class_balance",
}
MAPS_DEFAULTS = {
    "max_epochs": 500,
    "min_epochs": 250,
    "patience": 100,
    "znorm": True,
    # Unified on the shared DCT sampler by default.
    "class_balance": "dct",
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
    # Canonical external-val selection (feat/baselines-canonical-val-selection).
    "val_split_file",
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
# CellSighter defaults to the shared DCT sampler ('sqrt'); faithful
# equal-proportion is the '--class_balance equal' ablation.
CELLSIGHTER_DEFAULTS = {"class_balance": "sqrt"}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def _param_defaults(cmd):
    return {p.name: p.default for p in cmd.params}


def test_registry_has_maps():
    assert REGISTRY["maps"] == "deepcell_types.baselines.maps.run:main"


def test_maps_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "maps")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == MAPS_OPTS
    defaults = _param_defaults(cmd)
    for name, expected in MAPS_DEFAULTS.items():
        assert defaults[name] == expected


def test_registry_has_cellsighter():
    assert REGISTRY["cellsighter"] == "deepcell_types.baselines.cellsighter.run:main"


def test_cellsighter_subcommand_options_frozen():
    pytest.importorskip("torchvision")
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "cellsighter")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == CELLSIGHTER_OPTS
    defaults = _param_defaults(cmd)
    for name, expected in CELLSIGHTER_DEFAULTS.items():
        assert defaults[name] == expected
