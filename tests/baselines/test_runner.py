"""Structure + CLI option-snapshot tests for the unified baselines runner.

These pin the public surface (which subcommands exist and their exact options)
without running any baseline. Option sets are frozen snapshots of the original
per-baseline click commands.
"""

import click

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


# Frozen option snapshots (verbatim from the original run.py click commands).
XGBOOST_OPTS = {
    "model_name",
    "enable_wandb",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "split_file",
    "features_cache",
    "min_channels",
}
NIMBUS_OPTS = {
    "model_name",
    "device_num",
    "enable_wandb",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "checkpoint",
    "batch_size",
    "test_time_aug",
    "threshold",
    "max_fovs",
}
XGBOOST_TUNE_OPTS = {
    "study_name",
    "n_trials",
    "metric",
    "enable_wandb",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "storage",
    "split_mode",
    "split_file",
    "min_channels",
    "max_tuning_samples",
    "device_num",
}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def test_registry_has_xgboost():
    assert "xgboost" in REGISTRY
    assert REGISTRY["xgboost"] == "deepcell_types.baselines.xgb.run:main"


def test_xgboost_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == XGBOOST_OPTS


def test_xgboost_tune_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost-tune")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == XGBOOST_TUNE_OPTS
