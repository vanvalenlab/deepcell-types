"""Structure + CLI option-snapshot tests for the unified baselines runner.

These pin the public surface (which subcommands exist and their exact options)
without running any baseline. Option sets are frozen snapshots of the
per-baseline click commands (re-frozen after the wandb logging option was
removed across all baselines).
"""

import click

from deepcell_types.baselines import REGISTRY
from deepcell_types.baselines.__main__ import cli


# Frozen option snapshots of the run.py click commands.
XGBOOST_OPTS = {
    "model_name",
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "split_file",
    "features_cache",
    # Class-balancing scheme (shared DCT sampler default + faithful ablation).
    "class_balance",
}
NIMBUS_OPTS = {
    "model_name",
    "device_num",
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
    "zarr_dir",
    "skip_datasets",
    "keep_datasets",
    "storage",
    "split_mode",
    "split_file",
    "max_tuning_samples",
    "device_num",
    # Class-balancing scheme (shared DCT sampler default + faithful ablation).
    "class_balance",
}
# Lock the unified-sampler defaults so a silent flip is caught.
XGBOOST_DEFAULTS = {"class_balance": "dct"}
XGBOOST_TUNE_DEFAULTS = {"class_balance": "dct"}


def _param_names(cmd):
    return {p.name for p in cmd.params}


def _param_defaults(cmd):
    return {p.name: p.default for p in cmd.params}


def test_registry_has_xgboost():
    assert "xgboost" in REGISTRY
    assert REGISTRY["xgboost"] == "deepcell_types.baselines.xgb.run:main"


def test_xgboost_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == XGBOOST_OPTS
    defaults = _param_defaults(cmd)
    for name, expected in XGBOOST_DEFAULTS.items():
        assert defaults[name] == expected


def test_xgboost_tune_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "xgboost-tune")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == XGBOOST_TUNE_OPTS
    defaults = _param_defaults(cmd)
    for name, expected in XGBOOST_TUNE_DEFAULTS.items():
        assert defaults[name] == expected


def test_registry_has_nimbus():
    assert REGISTRY["nimbus"] == "deepcell_types.baselines.nimbus.run:main"


def test_nimbus_subcommand_options_frozen():
    ctx = click.Context(cli)
    cmd = cli.get_command(ctx, "nimbus")
    assert isinstance(cmd, click.Command)
    assert _param_names(cmd) == NIMBUS_OPTS
