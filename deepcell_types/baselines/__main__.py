"""Unified runner for deepcell-types baselines.

Usage:
    python -m deepcell_types.baselines <baseline> [options]
    python -m deepcell_types.baselines xgboost --split_file ... --zarr_dir ...
    python -m deepcell_types.baselines nimbus --zarr_dir ...
"""

import importlib

import click

from deepcell_types.baselines import REGISTRY


class LazyGroup(click.Group):
    """Click group that imports a baseline's module only when invoked.

    Returning the original click command object preserves every option and
    default verbatim, while deferring the import keeps per-baseline optional
    dependencies (xgboost, nimbus-inference) out of the import path unless that
    subcommand actually runs.
    """

    def list_commands(self, ctx):
        return sorted(REGISTRY)

    def get_command(self, ctx, name):
        target = REGISTRY.get(name)
        if target is None:
            return None
        module_path, attr = target.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, attr)


@click.command(cls=LazyGroup)
def cli():
    """Run deepcell-types comparison baselines."""


if __name__ == "__main__":
    cli()
