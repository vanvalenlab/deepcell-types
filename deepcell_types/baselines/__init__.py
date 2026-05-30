"""deepcell-types comparison baselines.

Each baseline is a self-contained click command, registered here by name and
imported lazily by the runner (``python -m deepcell_types.baselines``) so that a
baseline's optional dependencies are only required when that subcommand runs.
"""

# subcommand name -> "module_path:click_command_attr"
REGISTRY = {
    "xgboost": "deepcell_types.baselines.xgb.run:main",
    "xgboost-tune": "deepcell_types.baselines.xgb.tuning:main",
    "nimbus": "deepcell_types.baselines.nimbus.run:main",
}
