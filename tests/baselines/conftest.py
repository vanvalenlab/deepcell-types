"""Skip baseline tests when their optional deps are absent (inference-only env)."""

import importlib


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
    except ImportError:
        return False
    return True


# filenames relative to this conftest's directory
collect_ignore = []
# test_runner.py exercises both xgb.run (module-level `import xgboost`) and
# xgb.tuning (module-level `import optuna`) via the runner's lazy get_command,
# so both must be importable or the tune option-snapshot test would error, not
# skip. (The baseline-xgboost extra always co-installs them.)
if not (_have("xgboost") and _have("optuna")):
    collect_ignore.append("test_runner.py")
# pandas proxies for the whole train stack here: scikit-learn (used by the
# metric reducer) is always co-installed with pandas via the [train] extra.
if not _have("pandas"):
    collect_ignore.append("test_nimbus_metrics_characterization.py")
if not _have("pandas"):
    collect_ignore.append("test_maps_cellsighter_cli.py")
# cellsighter.model imports torchvision AND (via deepcell_types.training.utils)
# pandas, so both must be present or this test would error, not skip. torchvision
# does not pull pandas, so guarding on torchvision alone is insufficient.
if not (_have("torchvision") and _have("pandas")):
    collect_ignore.append("test_cellsighter_convert_batch_characterization.py")
# maps.run does `import click` (the CLI) at module top, so importing
# normalize_features from it errors rather than skips on an inference-only
# install (no [train] / baseline-maps extra). Gate on click.
if not _have("click"):
    collect_ignore.append("test_maps_normalization.py")
