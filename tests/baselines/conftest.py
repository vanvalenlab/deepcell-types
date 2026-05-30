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
if not _have("xgboost"):
    collect_ignore.append("test_runner.py")
if not _have("pandas"):
    collect_ignore.append("test_nimbus_metrics_characterization.py")
