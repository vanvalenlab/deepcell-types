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
# pandas proxies for the whole train stack here: scikit-learn (used by the
# metric reducer) is always co-installed with pandas via the [train] extra.
if not _have("pandas"):
    collect_ignore.append("test_nimbus_metrics_characterization.py")
