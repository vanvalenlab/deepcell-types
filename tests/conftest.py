"""Pytest configuration for the deepcell-types test suite.

Several test modules exercise the training pipeline and so need the ``[train]``
install extra (``zarr``, ``pandas``, ``scikit-learn``, ``torchmetrics``, ...).
On an inference-only checkout (``pip install -e .``) those packages are absent,
and the affected tests must be *skipped*, not raise collection errors.

Rather than hand-maintaining a list of which test file needs which extra (which
silently rots whenever a new train-dependent test file is added, and bit us
when a missing entry turned the inference-only CI job red), we autodetect: for
each ``test_*.py`` we parse its top-level imports and try to import them. If one
fails because an *optional* (extra-only) package is missing -- directly or
transitively, e.g. ``from deepcell_types.training.dataset import ...`` pulling
``zarr`` -- the file is excluded from collection. A missing *core* dependency,
or any other import error, is left to surface as a normal collection error so
real breakage stays visible. The only list that must be maintained is the set
of optional package names below, which changes when a new optional *dependency*
is added -- not when a new test file is added.

We import each test file's *dependencies*, never the test module itself, so
pytest's assertion rewriting is untouched.

Caveat: a test whose use of a train dependency is hidden from static import
analysis -- imported inside a subprocess string, or lazily inside a function --
cannot be autodetected. Guard those at the top of the module with
``pytest.importorskip("<pkg>")`` instead (see test_training_import_order.py).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

# Import names provided ONLY by optional extras ([train] / [baselines]
# in pyproject.toml), never by the core inference install. A test
# file that fails to import because one of these is missing gets skipped; a
# missing core dependency is deliberately absent here, so it still errors loudly.
_OPTIONAL_PACKAGES = frozenset(
    {
        # [train]
        "zarr",
        "torchinfo",
        "torchmetrics",
        "pandas",
        "sklearn",  # scikit-learn
        "click",
        "plotly",
        "kaleido",
        "tifffile",
        "openai",
        # [baselines]
        "xgboost",
        "optuna",
        "torchvision",
        "nimbus_inference",  # nimbus-inference
        "cv2",  # opencv-python-headless
    }
)


def _top_level_imported_modules(path: Path):
    """Yield the module names imported at the top level of ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _missing_optional_dep(path: Path) -> str | None:
    """Return the optional package whose absence makes ``path`` unimportable, or
    ``None`` if every top-level import resolves. Import failures for non-optional
    modules are ignored here so pytest's own collection surfaces them."""
    for module in _top_level_imported_modules(path):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root in _OPTIONAL_PACKAGES:
                return root
        except ImportError:
            # A non-"module not found" import error (e.g. a real circular
            # import): not a missing-extra signal, so let collection report it.
            pass
    return None


_here = Path(__file__).parent
collect_ignore = [
    path.name
    for path in sorted(_here.glob("test_*.py"))
    if _missing_optional_dep(path) is not None
]
