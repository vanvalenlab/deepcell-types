"""Guard that the inference path stays free of training-only dependencies.

If a future change adds ``import wandb`` (or zarr, sklearn, pandas, etc.)
anywhere in the ``deepcell_types.predict`` import graph, this test fails
loudly. The split is the whole point of the ``[train]`` extra: a user
who only wants to load a checkpoint and run inference should be able to
``pip install deepcell-types`` and have no heavy ML-ops deps pulled in.

Runs as a subprocess so the test environment's own imports
(pytest, etc.) don't pollute ``sys.modules`` and produce false negatives.
"""

import subprocess
import sys
import textwrap


# Modules that belong only to the training pipeline. If any of these ends
# up in ``sys.modules`` after importing the inference entry points, the
# split has been broken — find the leaking import and move it under
# ``deepcell_types.training`` or behind a function-local import.
TRAINING_ONLY_MODULES = (
    "wandb",
    "zarr",
    "sklearn",
    "pandas",
    "torchvision",
    "torchinfo",
    "torchmetrics",
    "matplotlib",
)


def test_predict_import_does_not_pull_training_deps():
    probe = textwrap.dedent(
        f"""
        import sys
        import deepcell_types  # noqa: F401
        from deepcell_types import predict  # noqa: F401
        from deepcell_types.predict import predict, PredLogger  # noqa: F401
        from deepcell_types.dct_kit.config import DCTConfig  # noqa: F401
        from deepcell_types.dataset import PatchDataset  # noqa: F401
        from deepcell_types.model import CellTypeAnnotator, create_model  # noqa: F401

        leaked = sorted(m for m in {TRAINING_ONLY_MODULES!r} if m in sys.modules)
        if leaked:
            raise SystemExit(
                "training-only dependencies leaked into inference path: "
                + ", ".join(leaked)
            )
        """
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
