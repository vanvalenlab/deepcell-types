"""Training-side modules for deepcell-types.

This subpackage holds the training pipeline — losses, the zarr-backed
dataset, annotation extraction, training utilities, etc. None of its
contents are imported by the inference path (``deepcell_types.predict``);
importing it requires the ``[train]`` extra:

    pip install deepcell-types[train]

Inference users do not need any of this. The split is enforced by a CI
guard (``tests/test_inference_deps.py``) that fails if the inference path
ever transitively imports a training-only dependency (wandb, zarr,
scikit-learn, pandas, ...).
"""
