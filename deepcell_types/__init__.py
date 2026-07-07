# Copyright 2024-2026 The Van Valen Lab at the California Institute of
# Technology (Caltech).  All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/vanvalenlab/deepcell-types/blob/master/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.

from importlib.metadata import PackageNotFoundError, version

# Importing deepcell_types must not pull in torch (a ~1.4s import). None of the
# modules below import torch at module scope — in particular .predict defers its
# torch / model / dataset imports into the functions that need them — so
# `import deepcell_types` stays torch-free until predict() is actually called.
from .abstention import ABSTENTION_LABEL as ABSTENTION_LABEL
from .config import DCTConfig as DCTConfig
from .predict import predict as predict
from .predict import PredictionResult as PredictionResult
from .preprocessing import preprocess_fov as preprocess_fov
from .preprocessing import PreprocessedFov as PreprocessedFov
from .preprocessing_ops import apply_config as apply_config
from .preprocessing_ops import make_preprocessor as make_preprocessor
from .preprocessing_ops import DEFAULT_CONFIG as DEFAULT_CONFIG
from .utils import download_baseline_checkpoint as download_baseline_checkpoint
from .utils import download_model as download_model
from .utils import download_training_data as download_training_data
from .utils import list_baseline_names as list_baseline_names
from .utils import list_model_versions as list_model_versions
from .utils import list_supported_cell_types as list_supported_cell_types
from .utils import list_supported_markers as list_supported_markers

try:
    __version__ = version("deepcell-types")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "predict",
    "PredictionResult",
    "DCTConfig",
    "preprocess_fov",
    "PreprocessedFov",
    "apply_config",
    "make_preprocessor",
    "DEFAULT_CONFIG",
    "download_model",
    "download_baseline_checkpoint",
    "download_training_data",
    "list_model_versions",
    "list_baseline_names",
    "list_supported_markers",
    "list_supported_cell_types",
    "ABSTENTION_LABEL",
    "__version__",
]
