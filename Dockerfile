# deepcell-types container image.
#
# A single Dockerfile covers every install profile via the DCT_EXTRAS build arg,
# which is appended to the editable install of this package:
#
#   # inference-only (default) — matches `pip install deepcell-types`
#   docker build -t deepcell-types .
#
#   # full training pipeline (deepcell_types.training + scripts/)
#   docker build --build-arg DCT_EXTRAS="[train]" -t deepcell-types:train .
#
#   # all four comparison baselines (xgboost / nimbus / maps / cellsighter)
#   docker build --build-arg DCT_EXTRAS="[baselines]" -t deepcell-types:baselines .
#
#   # everything (train + baselines)
#   docker build --build-arg DCT_EXTRAS="[all]" -t deepcell-types:all .
#
# The optional-dependency names mirror [project.optional-dependencies] in
# pyproject.toml. Leave DCT_EXTRAS empty for the inference-only image.

# Base image must ship Python >=3.11 to satisfy this package's
# `requires-python = ">=3.11"` floor (pyproject.toml). The previous
# `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` tag ships Python 3.10.14, so
# `pip install .` failed for EVERY profile with
# "requires a different Python: 3.10.14 not in '>=3.11'". The 2.4.1 runtime tag
# below bundles Python 3.11, clearing the floor (and bumps cuDNN 8 -> 9).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Which optional-dependency set to install (e.g. "[train]", "[baselines]",
# "[all]"); empty means inference-only.
ARG DCT_EXTRAS=""

RUN python -m pip install --upgrade pip \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install the package itself (plus any requested extras) from pyproject.toml.
# The legacy requirements.txt / deepcelltypes-kit/ install steps are gone —
# dependencies are declared in pyproject.toml now.
RUN python -m pip install --no-cache-dir ".${DCT_EXTRAS}"
