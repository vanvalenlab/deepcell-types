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

# The base image supplies Python 3.11, matching the package's supported floor.
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
# Dependencies are declared in pyproject.toml.
RUN python -m pip install --no-cache-dir ".${DCT_EXTRAS}"
