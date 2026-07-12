# DeepCell Types

DeepCell Types is a generalized cell-phenotyping model for spatial
proteomics. It addresses generalization across datasets with varying
marker panels by matching each image's channels against a marker /
cell-type registry, which ships with the package as a `vocab.json`
snapshot — so inference runs on any in-memory image with no extra data
download.

> **License notice.** This project is distributed under a *Modified Apache
> License, Version 2.0* with non-commercial / academic-only carve-outs (see
> the [LICENSE](LICENSE) file for the full text). For any other use,
> including commercial use, contact `vanvalenlab@gmail.com`.

## Installation

As with all Python packages, users are encouraged to use some form
of virtual environment for package installation.
Popular options include `venv`/`virtualenv`, `conda`/`mamba`, `uv`,
or `pixi`.
Users are encouraged to use whatever environment management toolchain
they are most comfortable with.
For those unsure, the quickest way to start is to use the `venv` module,
part of the Python standard library:

```bash
# Create a new virtual environment
python -m venv dct-env
# Enter the virtual environment
source dct-env/bin/activate

# Once inside the environment, install deepcell-types
pip install git+https://github.com/vanvalenlab/deepcell-types@master
```

## Download the model

Downloading the checkpoint requires a free access token; register at
[`https://users.deepcell.org`](https://users.deepcell.org) and set it in your
environment (see [`docs/site/API-key.md`](docs/site/API-key.md)):

```bash
export DEEPCELL_ACCESS_TOKEN=<your token>
```

```python
from deepcell_types.utils import download_model

# Downloads the latest checkpoint into ~/.deepcell/models and returns its path.
model_path = download_model()
```

## Running inference (no archive required)

Inference needs only the model checkpoint (`download_model()` above) and
your image as an in-memory array — **no TissueNet archive download is
required**. The marker / cell-type registry ships inside the package as a
`vocab.json` snapshot, and `predict` resolves your channels against it
automatically:

```python
import numpy as np
import torch

from deepcell_types import PredictionResult, predict
from deepcell_types.utils import download_model

# Pick a device: the default GPU if one is available, otherwise the CPU
# (inference works the same on CPU, just slower). Use "cuda:1", "cuda:2",
# etc. to target a specific GPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Minimal two-cell, two-marker example. Replace these arrays with your image
# (C, H, W), integer instance mask (H, W), marker names, and image resolution.
raw = np.zeros((2, 64, 64), dtype=np.float32)
raw[0, 8:28, 8:28] = 100.0
raw[1, 36:56, 36:56] = 100.0
mask = np.zeros((64, 64), dtype=np.uint32)
mask[8:28, 8:28] = 1
mask[36:56, 36:56] = 2
channel_names = ["CD3", "CD20"]
mpp = 0.5
model_path = download_model()

# Pass the path returned by download_model() straight through to predict();
# predict() also accepts a filesystem path to a .pt file directly.
result = predict(
    raw,
    mask,
    channel_names,
    mpp,
    model_name=model_path,
    device=device,
    return_probabilities=True,
)
assert isinstance(result, PredictionResult)
print(dict(zip(result.cell_indices.tolist(), result.cell_types)))
```

For a complete example of the cell-type inference pipeline, check out
the [tutorial](https://vanvalenlab.github.io/deepcell-types/site/tutorial.html).

## TissueNet zarr archive (optional)

The archive is **only** needed for training or to supply a registry that exactly
matches a compatible checkpoint. Input panels may use any subset of the
packaged markers, and acquisition-name aliases can be added to
`channel_mapping.yaml`; neither case requires an archive. Adding, removing, or
reordering model markers changes the model vocabulary and requires new marker
embeddings plus a checkpoint trained for that exact registry. When an archive
is present, `predict` reads its registry instead of `vocab.json`; pass
`zarr_path=...` directly or set `DEEPCELL_TYPES_ZARR_PATH`.

A registered user can download a public TissueNet zarr archive from
`https://users.deepcell.org`; see `docs/site/API-key.md` for the access
token flow. Place the resulting `.zarr` directory anywhere, then:

```bash
export DEEPCELL_TYPES_ZARR_PATH=/absolute/path/to/tissuenet.zarr
```

### Validating an archive before publishing

The archive's `all_standardized_channels` attribute *is* the model's
marker→index map, so it must match the order the released checkpoint and
`embeddings/svd_512.npz` were built with — reordering or resizing it silently
breaks inference. Before publishing an archive (or a checkpoint), run the
release gate against the real archive and the released embeddings:

```bash
scripts/check_release_archive.sh /path/to/tissuenet.zarr /path/to/svd_512.npz
```

It exits non-zero on any marker-order/size drift. (The check's logic is
unit-tested in CI via `tests/test_archive_contract_validator.py`; this script
runs it against the actual archive, which CI cannot access.)

## Custom preprocessing (advanced)

When a single FOV's predictions look biologically implausible — usually because
one channel is saturated or has heavy background and is steering the calls — the
fix is to adapt the per-channel normalization for that FOV.

Use `predict`'s optional `preprocess` hook to apply a declarative, reviewable
sequence of bounded operations:

```python
import torch

from deepcell_types import predict, make_preprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"

config = [
    {"op": "clip_percentile", "p": 99.9},
    {"op": "channel_drop", "names": ["NeuN"]},  # drop a confounding marker
    {"op": "min_max_normalize"},                # model sees [0, 1]
]
labels = predict(raw, mask, channel_names, mpp, model_name=model_path,
                 device=device, preprocess=make_preprocessor(config))
```

The hook receives the resampled, in-vocabulary raw `(C, H, W)` array and the
resolved marker names, and must return a `(C, H, W)` array in `[0, 1]`. With
`preprocess=None` (default) the built-in p99.9 clip + min-max is used;
`make_preprocessor(DEFAULT_CONFIG)` reproduces that default exactly.

Repository contributors who use a compatible coding agent may optionally use
the [`preproc-adapt` skill](skills/preproc-adapt/) to inspect prediction results
and iterate on this configuration. The Python API above remains the supported
and tool-independent interface.

## Training

Training and archive evaluation are source-checkout workflows; their scripts
are not installed by the wheel. Clone the repository and install its training
environment from the repository root:

```bash
git clone https://github.com/vanvalenlab/deepcell-types.git
cd deepcell-types
uv sync --extra train
```

Training entry points live under `scripts/`:

- `scripts/train.py` — main training loop (stage 1: backbone, weighted sampler on).
- `scripts/retrain_head.py` — stage 2: freeze the backbone, retrain the residual-MLP
  cell-type head on the natural class distribution (sampler off). This decoupled
  recipe is the default and produces the best model; the residual-MLP head is
  auto-detected from the checkpoint at inference (no flag to set).
- `scripts/pretrain.py` — masked-marker pretraining.
- `scripts/predict.py` — batched evaluation over a zarr archive.
- `scripts/evaluate_on_test.sh` — evaluation on the held-out 129-FOV test split
  (`splits/fov_split_test_current.json`). See the cited paper for reported
  results; this repository does not currently include the prediction artifacts
  needed to independently verify its headline values.

All training scripts read configuration from a TissueNet zarr v3 archive.
Pass `--zarr_dir` (training scripts) or set `DEEPCELL_TYPES_ZARR_PATH`
(picked up by `deepcell_types.predict`). The training-side
modules under `deepcell_types.training` (e.g. `TissueNetConfig`,
`FullImageDataset`, `FocalLoss`, `HierarchicalLoss`) are stable enough to
import directly for custom training scripts.

From the repository root, a minimal smoke run and the two-stage training recipe
look like this (replace the archive and embedding paths with downloaded assets):

```bash
uv run python scripts/pretrain.py \
  --zarr_dir /data/tissuenet.zarr \
  --svd_embeddings_path /data/svd_512.npz \
  --split_file splits/fov_split.json \
  --model_name pretrain

uv run python scripts/train.py \
  --zarr_dir /data/tissuenet.zarr \
  --svd_embeddings_path /data/svd_512.npz \
  --split_file splits/fov_split.json \
  --model_name smoke --epochs 1 --max_samples_per_epoch 256 \
  --max_val_samples 128 --batch_size 16 --num_workers 0

uv run python scripts/train.py \
  --zarr_dir /data/tissuenet.zarr \
  --svd_embeddings_path /data/svd_512.npz \
  --split_file splits/fov_split.json \
  --model_name stage1

uv run python scripts/retrain_head.py \
  --zarr_dir /data/tissuenet.zarr \
  --svd_embeddings_path /data/svd_512.npz \
  --split_file splits/fov_split.json \
  --pretrained_path models/model_stage1_best.pt \
  --output models/model_stage2_resmlp_best.pt
```

The scripts write checkpoints under `models/` and validation predictions under
`output/`. Full training is GPU-oriented; use `--device_num cpu` only for small
smoke runs. Run any script with `--help` for resource-control options.

## Baselines

All four paper comparison baselines are folded into `deepcell_types.baselines`
and run via the unified runner `python -m deepcell_types.baselines <name>`.
No submodules are required.

> **Reproducing the paper comparison (fairness contract).** For an
> apples-to-apples comparison, run every baseline *and* the main model with the
> same class-balancing sampler and the same checkpoint-selection validation
> split: `--class_balance dct` (the default) and
> `--val_split_file splits/fov_split_valsubset.json` (the shared canonical
> model-selection split, seed 42), then evaluate on the frozen test split
> `splits/fov_split_test_current.json`. These are the settings behind the
> reported numbers. Each baseline's own native sampler / self-carved val split
> stays available (via `--class_balance`) for the appendix but is not the
> headline comparison. Note these flags are opt-in, not defaults, so parity
> depends on passing them consistently to every method.

- **XGBoost** — XGBoost on mean-marker-intensity features.

  ```bash
  uv sync --extra baseline-xgboost
  uv run python -m deepcell_types.baselines xgboost \
    --zarr_dir /data/tissuenet.zarr \
    --split_file splits/fov_split_test_current.json \
    --val_split_file splits/fov_split_valsubset.json \
    --features_cache output/xgboost_features.npz
  uv run python -m deepcell_types.baselines xgboost-tune \
    --zarr_dir /data/tissuenet.zarr \
    --split_file splits/fov_split_test_current.json \
    --val_split_file splits/fov_split_valsubset.json \
    --features_cache output/xgboost_features.npz \
    --storage sqlite:///output/xgboost_tuning.db --n_trials 50
  ```

- **Nimbus** — Nimbus UNet marker-positivity baseline
  ([Rumberger et al., *Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02826-9)).

  ```bash
  uv sync --extra baseline-nimbus
  uv run python -m deepcell_types.baselines nimbus \
    --zarr_dir /data/tissuenet.zarr \
    --checkpoint latest --batch_size 8
  ```

  > **Note:** `baseline-nimbus` pins `nimbus-inference==0.0.5`, which
  > requires Python <3.12. Use a Python 3.11 environment for this baseline.

- **MAPS** — MAPS MLP classifier
  ([*Nature Communications* 2023](https://doi.org/10.1038/s41467-023-44188-w)).

  ```bash
  uv sync --extra baseline-maps
  uv run python -m deepcell_types.baselines maps \
    --zarr_dir /data/tissuenet.zarr \
    --split_file splits/fov_split_test_current.json \
    --val_split_file splits/fov_split_valsubset.json \
    --features_cache output/maps_features.npz
  ```

- **CellSighter** — ResNet-50 multiplexed cell classifier
  ([Amitay et al., *Nature Communications* 2023](https://doi.org/10.1038/s41467-023-40066-7)).
  Pulls in `torchvision`.

  ```bash
  uv sync --extra baseline-cellsighter
  uv run python -m deepcell_types.baselines cellsighter \
    --zarr_dir /data/tissuenet.zarr \
    --split_file splits/fov_split.json \
    --val_split_file splits/fov_split_valsubset.json \
    --test_split_file splits/fov_split_test_current.json
  ```

## Citation
```
@article{deepcelltypes,
  title={Generalized cell phenotyping for spatial proteomics with language-informed vision models},
  author={Wang, Xuefei and Dilip, Rohit and Iqbal, Ahamed Raffey and Bussi, Yuval and Brown, Caitlin and Pradhan, Elora and Jain, Yashvardhan and Yu, Kevin and Li, Shenyi and Abt, Martin and Borner, Katy and Keren, Leeat and Yue, Yisong and Barnowski, Ross and Van Valen, David},
  journal={bioRxiv},
  pages={2026--07},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  url={https://www.biorxiv.org/content/10.1101/2024.11.02.621624v4}
}
```
