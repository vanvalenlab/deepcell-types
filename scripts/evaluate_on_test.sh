#!/usr/bin/env bash
# Canonical evaluation: the held-out 129-FOV TEST split.
#
# This is the repo's default/headline evaluation. All reported cell-type numbers
# are on this frozen, leakage-free test set (no method uses it for selection).
#
#   splits/fov_split_test_current.json  -- current-archive (f5b6ed52) test split:
#       current train + the frozen 129 test FOVs as `val`. The 129 test FOVs are
#       bit-identical to the prior-archive splits/fov_split_test.json (verified:
#       486,705 cells, 100% ground-truth match), so results compare directly to
#       the baseline test CSVs in dct-final-ckpt/.
#
# Usage:
#   DATA_DIR=/path/to/archive bash scripts/evaluate_on_test.sh [MODEL_CKPT] [EMB]
#
# Defaults to the NEW BEST resMLP-head checkpoint in the download_model() cache
# (~/.deepcell/models). predict.py auto-detects the residual-MLP head (config
# ct_head_arch="resmlp") and prints hierarchical macro/weighted F1 on the test
# split. EMB (the SVD marker embeddings) is OPTIONAL: the checkpoint already
# carries the marker embeddings, so predict.py builds a placeholder when it is
# unset — pass a path only to override with a specific embeddings file.
set -euo pipefail

MODEL_CKPT="${1:-$HOME/.deepcell/models/deepcell-types_2026-06-15_resmlp.pt}"
EMB="${2:-}"
SPLIT="${SPLIT:-splits/fov_split_test_current.json}"
: "${DATA_DIR:?set DATA_DIR to the zarr archive}"

echo "Evaluating $MODEL_CKPT on the held-out test split ($SPLIT)"
# Only pass --svd_embeddings_path when EMB is set (empty-array expansion is
# written to stay safe under `set -u` on older bash, e.g. macOS bash 3.2).
svd_args=()
[ -n "$EMB" ] && svd_args=(--svd_embeddings_path "$EMB")
python scripts/predict.py \
  --model_name eval_test \
  --model_path "$MODEL_CKPT" \
  ${svd_args[@]+"${svd_args[@]}"} \
  --split_file "$SPLIT" \
  --ct_abstention_k 0

# For a full multi-method head-to-head table (ours + XGBoost/MAPS/CellSighter),
# generate each baseline's predictions on this same split (see
# deepcell_types.baselines) and compare the per-cell prediction CSVs with your
# own scoring. All methods evaluate on the identical 129-FOV test split.
