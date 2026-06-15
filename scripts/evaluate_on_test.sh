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
# Defaults to the NEW BEST resMLP-head checkpoint. predict.py auto-detects the
# residual-MLP head (config ct_head_arch="resmlp") and prints hierarchical macro/
# weighted F1 on the test split.
set -euo pipefail

MODEL_CKPT="${1:-$HOME/dct-final-ckpt/deepcell-types_2026-06-15_resmlp.pt}"
EMB="${2:-embeddings/svd_512.npz}"
SPLIT="${SPLIT:-splits/fov_split_test_current.json}"
: "${DATA_DIR:?set DATA_DIR to the zarr archive}"

echo "Evaluating $MODEL_CKPT on the held-out test split ($SPLIT)"
python scripts/predict.py \
  --model_name eval_test \
  --model_path "$MODEL_CKPT" \
  --svd_embeddings_path "$EMB" \
  --split_file "$SPLIT" \
  --ct_abstention_k 0

# For the full multi-method head-to-head table (ours + XGBoost/MAPS/CellSighter),
# score the prediction CSVs with the research-workspace scorer:
#   python -m analysis.test_split_summary --methods \
#       resMLP=output/eval_test_prediction.csv \
#       XGBoost-tuned=<dct-final-ckpt>/baseline_xgboost_test_prediction.csv \
#       MAPS=<...>/baseline_maps_test_prediction.csv \
#       CellSighter=<...>/baseline_cellsighter_test_prediction.csv
# (baseline test CSVs ship in dct-final-ckpt/; their 129 test FOVs are bit-identical.)
