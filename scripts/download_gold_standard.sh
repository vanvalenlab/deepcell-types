#!/usr/bin/env bash
# Download the Pan-Multiplex Gold Standard dataset from HuggingFace.
#
# Reference: Rumberger et al., "Automated classification of cellular expression
#   in multiplexed imaging data with Nimbus", Nature Methods 2025.
# Dataset: https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard
#
# The dataset contains ~1.1M expert-annotated marker positivity labels across
# 5 tissue/modality subsets:
#   - codex_colon, mibi_breast, mibi_decidua, vectra_colon, vectra_pancreas
#
# Labels: 0=negative, 1=positive, 2=ambiguous, 3=ambiguous
#
# Usage:
#   bash scripts/download_gold_standard.sh [output_dir]
#
# Default output: data/gold_standard/

set -euo pipefail

OUTPUT_DIR="${1:-data/gold_standard}"
ZIP_NAME="gold_standard_labelled.zip"
HF_URL="https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard/resolve/main/${ZIP_NAME}"

mkdir -p "$OUTPUT_DIR"

echo "Downloading Pan-Multiplex Gold Standard dataset..."
echo "  Source: $HF_URL"
echo "  Output: $OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/$ZIP_NAME" ]; then
    echo "  Zip already exists, skipping download."
else
    wget -q --show-progress -O "$OUTPUT_DIR/$ZIP_NAME" "$HF_URL"
fi

echo "Extracting..."
cd "$OUTPUT_DIR"
unzip -q -o "$ZIP_NAME"

echo "Done. Contents:"
ls -la
echo ""
echo "Subsets:"
find . -maxdepth 2 -type d | head -20
