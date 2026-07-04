#!/usr/bin/env bash
#
# Release gate: verify a TissueNet zarr archive is compatible with the released
# model BEFORE publishing the archive or a checkpoint.
#
# `all_standardized_channels` IS the released model's marker->index map
# (config.DCTConfig builds marker2idx = enumerate(all_standardized_channels)),
# and the released checkpoint + embeddings/svd_512.npz are built for that exact
# order. Reordering or resizing it silently misaligns the checkpoint's
# per-marker weights, or fails the n_markers guard in predict._build_model so
# the checkpoint won't load. This wrapper runs the archive-contract validator
# with the released marker order as the reference and exits non-zero on any
# drift, so it can be dropped into a release pipeline or run by hand.
#
# (GitHub CI cannot run this — the multi-TB archive and the released svd are not
# present there; the validator's own unit tests do run in CI. Run this wherever
# the real archive + released embeddings live, as a pre-publish step.)
#
# Usage:
#   scripts/check_release_archive.sh <archive.zarr> <svd_512.npz>
# or via environment variables:
#   DEEPCELL_TYPES_ZARR_PATH=/path/to/archive.zarr \
#   DCT_RELEASE_SVD=/path/to/embeddings/svd_512.npz \
#   scripts/check_release_archive.sh
#
set -euo pipefail

ARCHIVE="${1:-${DEEPCELL_TYPES_ZARR_PATH:-}}"
SVD="${2:-${DCT_RELEASE_SVD:-}}"

if [[ -z "${ARCHIVE}" || -z "${SVD}" ]]; then
    echo "usage: $0 <archive.zarr> <svd_512.npz>" >&2
    echo "   or: set DEEPCELL_TYPES_ZARR_PATH and DCT_RELEASE_SVD" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
exec python "${here}/validate_archive_contract.py" \
    --zarr "${ARCHIVE}" \
    --marker-order-from "${SVD}"
