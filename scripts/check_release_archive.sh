#!/usr/bin/env bash
#
# Release gate: verify a TissueNet zarr archive is compatible with the released
# model BEFORE publishing the archive or a checkpoint.
#
# `all_standardized_channels` IS the released model's marker->index map
# (config.DCTConfig builds marker2idx = enumerate(all_standardized_channels)),
# and the released checkpoint is built for that exact
# order. Reordering or resizing it silently misaligns the checkpoint's
# per-marker weights, or fails the n_markers guard in predict._build_model so
# the checkpoint won't load. This wrapper runs the archive-contract validator
# with the released marker order as the reference and exits non-zero on any
# drift, so it can be dropped into a release pipeline or run by hand.
#
# (GitHub CI cannot run this — the multi-TB archive and released marker map are not
# present there; the validator's own unit tests do run in CI. Run this wherever
# the real archive + released marker map live, as a pre-publish step.)
#
# Usage:
#   scripts/check_release_archive.sh <archive.zarr> <marker_order.json>
# or via environment variables:
#   DEEPCELL_TYPES_ZARR_PATH=/path/to/archive.zarr \
#   DCT_RELEASE_MARKER_ORDER=/path/to/marker_order.json \
#   scripts/check_release_archive.sh
#
set -euo pipefail

ARCHIVE="${1:-${DEEPCELL_TYPES_ZARR_PATH:-}}"
MARKER_ORDER="${2:-${DCT_RELEASE_MARKER_ORDER:-}}"

if [[ -z "${ARCHIVE}" || -z "${MARKER_ORDER}" ]]; then
    echo "usage: $0 <archive.zarr> <marker_order.json>" >&2
    echo "   or: set DEEPCELL_TYPES_ZARR_PATH and DCT_RELEASE_MARKER_ORDER" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
exec python "${here}/validate_archive_contract.py" \
    --zarr "${ARCHIVE}" \
    --marker-order-from "${MARKER_ORDER}"
