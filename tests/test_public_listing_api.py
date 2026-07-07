"""Unit tests for the public registry-listing helpers (``list_supported_markers`` /
``list_supported_cell_types``). These let a user pre-flight-check their marker
panel against the packaged registry before downloading a checkpoint; they must
stay importable from the top-level package and free of a torch import (same
inference/train dependency split guarded by ``test_inference_deps.py``).
"""

import deepcell_types
from deepcell_types import list_supported_cell_types, list_supported_markers
from deepcell_types.config import DCTConfig


def test_list_supported_markers_matches_config():
    markers = list_supported_markers()
    assert markers == sorted(markers)
    assert markers
    assert set(markers) == set(DCTConfig().marker2idx)


def test_list_supported_cell_types_matches_config():
    cell_types = list_supported_cell_types()
    assert cell_types == sorted(cell_types)
    assert cell_types
    assert set(cell_types) == set(DCTConfig().ct2idx)


def test_listing_helpers_importable_from_top_level_package():
    assert deepcell_types.list_supported_markers is list_supported_markers
    assert deepcell_types.list_supported_cell_types is list_supported_cell_types
