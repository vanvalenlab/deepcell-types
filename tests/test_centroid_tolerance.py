"""Unit tests for centroid_to_cell_idx_fast tolerance behavior.

Round-2 audit found 4 mcmicro_TMA11 FOVs losing 38-40% of cells at tol=1.0
because of sub-pixel drift between standardized_source centroids and
preprocessed centroids amplified by scale_factor=1.3. Default tolerance was
bumped to 1.5 to recover those cells. These tests pin the behavior at both
tolerance values so the bump can't be reverted accidentally.
"""
import numpy as np

from deepcell_types.training.annotations import build_centroid_tree, centroid_to_cell_idx_fast


def _tree_from(centroids):
    """Build (tree, keys) for a list of centroids assigned cell-id 1..N."""
    centroids_raw = {str(i + 1): list(c) for i, c in enumerate(centroids)}
    tree, keys = build_centroid_tree(centroids_raw)
    return tree, keys, centroids_raw


def test_default_tol_resolves_subpixel_drift():
    """A 1.3 px offset (typical mcmicro_TMA11 drift) must resolve at tol=1.5."""
    tree, keys, _ = _tree_from([(100.0, 100.0)])
    # Target with 1.3 px offset (Euclidean, e.g., +1.3 in row, 0 in col)
    idx = centroid_to_cell_idx_fast(tree, keys, [101.3, 100.0])
    assert idx == 1


def test_default_tol_drops_clearly_out_of_range():
    """A 1.6 px offset should still drop at tol=1.5."""
    tree, keys, _ = _tree_from([(100.0, 100.0)])
    idx = centroid_to_cell_idx_fast(tree, keys, [101.6, 100.0])
    assert idx is None


def test_explicit_tight_tolerance_drops_subpixel_drift():
    """The same 1.3 px offset that resolves at tol=1.5 must drop at tol=1.0
    (defends against accidental tolerance regression)."""
    tree, keys, _ = _tree_from([(100.0, 100.0)])
    idx = centroid_to_cell_idx_fast(tree, keys, [101.3, 100.0], tol=1.0)
    assert idx is None


def test_exact_match():
    tree, keys, _ = _tree_from([(50.0, 50.0)])
    idx = centroid_to_cell_idx_fast(tree, keys, [50.0, 50.0])
    assert idx == 1


def test_picks_nearest_among_neighbors():
    """When multiple cells are within tolerance, pick the closest one."""
    # Cells 5 px apart along the row axis
    tree, keys, _ = _tree_from([(100.0, 100.0), (105.0, 100.0)])
    # Target at (100.5, 100.0) — closer to cell 1 (0.5px) than cell 2 (4.5px).
    idx = centroid_to_cell_idx_fast(tree, keys, [100.5, 100.0])
    assert idx == 1
