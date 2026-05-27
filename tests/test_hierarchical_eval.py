"""Test hierarchical accuracy computation."""
import numpy as np
from deepcell_types.training.utils import adjust_conf_mat_hierarchy


def test_child_predictions_moved_to_diagonal():
    hierarchy = {"Tcell": ["CD4T", "CD8T"]}
    ct2idx = {"CD4T": 0, "CD8T": 1, "Tcell": 2, "Tumor": 3}

    conf_mat = np.zeros((4, 4), dtype=int)
    conf_mat[2, 0] = 6   # Tcell -> CD4T (should become correct)
    conf_mat[2, 1] = 3   # Tcell -> CD8T (should become correct)
    conf_mat[2, 3] = 1   # Tcell -> Tumor (stays wrong)
    conf_mat[0, 0] = 5   # CD4T -> CD4T (unchanged)

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    assert adjusted[2, 2] == 9  # 6+3 moved to diagonal
    assert adjusted[2, 0] == 0  # cleared
    assert adjusted[2, 1] == 0  # cleared
    assert adjusted[2, 3] == 1  # unchanged
    assert adjusted[0, 0] == 5  # unchanged


def test_original_unchanged():
    hierarchy = {"Tcell": ["CD4T"]}
    ct2idx = {"CD4T": 0, "Tcell": 1}
    conf_mat = np.array([[5, 0], [3, 2]])

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    # Original should not be modified
    assert conf_mat[1, 0] == 3
    assert adjusted[1, 0] == 0
    assert adjusted[1, 1] == 5  # 2 + 3


def test_missing_types_ignored():
    hierarchy = {"Tcell": ["CD4T", "NonExistent"]}
    ct2idx = {"CD4T": 0, "Tcell": 1}
    conf_mat = np.array([[5, 0], [3, 2]])

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)
    assert adjusted[1, 1] == 5


def test_empty_hierarchy():
    conf_mat = np.array([[5, 1], [2, 3]])
    adjusted = adjust_conf_mat_hierarchy(conf_mat, {}, {"A": 0, "B": 1})
    np.testing.assert_array_equal(adjusted, conf_mat)


def test_parent_not_in_ct2idx():
    """If the parent type is not in ct2idx, it should be skipped."""
    hierarchy = {"UnknownParent": ["CD4T", "CD8T"]}
    ct2idx = {"CD4T": 0, "CD8T": 1, "Tumor": 2}
    conf_mat = np.array([[5, 1, 0], [2, 3, 0], [0, 0, 4]])

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)
    np.testing.assert_array_equal(adjusted, conf_mat)


def test_accumulates_with_existing_diagonal():
    """If parent already has correct predictions on diagonal, child counts add to them."""
    hierarchy = {"Tcell": ["CD4T"]}
    ct2idx = {"CD4T": 0, "Tcell": 1}
    conf_mat = np.array([[5, 0], [3, 7]])  # Tcell: 3 pred as CD4T, 7 correct

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)
    assert adjusted[1, 1] == 10  # 7 + 3
    assert adjusted[1, 0] == 0
