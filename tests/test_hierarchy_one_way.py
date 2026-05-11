"""Regression test for adjust_conf_mat_hierarchy one-way direction (issue #14, gap 1).

A future refactor that accidentally applies both directions (parent->child AND
child->parent) would silently inflate headline accuracy by ~1-2pp with no test
failure from the existing tests in `test_hierarchical_eval.py`. The existing
tests only check the parent->child direction is credited; they do not verify
the child->parent direction is NOT credited.

Reference: deepcelltypes/utils.py::adjust_conf_mat_hierarchy
"""
import numpy as np

from deepcell_types.training.utils import adjust_conf_mat_hierarchy


def test_child_to_parent_not_credited():
    """Child-pred-on-parent-GT is credited; parent-pred-on-child-GT is not."""
    hierarchy = {"Tcell": ["CD4T", "CD8T", "Treg", "NKT"]}
    ct2idx = {"CD4T": 0, "CD8T": 1, "Treg": 2, "NKT": 3, "Tcell": 4, "Other": 5}

    n = len(ct2idx)
    conf_mat = np.zeros((n, n), dtype=np.int64)

    parent_idx = ct2idx["Tcell"]
    child_idx = ct2idx["CD4T"]

    # GT=parent, pred=child: should move to diagonal (parent credited).
    conf_mat[parent_idx, child_idx] = 100
    # GT=child, pred=parent: must STAY off-diagonal.
    #   (Predicting "Tcell" when the ground truth is "CD4T" is a less-specific
    #    prediction and should not be credited as correct — the model failed to
    #    distinguish the subtype.)
    conf_mat[child_idx, parent_idx] = 50

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    # Parent row: 100 moved to diagonal
    assert adjusted[parent_idx, parent_idx] == 100
    assert adjusted[parent_idx, child_idx] == 0

    # Child row: 50 stays off-diagonal, diagonal is NOT inflated
    assert adjusted[child_idx, child_idx] == 0, (
        "child->parent predictions must not be credited as correct; found "
        f"adjusted[{child_idx}, {child_idx}] = {adjusted[child_idx, child_idx]} "
        "(expected 0). This would silently inflate headline accuracy."
    )
    assert adjusted[child_idx, parent_idx] == 50


def test_bidirectional_multiple_children():
    """Same one-way property holds for every (parent, child) pair in hierarchy."""
    hierarchy = {
        "Tcell": ["CD4T", "CD8T", "Treg", "NKT"],
        "Stromal": ["Fibroblast", "Pericyte"],
    }
    ct2idx = {
        "CD4T": 0, "CD8T": 1, "Treg": 2, "NKT": 3, "Tcell": 4,
        "Fibroblast": 5, "Pericyte": 6, "Stromal": 7,
    }
    n = len(ct2idx)

    conf_mat = np.zeros((n, n), dtype=np.int64)
    # Seed each child row with a parent-prediction count.
    for parent, children in hierarchy.items():
        parent_idx = ct2idx[parent]
        for child in children:
            child_idx = ct2idx[child]
            conf_mat[child_idx, parent_idx] = 7  # child-GT, parent-pred
            conf_mat[parent_idx, child_idx] = 3  # parent-GT, child-pred

    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    for parent, children in hierarchy.items():
        parent_idx = ct2idx[parent]
        for child in children:
            child_idx = ct2idx[child]
            # Child-GT diagonal untouched; parent-pred still off-diagonal
            assert adjusted[child_idx, child_idx] == 0
            assert adjusted[child_idx, parent_idx] == 7
        # Parent diagonal aggregates the children's 3s
        assert adjusted[parent_idx, parent_idx] == 3 * len(children)
        for child in children:
            assert adjusted[parent_idx, ct2idx[child]] == 0


def test_accuracy_inflation_detector():
    """Regression anchor: if both directions were credited, headline accuracy
    would be artificially higher.

    Sets up a matrix where a naive bidirectional refactor adds exactly
    50 extra correct predictions to the diagonal. This test pins the
    correct-accuracy value so accidental inflation fails here first.
    """
    hierarchy = {"Tcell": ["CD4T"]}
    ct2idx = {"CD4T": 0, "Tcell": 1}

    conf_mat = np.array(
        [
            [10, 50],   # GT=CD4T: 10 correct, 50 predicted as Tcell (parent)
            [0, 40],    # GT=Tcell: 40 correct, 0 predicted as CD4T
        ],
        dtype=np.int64,
    )
    adjusted = adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx)

    # Correct one-way behavior: only the parent-GT row is adjusted
    total = adjusted.sum()
    correct = np.trace(adjusted)
    assert total == 100
    assert correct == 50, (
        f"Expected 50 correct (10 CD4T + 40 Tcell); got {correct}. "
        "If this is 100, both hierarchy directions were credited — a bug."
    )
