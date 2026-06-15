"""Regression tests for the FOV-grouped inner-validation carve.

The baselines (MAPS / CellSighter) select their checkpoint on this inner-val
set rather than the reported test set. For that to be leakage-free, the carve
must be at FOV granularity: a whole FOV goes entirely to inner-train or
inner-val, never split, so no cell used to select the checkpoint also trained
the model.
"""

# Import dataset before dataloader: the two modules have a pre-existing
# circular import (dataloader imports FullImageDataset; dataset re-exports from
# dataloader for back-compat), so importing dataloader first raises ImportError.
from deepcell_types.training.dataset import CellIndexRecord
from deepcell_types.training.dataloader import _carve_inner_val_fovs


class _MockDataset:
    """Stub matching the 8-tuple index layout: position 5 = fov_name, 6 = dataset_name."""

    def __init__(self, tuples):
        self.indices = tuples


def _dataset_with_fovs(n_fovs=5, cells_per_fov=3):
    tuples = []
    cell_idx = 0
    for f in range(n_fovs):
        for _ in range(cells_per_fov):
            tuples.append(
                CellIndexRecord(
                    0, "A", "A", "CODEX", cell_idx, f"FOV_{f}", "DS1", (10, 10)
                )
            )
            cell_idx += 1
    return _MockDataset(tuples)


def _fovs(dataset, indices):
    return {(dataset.indices[i][6], dataset.indices[i][5]) for i in indices}


def test_inner_val_carve_is_fov_grouped_and_disjoint():
    dataset = _dataset_with_fovs(n_fovs=5, cells_per_fov=3)
    train_indices = list(range(len(dataset.indices)))

    inner_train, inner_val = _carve_inner_val_fovs(
        dataset, train_indices, inner_val_ratio=0.4, inner_val_seed=42
    )

    assert inner_val is not None and len(inner_val) > 0
    train_fovs = _fovs(dataset, inner_train)
    val_fovs = _fovs(dataset, inner_val)

    # No FOV straddles the inner-train / inner-val boundary (the leakage guard).
    assert train_fovs.isdisjoint(val_fovs)
    # The cells form a clean partition of the original train indices.
    assert set(inner_train).isdisjoint(set(inner_val))
    assert set(inner_train) | set(inner_val) == set(train_indices)
    assert train_fovs | val_fovs == _fovs(dataset, train_indices)
    # round(5 * 0.4) == 2 inner-val FOVs.
    assert len(val_fovs) == 2


def test_inner_val_carve_noop_when_ratio_zero():
    dataset = _dataset_with_fovs()
    train_indices = list(range(len(dataset.indices)))

    out_train, out_val = _carve_inner_val_fovs(dataset, train_indices, 0.0, 42)

    assert out_val is None
    assert out_train == train_indices


def test_inner_val_carve_keeps_at_least_one_inner_train_fov():
    dataset = _dataset_with_fovs(n_fovs=2, cells_per_fov=2)
    train_indices = list(range(len(dataset.indices)))

    # ratio 1.0 would take every FOV; the cap must keep >=1 inner-train FOV.
    inner_train, inner_val = _carve_inner_val_fovs(dataset, train_indices, 1.0, 0)

    assert len(_fovs(dataset, inner_train)) >= 1
    assert _fovs(dataset, inner_train).isdisjoint(_fovs(dataset, inner_val))
