"""Tests for shared zarr annotation extraction."""

import logging

from deepcell_types.training.annotations import extract_cell_annotations


class _Group(dict):
    def __init__(self, attrs=None, **children):
        super().__init__(children)
        self.attrs = attrs or {}

    def __getitem__(self, key):
        if "/" not in key:
            return super().__getitem__(key)
        current = self
        for part in key.split("/"):
            current = dict.__getitem__(current, part)
        return current


def _dataset_with_source(source):
    preproc = _Group(
        attrs={
            "centroids": {
                "1": [10.0, 10.0],
                "2": [20.0, 20.0],
                "3": [30.0, 30.0],
            },
            "scale_factor": 1.0,
        }
    )
    annotations = _Group(attrs={"standardized_source": source})
    cell_types = _Group(annotations=annotations)
    ds = _Group(preprocessed=preproc, cell_types=cell_types)
    return ds, preproc


def test_duplicate_agreeing_labels_collapse_to_one_cell():
    ds, preproc = _dataset_with_source(
        {
            "CD4T": [1, 1],
            "CD8T": [2],
        }
    )

    cell_types, cell_indices, centroids = extract_cell_annotations(
        ds, "mock_fov", preproc, include_centroids=True
    )

    assert cell_types == ["CD4T", "CD8T"]
    assert cell_indices == [1, 2]
    assert centroids == [[10.0, 10.0], [20.0, 20.0]]


def test_conflicting_duplicate_labels_are_dropped(caplog):
    ds, preproc = _dataset_with_source(
        {
            "CD4T": [1],
            "Epithelial": [1, 1],
            "CD8T": [2],
        }
    )

    with caplog.at_level(logging.WARNING, logger="deepcell_types.training.annotations"):
        cell_types, cell_indices, _ = extract_cell_annotations(
            ds, "mock_fov", preproc, include_centroids=True
        )

    assert cell_types == ["CD8T"]
    assert cell_indices == [2]
    assert any(
        "dropped 1 cells with conflicting duplicate labels" in rec.getMessage()
        for rec in caplog.records
    )
