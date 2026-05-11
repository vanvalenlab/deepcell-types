"""Regression tests for dataset centroid lookup and FOV split logic (issue #14, gaps 2-4).

Exercises:
- `FullImageDataset._lookup_centroid` — zarr-v3 string/int key duality.
- `_find_sole_source_fovs` / `create_fov_splits` — rare-class stratification.
- `save_fov_splits` / `load_fov_splits` — JSON round-trip between writer and reader.
"""

import json
from collections import defaultdict

import pytest

from deepcell_types.training.dataset import (
    CellIndexRecord,
    FullImageDataset,
    _find_sole_source_fovs,
    create_fov_splits,
    load_fov_splits,
    save_fov_splits,
)


# =============================================================================
# Gap 2: _lookup_centroid zarr-v3 key duality
# =============================================================================


class TestLookupCentroidKeyDuality:
    """Zarr v3 serializes attribute keys as strings (JSON) regardless of the
    originating Python type. A centroid dict with integer cell-index keys
    becomes {"1": [r, c], "2": [r, c]} after round-tripping through the
    archive. `_lookup_centroid` must find the centroid whether the caller
    passes an int or a string, and whether the dict was stored with int or
    string keys.
    """

    def test_string_keyed_dict_int_query(self):
        """Zarr-v3 canonical case: dict has string keys, caller has int."""
        ann = {"1": [10.0, 20.0], "2": [30.0, 40.0]}
        assert FullImageDataset._lookup_centroid(ann, 1) == [10.0, 20.0]
        assert FullImageDataset._lookup_centroid(ann, 2) == [30.0, 40.0]

    def test_int_keyed_dict_int_query(self):
        """Legacy (v2) path: dict has int keys, caller has int."""
        ann = {1: [10.0, 20.0], 2: [30.0, 40.0]}
        assert FullImageDataset._lookup_centroid(ann, 1) == [10.0, 20.0]
        assert FullImageDataset._lookup_centroid(ann, 2) == [30.0, 40.0]

    def test_missing_key_returns_none(self):
        """Non-existent cell index returns None (both str and int miss)."""
        ann = {"1": [10.0, 20.0]}
        assert FullImageDataset._lookup_centroid(ann, 99) is None

    def test_empty_dict(self):
        assert FullImageDataset._lookup_centroid({}, 1) is None

    def test_large_int_indices(self):
        """Sanity: string conversion of large ints matches."""
        ann = {"1000000": [7.0, 8.0]}
        assert FullImageDataset._lookup_centroid(ann, 1000000) == [7.0, 8.0]


# =============================================================================
# Gap 3: rare-class stratification
# =============================================================================


class _MockDataset:
    """Minimal dataset stub whose `indices` matches the 8-tuple layout produced
    by `FullImageDataset.__init__`:
        (ds_idx, ct_label, ct_label_standard, domain, cell_idx,
         fov_name, dataset_name, centroid)
    """

    def __init__(self, tuples):
        self.indices = tuples


def _make_indices(fovs_by_class):
    """Build a flat list of index tuples from a {ct_name: [(dataset, fov), ...]} mapping.

    Produces 1 cell per (dataset, fov, ct_name) entry — enough for stratification
    logic which only looks at per-FOV class membership, not cell count.
    """
    tuples = []
    cell_idx = 0
    for ct_name, fovs in fovs_by_class.items():
        for ds_name, fov_name in fovs:
            tuples.append(
                CellIndexRecord(0, ct_name, ct_name, "CODEX", cell_idx, fov_name, ds_name, (10, 10))
            )
            cell_idx += 1
    return tuples


class TestFindSoleSourceFovs:
    """The rare-class stratification primitive that prevents v5 `0-train` bugs."""

    def test_single_fov_class_is_forced(self):
        """If a class appears in only 1 FOV, that FOV must be forced into train."""
        tuples = _make_indices(
            {
                "Common": [("DS1", "FOV_A"), ("DS1", "FOV_B"), ("DS1", "FOV_C")],
                "RareCT": [("DS1", "FOV_A")],  # sole source: FOV_A
            }
        )
        dataset = _MockDataset(tuples)

        fov_to_indices = defaultdict(list)
        for i, t in enumerate(dataset.indices):
            fov_to_indices[(t[6], t[5])].append(i)

        forced = _find_sole_source_fovs(dataset, fov_to_indices)
        assert ("DS1", "FOV_A") in forced

    def test_multi_fov_class_is_not_forced(self):
        """A class with 2+ FOVs is free to split normally."""
        tuples = _make_indices(
            {
                "Common": [("DS1", "FOV_A"), ("DS1", "FOV_B"), ("DS1", "FOV_C")],
                "NotRare": [("DS1", "FOV_A"), ("DS1", "FOV_B")],  # 2 FOVs
            }
        )
        dataset = _MockDataset(tuples)

        fov_to_indices = defaultdict(list)
        for i, t in enumerate(dataset.indices):
            fov_to_indices[(t[6], t[5])].append(i)

        forced = _find_sole_source_fovs(dataset, fov_to_indices)
        # Neither FOV_A nor FOV_B is forced solely because of NotRare
        # (they might still be forced by another class — check neither class makes them sole-source)
        assert forced == set(), "Classes with >=2 FOVs should not trigger forcing"


class TestCreateFovSplitsRareStratification:
    """End-to-end: rare classes must have non-zero train coverage."""

    def test_rare_class_lands_in_train(self):
        """A class that appears only in one FOV is guaranteed to be in train."""
        tuples = _make_indices(
            {
                "Common": [
                    ("DS1", "FOV_1"),
                    ("DS1", "FOV_2"),
                    ("DS1", "FOV_3"),
                    ("DS1", "FOV_4"),
                    ("DS1", "FOV_5"),
                ],
                "RareSingleSource": [("DS1", "FOV_3")],  # sole source
            }
        )
        dataset = _MockDataset(tuples)

        # Try multiple seeds — rare-source FOV must always be in train.
        for seed in range(10):
            train_idx, val_idx = create_fov_splits(dataset, train_ratio=0.5, seed=seed)
            train_cts = {dataset.indices[i][2] for i in train_idx}
            assert "RareSingleSource" in train_cts, (
                f"seed={seed}: sole-source FOV landed in val; rare-class "
                "stratification broken"
            )

    def test_n_train_remaining_nonnegative(self):
        """When forced_train exceeds target_train, n_train_remaining clamps at 0.

        This is the 'rare-class forcing clamp' branch — a negative slice would
        silently steal from val with slice-from-end semantics.
        """
        # 10 FOVs total, each a sole source of its own class.
        # target_train = int(10 * 0.1) = 1, forced_train = 10 -> remaining clamps to 0.
        tuples = _make_indices(
            {f"SoleClass{i}": [("DS1", f"FOV_{i}")] for i in range(10)}
        )
        dataset = _MockDataset(tuples)

        train_idx, val_idx = create_fov_splits(dataset, train_ratio=0.1, seed=0)
        # All 10 FOVs were sole-source, so all must be in train; val is empty.
        assert len(train_idx) == 10
        assert len(val_idx) == 0
        # No negative slicing bug (would yield spurious val entries)

    def test_no_overlap_with_forcing(self):
        """Train/val must remain disjoint even when forcing is active."""
        tuples = _make_indices(
            {
                "Common": [("DS1", f"FOV_{i}") for i in range(8)],
                "Rare": [("DS1", "FOV_0")],
            }
        )
        dataset = _MockDataset(tuples)

        train_idx, val_idx = create_fov_splits(dataset, train_ratio=0.6, seed=0)
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(range(len(dataset.indices)))


# =============================================================================
# Gap 4: save_fov_splits / load_fov_splits round-trip
# =============================================================================


class TestFovSplitRoundTrip:
    """save_fov_splits writer must match load_fov_splits reader schema exactly.

    Guards against silent JSON-schema drift that would leave some FOVs in a
    'skipped' limbo at load time.
    """

    def _build_dataset(self):
        tuples = _make_indices(
            {
                "CD4T": [("DS1", "FOV_1"), ("DS2", "FOV_2"), ("DS2", "FOV_3")],
                "CD8T": [("DS1", "FOV_1"), ("DS1", "FOV_4"), ("DS2", "FOV_5")],
                "Tumor": [("DS1", "FOV_4"), ("DS2", "FOV_5")],
            }
        )
        return _MockDataset(tuples)

    def test_round_trip_preserves_indices(self, tmp_path):
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"

        saved_train, saved_val = save_fov_splits(
            dataset, str(split_file), train_ratio=0.7, seed=42
        )
        loaded_train, loaded_val = load_fov_splits(dataset, str(split_file))

        # Set equality (order may differ because load_fov_splits iterates
        # dataset.indices, which can regroup cells from the same FOV).
        assert set(saved_train) == set(loaded_train)
        assert set(saved_val) == set(loaded_val)

    def test_round_trip_preserves_fov_assignment(self, tmp_path):
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"

        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)
        loaded_train, loaded_val = load_fov_splits(dataset, str(split_file))

        train_fovs = {
            (dataset.indices[i][6], dataset.indices[i][5]) for i in loaded_train
        }
        val_fovs = {(dataset.indices[i][6], dataset.indices[i][5]) for i in loaded_val}

        # A FOV must never appear in both splits
        assert train_fovs.isdisjoint(val_fovs)

    def test_schema_has_expected_keys(self, tmp_path):
        """The JSON writer must produce the keys the reader expects."""
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        assert "train" in data
        assert "val" in data
        assert "metadata" in data
        assert "min_channels" in data["metadata"]
        assert "max_channels" in data["metadata"]
        assert "num_marker_channels" in data["metadata"]
        assert "num_cell_types" in data["metadata"]
        assert "zarr_path" in data["metadata"]
        # Per-dataset FOV name lists
        for ds_name, fov_list in data["train"].items():
            assert isinstance(fov_list, list)
            assert all(isinstance(f, str) for f in fov_list)

    def test_load_warns_on_filter_metadata_mismatch_when_non_strict(
        self, tmp_path, caplog
    ):
        """Split files should expose stale filter/archive assumptions."""
        import logging

        dataset = self._build_dataset()
        dataset.min_channels = 3
        dataset.max_channels = 80
        dataset.marker2idx = {"CD3": 0}
        dataset.ct2idx = {"CD4T": 0, "CD8T": 1, "Tumor": 2}
        dataset._zarr_path = "/archive/original.zarr"
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        dataset.min_channels = 0

        with caplog.at_level(logging.WARNING, logger="deepcell_types.training.dataset"):
            load_fov_splits(dataset, str(split_file), strict=False)

        assert any(
            "split metadata mismatch: min_channels" in rec.getMessage()
            for rec in caplog.records
        )

    def test_load_fails_on_filter_metadata_mismatch(self, tmp_path):
        """Strict split loading rejects stale filter/archive assumptions."""
        dataset = self._build_dataset()
        dataset.min_channels = 3
        dataset.max_channels = 80
        dataset.marker2idx = {"CD3": 0}
        dataset.ct2idx = {"CD4T": 0, "CD8T": 1, "Tumor": 2}
        dataset._zarr_path = "/archive/original.zarr"
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        dataset.min_channels = 0

        with pytest.raises(ValueError, match="split metadata mismatch: min_channels"):
            load_fov_splits(dataset, str(split_file))

    def test_load_allows_zarr_path_metadata_mismatch(self, tmp_path, caplog):
        """Committed split files should not be tied to one mount point."""
        import logging

        dataset = self._build_dataset()
        dataset.min_channels = 3
        dataset.max_channels = 80
        dataset.marker2idx = {"CD3": 0}
        dataset.ct2idx = {"CD4T": 0, "CD8T": 1, "Tumor": 2}
        dataset._zarr_path = "/archive/original.zarr"
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        dataset._zarr_path = "/mnt/same-archive.zarr"

        with caplog.at_level(logging.WARNING, logger="deepcell_types.training.dataset"):
            loaded_train, loaded_val = load_fov_splits(dataset, str(split_file))

        assert len(loaded_train) + len(loaded_val) == len(dataset.indices)
        assert any("advisory mismatch" in rec.getMessage() for rec in caplog.records)

    def test_load_fails_when_required_metadata_missing(self, tmp_path):
        """Strict split loading rejects missing provenance fields."""
        dataset = self._build_dataset()
        dataset.min_channels = 3
        dataset.max_channels = 80
        dataset.marker2idx = {"CD3": 0}
        dataset.ct2idx = {"CD4T": 0, "CD8T": 1, "Tumor": 2}
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        del data["metadata"]["min_channels"]
        with open(split_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="min_channels: file is missing"):
            load_fov_splits(dataset, str(split_file))

    def test_no_skipped_samples_after_roundtrip(self, tmp_path, caplog):
        """load_fov_splits warns if any sample's FOV is absent from the split file.
        Round-trip must produce zero warnings.
        """
        import logging

        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with caplog.at_level(logging.WARNING, logger="deepcell_types.training.dataset"):
            loaded_train, loaded_val = load_fov_splits(dataset, str(split_file))

        # Every sample must be assigned to train or val.
        assert len(loaded_train) + len(loaded_val) == len(dataset.indices)
        # No "samples not found in split file" warning.
        missing_warnings = [
            rec
            for rec in caplog.records
            if "not found in split file" in rec.getMessage()
        ]
        assert missing_warnings == []

    def test_load_fails_when_split_omits_samples(self, tmp_path):
        """Strict split loading should not silently shrink the dataset."""
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        omitted_ds, omitted_fovs = next(iter(data["train"].items()))
        omitted_fovs.pop()
        if not omitted_fovs:
            del data["train"][omitted_ds]
        with open(split_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="samples not found in split file"):
            load_fov_splits(dataset, str(split_file))

    def test_load_fails_when_split_lists_filtered_fov(self, tmp_path):
        """Strict split loading should catch split FOVs missing after filters."""
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        data["train"].setdefault("DS_missing", []).append("FOV_missing")
        with open(split_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="split FOVs are not present"):
            load_fov_splits(dataset, str(split_file))

    def test_load_allows_declared_heldout_samples(self, tmp_path):
        """Subset split files can intentionally hold out omitted FOVs."""
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        omitted_ds, omitted_fovs = next(iter(data["train"].items()))
        omitted_fov = omitted_fovs.pop()
        if not omitted_fovs:
            del data["train"][omitted_ds]
        data["heldout"] = {omitted_ds: [omitted_fov]}
        with open(split_file, "w") as f:
            json.dump(data, f)

        loaded_train, loaded_val = load_fov_splits(dataset, str(split_file))

        heldout_count = sum(
            1
            for idx_tuple in dataset.indices
            if idx_tuple[6] == omitted_ds and idx_tuple[5] == omitted_fov
        )
        assert (
            len(loaded_train) + len(loaded_val) == len(dataset.indices) - heldout_count
        )

    def test_load_fails_when_heldout_lists_filtered_fov(self, tmp_path):
        """Strict split loading should validate heldout FOV existence too."""
        dataset = self._build_dataset()
        split_file = tmp_path / "splits.json"
        save_fov_splits(dataset, str(split_file), train_ratio=0.7, seed=42)

        with open(split_file) as f:
            data = json.load(f)
        data["heldout"] = {"DS_missing": ["FOV_missing"]}
        with open(split_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="heldout FOVs are not present"):
            load_fov_splits(dataset, str(split_file))
