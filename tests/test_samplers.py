"""Tests for FOVGroupedSampler (R7 L1) and LazyMarkerPositivityDict (R7 L2)."""

import pickle

import pytest
import torch

from deepcell_types.training.dataset import FOVGroupedSampler


# =============================================================================
# R7 L1 — FOVGroupedSampler: cells from the same FOV must stay batched together
# =============================================================================


class TestFOVGroupedSampler:
    def _build_dataset(self, fov_sizes):
        """Build a synthetic dataset.indices list with one cell per entry.

        fov_sizes: list of ints, one per FOV (i.e., ds_idx). E.g. [3, 5, 2]
        means ds_idx 0 has 3 cells, ds_idx 1 has 5, ds_idx 2 has 2.

        dataset.indices tuple matches the real layout:
            (ds_idx, ct_label, ct_label_standard, domain,
             cell_idx, fov_name, dataset_name, centroid)
        """
        indices = []
        for ds_idx, n in enumerate(fov_sizes):
            for cell_idx in range(n):
                indices.append(
                    (
                        ds_idx,
                        "T",
                        "T_cell",
                        "CODEX",
                        cell_idx,
                        f"FOV{ds_idx}",
                        f"DS{ds_idx}",
                        (float(cell_idx), float(cell_idx)),
                    )
                )
        return indices

    def test_groups_same_fov_together(self):
        """Each contiguous run of identical ds_idx must be preserved in output."""
        torch.manual_seed(0)
        fov_sizes = [4, 6, 5, 3]
        dataset_indices = self._build_dataset(fov_sizes)
        n = len(dataset_indices)  # 18
        train_indices = list(range(n))

        weights = torch.ones(n, dtype=torch.float)
        sampler = FOVGroupedSampler(
            weights=weights,
            num_samples=n,
            dataset_indices=dataset_indices,
            train_indices=train_indices,
            replacement=False,
        )

        yielded = list(sampler)
        assert len(yielded) == n

        # Walk the yielded sequence; each ds_idx appears as a single contiguous block.
        seen_blocks = []
        current_block = None
        for pos in yielded:
            ds_idx = dataset_indices[pos][0]
            if current_block is None or ds_idx != current_block:
                # Starting a new block: make sure we haven't seen this ds_idx before.
                assert ds_idx not in seen_blocks, (
                    f"ds_idx={ds_idx} appeared in two non-contiguous blocks — "
                    f"cross-FOV leakage detected"
                )
                seen_blocks.append(ds_idx)
                current_block = ds_idx

    def test_groups_same_fov_with_replacement(self):
        """With replacement=True, samples drawn repeatedly must still be
        grouped by FOV (ds_idx runs are contiguous)."""
        torch.manual_seed(0)
        fov_sizes = [3, 4, 2]
        dataset_indices = self._build_dataset(fov_sizes)
        n = len(dataset_indices)  # 9
        train_indices = list(range(n))

        weights = torch.ones(n, dtype=torch.float)
        sampler = FOVGroupedSampler(
            weights=weights,
            num_samples=50,  # oversample
            dataset_indices=dataset_indices,
            train_indices=train_indices,
            replacement=True,
        )

        yielded = list(sampler)
        assert len(yielded) == 50

        # Same invariant: each ds_idx block is contiguous (no interleaving).
        seen_blocks = []
        current_block = None
        for pos in yielded:
            ds_idx = dataset_indices[pos][0]
            if current_block is None or ds_idx != current_block:
                assert ds_idx not in seen_blocks, (
                    f"ds_idx={ds_idx} appears in two non-contiguous blocks"
                )
                seen_blocks.append(ds_idx)
                current_block = ds_idx

    def test_length_matches_num_samples(self):
        """__len__ should equal num_samples."""
        dataset_indices = self._build_dataset([2, 2])
        sampler = FOVGroupedSampler(
            weights=torch.ones(4),
            num_samples=123,
            dataset_indices=dataset_indices,
            train_indices=[0, 1, 2, 3],
            replacement=True,
        )
        assert len(sampler) == 123

    def test_zero_sample_epoch_is_empty(self):
        dataset_indices = self._build_dataset([2, 2])
        sampler = FOVGroupedSampler(
            weights=torch.ones(4),
            num_samples=0,
            dataset_indices=dataset_indices,
            train_indices=[0, 1, 2, 3],
            replacement=True,
        )

        assert list(sampler) == []
        assert len(sampler) == 0


# =============================================================================
# R7 L2 — LazyMarkerPositivityDict: lazy load semantics
# =============================================================================


class _StubConfig:
    """Stand-in for TissueNetConfig used only for _load_marker_positivity."""

    def __init__(self, data):
        # data: {dataset_key: DataFrame}
        self._data = data
        self.calls = []  # list of keys in the order they were loaded

    def _load_marker_positivity(self, dataset_key):
        self.calls.append(dataset_key)
        return self._data.get(dataset_key)


class TestLazyMarkerPositivityDict:
    def _make_df(self, cell_types, markers, values):
        import pandas as pd

        return pd.DataFrame(values, index=cell_types, columns=markers)

    def test_access_triggers_single_load(self):
        """Accessing a key via [] triggers _load_marker_positivity exactly once."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        df_a = self._make_df(["T"], ["CD3"], [[1.0]])
        df_b = self._make_df(["B"], ["CD19"], [[1.0]])
        stub = _StubConfig({"A": df_a, "B": df_b})
        lazy = LazyMarkerPositivityDict(stub, ["A", "B"])

        # Nothing loaded yet
        assert stub.calls == []

        # First access → one load
        out1 = lazy["A"]
        assert stub.calls == ["A"]
        # Values match direct read from underlying source
        assert out1.equals(df_a)

        # Second access → no additional load (cached)
        out2 = lazy["A"]
        assert stub.calls == ["A"], f"expected no extra load, got {stub.calls}"
        assert out2.equals(df_a)

    def test_contains_triggers_load(self):
        """`in` operator should also lazy-load exactly once."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        df_a = self._make_df(["T"], ["CD3"], [[1.0]])
        stub = _StubConfig({"A": df_a})
        lazy = LazyMarkerPositivityDict(stub, ["A"])

        assert stub.calls == []
        assert "A" in lazy
        assert stub.calls == ["A"]
        # Second membership check must not re-load
        assert "A" in lazy
        assert stub.calls == ["A"]

    def test_unknown_key_absent(self):
        """A key not in mp_keys returns False for `in` and raises KeyError on []."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        stub = _StubConfig({})
        lazy = LazyMarkerPositivityDict(stub, ["A"])

        assert "Z" not in lazy
        assert stub.calls == []
        with pytest.raises(KeyError):
            _ = lazy["Z"]

    def test_get_returns_default_for_missing_key(self):
        """`.get(missing_key, default)` returns default without raising."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        stub = _StubConfig({})
        lazy = LazyMarkerPositivityDict(stub, ["A"])  # "A" is declared but has no data
        sentinel = object()
        assert lazy.get("nonexistent", sentinel) is sentinel

    def test_values_match_underlying_source(self):
        """Values returned via [] match a direct call to _load_marker_positivity."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        df = self._make_df(
            ["T_cell", "B_cell"],
            ["CD3", "CD19"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        stub = _StubConfig({"DS1": df})
        lazy = LazyMarkerPositivityDict(stub, ["DS1"])

        # Pull the direct source of truth
        direct = stub._data["DS1"]
        lazy_result = lazy["DS1"]

        assert lazy_result.equals(direct)

    def test_iteration_loads_all(self):
        """Iterating (keys/values/items) should surface all declared MP keys."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        df_a = self._make_df(["T"], ["CD3"], [[1.0]])
        df_b = self._make_df(["B"], ["CD19"], [[1.0]])
        stub = _StubConfig({"A": df_a, "B": df_b})
        lazy = LazyMarkerPositivityDict(stub, ["A", "B"])

        seen = set(lazy.keys())
        assert seen == {"A", "B"}
        # All keys were loaded at most once
        assert sorted(stub.calls) == ["A", "B"]
        assert len(stub.calls) == len(set(stub.calls))

    def test_load_skips_keys_returning_none(self):
        """If _load_marker_positivity returns None, the key is absent from the dict."""
        from deepcell_types.training.config import LazyMarkerPositivityDict

        # "A" returns None (no MP group), "B" has data
        df_b = self._make_df(["T"], ["CD3"], [[1.0]])
        stub = _StubConfig({"B": df_b})  # "A" not in stub data → returns None
        lazy = LazyMarkerPositivityDict(stub, ["A", "B"])

        # "A" is in mp_keys but lookup returns None → absent
        assert (
            "A" not in lazy
        )  # triggers load, but value is None, so __contains__ returns False
        assert stub.calls == ["A"]
        # Access raises KeyError because None was not stored
        with pytest.raises(KeyError):
            _ = lazy["A"]

        # "B" is present
        assert "B" in lazy
        assert lazy["B"].equals(df_b)

    def test_pickling_does_not_force_full_load(self):
        from deepcell_types.training.config import LazyMarkerPositivityDict

        df_a = self._make_df(["T"], ["CD3"], [[1.0]])
        stub = _StubConfig({"A": df_a})
        lazy = LazyMarkerPositivityDict(stub, ["A", "B"])

        pickle.dumps(lazy)

        assert stub.calls == []
