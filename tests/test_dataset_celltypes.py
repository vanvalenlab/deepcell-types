"""Characterization tests for per-dataset cell-type aggregation.

Pins the behaviour of two pieces of ``TissueNetConfig`` so the
``celltype_mapping`` (identity-dict) -> ``dataset_celltypes`` (name list)
simplification provably preserves behaviour:

- ``_aggregate_metadata`` — the pure single-pass aggregation extracted from
  ``_compute_all_mappings`` (which otherwise reads ~1GB of zarr.json off disk
  via a ProcessPoolExecutor and cannot be unit-tested directly).
- ``build_tissue_mapping_from_split`` — the one consumer that genuinely needs
  the per-dataset cell-type *names* (the other two call sites only ever did
  ``ct in ct2idx``).

The key invariant being locked in: ``dataset_celltypes`` retains *every*
annotated name per dataset (including names absent from ``ct2idx``); the
``ct2idx`` filter is applied only by the tissue-level mappings and the
consumers, exactly as the old identity-dict code did.
"""

import json

from deepcell_types.training.config import TissueNetConfig


def _meta(key, domain="CODEX", tissue=None, ct_names=None, has_mp=False):
    """Build one ``_read_dataset_metadata``-shaped result dict."""
    return {
        "key": key,
        "domain": domain,
        "tissue": tissue,
        "ct_names": set(ct_names) if ct_names is not None else None,
        "has_mp": has_mp,
    }


class TestAggregateMetadata:
    def test_domain_mapping_covers_every_dataset(self):
        results = [_meta("d1", domain="MIBI"), _meta("d2", domain="CODEX")]
        domain, _, _, _ = TissueNetConfig._aggregate_metadata(results, {})
        assert domain == {"d1": "MIBI", "d2": "CODEX"}

    def test_dataset_celltypes_lists_all_annotated_names_unfiltered(self):
        # "Weird" is not in ct2idx but must still be retained per-dataset.
        results = [_meta("d1", ct_names=["T_cell", "B_cell", "Weird"])]
        _, ds_ct, _, _ = TissueNetConfig._aggregate_metadata(results, {"T_cell": 0})
        assert ds_ct == {"d1": ["B_cell", "T_cell", "Weird"]}

    def test_dataset_without_annotations_is_absent(self):
        results = [_meta("d1", ct_names=None)]
        _, ds_ct, _, _ = TissueNetConfig._aggregate_metadata(results, {})
        assert ds_ct == {}

    def test_tissue_mapping_filters_to_ct2idx_and_aggregates(self):
        results = [
            _meta("d1", tissue="lung", ct_names=["T_cell", "Weird"]),
            _meta("d2", tissue="lung", ct_names=["B_cell"]),
        ]
        ct2idx = {"T_cell": 0, "B_cell": 1}
        _, _, tissue_ct, _ = TissueNetConfig._aggregate_metadata(results, ct2idx)
        # "Weird" dropped (not in ct2idx); d1+d2 merged for "lung"; sorted.
        assert tissue_ct == {"lung": ["B_cell", "T_cell"]}

    def test_tissue_with_only_unknown_cts_is_dropped(self):
        results = [_meta("d1", tissue="spleen", ct_names=["Weird"])]
        _, _, tissue_ct, _ = TissueNetConfig._aggregate_metadata(results, {"T_cell": 0})
        # Empty allowed-set tissue dropped to avoid an all-Inf logit mask.
        assert tissue_ct == {}

    def test_celltypes_recorded_but_no_tissue_yields_no_tissue_entry(self):
        results = [_meta("d1", tissue=None, ct_names=["T_cell"])]
        _, ds_ct, tissue_ct, _ = TissueNetConfig._aggregate_metadata(
            results, {"T_cell": 0}
        )
        assert ds_ct == {"d1": ["T_cell"]}
        assert tissue_ct == {}

    def test_mp_keys_collected_in_order(self):
        results = [
            _meta("d1", has_mp=True),
            _meta("d2", has_mp=False),
            _meta("d3", has_mp=True),
        ]
        _, _, _, mp = TissueNetConfig._aggregate_metadata(results, {})
        assert mp == ["d1", "d3"]


class TestBuildTissueMappingFromSplit:
    @staticmethod
    def _make_config(dataset_celltypes, ct2idx, tissue_of):
        cfg = object.__new__(TissueNetConfig)
        cfg._dataset_celltypes_cache = dataset_celltypes
        cfg._ct2idx = ct2idx
        # Instance attribute shadows the zarr-reading method.
        cfg.get_tissue_for_dataset = lambda ds_key: tissue_of.get(ds_key)
        return cfg

    def test_aggregates_train_datasets_by_tissue(self, tmp_path):
        cfg = self._make_config(
            dataset_celltypes={
                "d1": ["B_cell", "T_cell", "Weird"],
                "d2": ["Macrophage"],
                "d3": ["T_cell"],
            },
            ct2idx={"T_cell": 0, "B_cell": 1, "Macrophage": 2},
            tissue_of={"d1": "lung", "d2": "lung", "d3": "spleen"},
        )
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"train": {"d1": [], "d2": [], "d3": []}}))
        out = cfg.build_tissue_mapping_from_split(str(split))
        assert out == {
            "lung": ["B_cell", "Macrophage", "T_cell"],
            "spleen": ["T_cell"],
        }

    def test_dataset_with_no_tissue_is_skipped(self, tmp_path):
        cfg = self._make_config(
            dataset_celltypes={"d1": ["T_cell"]},
            ct2idx={"T_cell": 0},
            tissue_of={"d1": None},
        )
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"train": {"d1": []}}))
        assert cfg.build_tissue_mapping_from_split(str(split)) == {}
