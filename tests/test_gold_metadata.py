"""Tests for ``deepcell_types.training.gold_metadata.resolve_gold_metadata``.

Covers: known-key resolution, unknown-key errors, the ``strict`` refusal of
non-canonical substitutions, and the ``dct_config`` vocab-validation branch.
"""

import pytest

from deepcell_types.training.gold_metadata import (
    GOLD_DATASET_METADATA,
    NONCANONICAL_DATASETS,
    resolve_gold_metadata,
)


class _ConfigStub:
    """Minimal stand-in for ``TissueNetConfig``: only ``tissue2idx`` /
    ``domain2idx`` are read by ``resolve_gold_metadata``."""

    def __init__(self, tissue2idx=None, domain2idx=None):
        self.tissue2idx = {} if tissue2idx is None else tissue2idx
        self.domain2idx = {} if domain2idx is None else domain2idx


# --- known-key resolution ---------------------------------------------------


@pytest.mark.parametrize("dataset_key", sorted(GOLD_DATASET_METADATA))
def test_known_key_resolves_to_canonical_tuple(dataset_key):
    meta = GOLD_DATASET_METADATA[dataset_key]
    tissue, modality = resolve_gold_metadata(dataset_key)
    assert (tissue, modality) == (meta["tissue_canonical"], meta["modality_canonical"])


def test_all_five_pan_m_subsets_are_registered():
    # Pin the known Pan-M Gold-Standard subset roster (per the module docstring).
    assert set(GOLD_DATASET_METADATA) == {
        "codex_colon",
        "mibi_breast",
        "mibi_decidua",
        "vectra_colon",
        "vectra_pancreas",
    }


# --- unknown key -------------------------------------------------------------


def test_unknown_dataset_key_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_gold_metadata("not_a_real_dataset")


def test_unknown_dataset_key_error_lists_valid_keys():
    with pytest.raises(KeyError, match="not_a_real_dataset"):
        resolve_gold_metadata("not_a_real_dataset")


# --- strict=True refuses non-canonical substitutions ------------------------


def test_noncanonical_dataset_roster():
    # Pin the actual NONCANONICAL_DATASETS names this test parametrizes over.
    assert NONCANONICAL_DATASETS == frozenset(
        {"mibi_decidua", "vectra_colon", "vectra_pancreas"}
    )


@pytest.mark.parametrize("dataset_key", sorted(NONCANONICAL_DATASETS))
def test_strict_raises_for_noncanonical_dataset(dataset_key):
    with pytest.raises(ValueError, match="non-direct"):
        resolve_gold_metadata(dataset_key, strict=True)


@pytest.mark.parametrize(
    "dataset_key", sorted(set(GOLD_DATASET_METADATA) - NONCANONICAL_DATASETS)
)
def test_strict_does_not_raise_for_canonical_dataset(dataset_key):
    meta = GOLD_DATASET_METADATA[dataset_key]
    tissue, modality = resolve_gold_metadata(dataset_key, strict=True)
    assert (tissue, modality) == (meta["tissue_canonical"], meta["modality_canonical"])


# --- strict=False (default) accepts the substitution ------------------------


@pytest.mark.parametrize("dataset_key", sorted(NONCANONICAL_DATASETS))
def test_non_strict_returns_substituted_tuple(dataset_key):
    meta = GOLD_DATASET_METADATA[dataset_key]
    tissue, modality = resolve_gold_metadata(dataset_key, strict=False)
    assert (tissue, modality) == (meta["tissue_canonical"], meta["modality_canonical"])


def test_default_strict_is_false():
    # codex_colon is canonical either way; use a noncanonical key to prove the
    # default doesn't raise (i.e. strict defaults to False).
    tissue, modality = resolve_gold_metadata("mibi_decidua")
    assert (tissue, modality) == ("uterus", "mibi")


# --- dct_config vocab-validation branch -------------------------------------


def test_dct_config_none_skips_validation():
    # No dct_config -> no vocab check performed, resolution still succeeds.
    tissue, modality = resolve_gold_metadata("codex_colon", dct_config=None)
    assert (tissue, modality) == ("colon", "codex")


def test_dct_config_validation_passes_when_vocab_contains_names():
    config = _ConfigStub(tissue2idx={"colon": 0}, domain2idx={"codex": 0})
    tissue, modality = resolve_gold_metadata("codex_colon", dct_config=config)
    assert (tissue, modality) == ("colon", "codex")


def test_dct_config_validation_raises_when_tissue_missing():
    config = _ConfigStub(tissue2idx={}, domain2idx={"codex": 0})
    with pytest.raises(ValueError, match="tissue2idx"):
        resolve_gold_metadata("codex_colon", dct_config=config)


def test_dct_config_validation_raises_when_modality_missing():
    config = _ConfigStub(tissue2idx={"colon": 0}, domain2idx={})
    with pytest.raises(ValueError, match="domain2idx"):
        resolve_gold_metadata("codex_colon", dct_config=config)


def test_dct_config_validation_checks_tissue_before_modality():
    # Both missing -> the tissue check fires first (raises on tissue, not modality).
    config = _ConfigStub(tissue2idx={}, domain2idx={})
    with pytest.raises(ValueError, match="Resolved tissue"):
        resolve_gold_metadata("codex_colon", dct_config=config)


def test_dct_config_without_tissue2idx_attr_treated_as_empty():
    # getattr(..., "tissue2idx", {}) fallback: an object missing the attribute
    # entirely behaves like an empty vocab, not a crash.
    class _BareObject:
        pass

    with pytest.raises(ValueError, match="tissue2idx"):
        resolve_gold_metadata("codex_colon", dct_config=_BareObject())


def test_strict_refusal_takes_priority_over_vocab_validation():
    # strict=True raises the "non-direct canonicalization" error before ever
    # consulting dct_config, even if the config would also fail vocab checks.
    config = _ConfigStub(tissue2idx={}, domain2idx={})
    with pytest.raises(ValueError, match="non-direct"):
        resolve_gold_metadata("mibi_decidua", dct_config=config, strict=True)
