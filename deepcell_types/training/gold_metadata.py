"""Nimbus Pan-M Gold-Standard dataset → (tissue, modality) lookup.

The Pan-Multiplex Gold-Standard set (Greenwald et al., *Nat Methods* 2025;
DOI 10.1038/s41592-025-02826-9) ships 5 subsets named after their imaging
platform and tissue-of-origin. Each top-level folder is one (modality,
tissue, source-study) — there is no FOV-level tissue variation.

Two of the five names use vocabulary outside our training-archive's
canonical {tissue, modality} sets, so a defensible canonicalization is
required to run our trained model on them. The choices below are
documented inline; pass ``strict=True`` to ``resolve_gold_metadata`` to
refuse the canonicalization and raise instead.

Sources
-------
- Nimbus paper:                  https://www.nature.com/articles/s41592-025-02826-9
- Pan-M Gold-Standard on HF:     https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard
- Greenbaum 2023 (decidua src):  https://www.nature.com/articles/s41586-023-06298-9
- Lin 2018 CyCIF:                https://elifesciences.org/articles/31657
- Lu 2019 Vectra/Opal mIHC:      https://jamanetwork.com/journals/jamaoncology/fullarticle/2733805
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# Verbatim labels from the Nimbus paper / Pan-M HuggingFace README, plus a
# canonical mapping into our archive's vocab. ``modality_canonical`` and
# ``tissue_canonical`` are the names found in the trained model's vocab;
# ``modality_paper`` and ``tissue_paper`` preserve the source terminology
# for audit purposes.
GOLD_DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "codex_colon": {
        "modality_paper": "CODEX",
        "tissue_paper": "colon",
        "modality_canonical": "codex",
        "tissue_canonical": "colon",
    },
    "mibi_breast": {
        "modality_paper": "MIBI-TOF",
        "tissue_paper": "breast",
        "modality_canonical": "mibi",
        "tissue_canonical": "breast",
    },
    "mibi_decidua": {
        "modality_paper": "MIBI-TOF",
        # Greenbaum et al. *Nature* 2023 imaged human decidua basalis and
        # parietalis — the gestational endometrium of the maternal uterus.
        # Decidua is anatomically uterine mucosa during pregnancy, so
        # uterus is the correct canonical tissue (vs the broader
        # "reproductive" umbrella, which loses specificity).
        "tissue_paper": "decidua (maternal-fetal interface)",
        "modality_canonical": "mibi",
        "tissue_canonical": "uterus",
    },
    "vectra_colon": {
        # Akoya Vectra (Polaris/3) uses Opal tyramide-signal-amplification
        # fluorophores in single-cycle multispectral imaging — fluorescent
        # IF, low-to-mid plex (~7). The closest training-vocab analog is
        # CyCIF (Lin et al. *eLife* 2018) — fluorescent IF, modest plex
        # per cycle. NB: our archive's `mihc` is mcmicro_wsi_tonsil_mihc
        # which uses a HEMATOXYLIN nuclear channel = chromogenic mIHC =
        # wrong signal domain for Vectra.
        "modality_paper": "Vectra (Akoya, Opal multispectral IF)",
        "tissue_paper": "colon",
        "modality_canonical": "cycif",
        "tissue_canonical": "colon",
    },
    "vectra_pancreas": {
        "modality_paper": "Vectra (Akoya, Opal multispectral IF)",
        "tissue_paper": "pancreas",
        "modality_canonical": "cycif",
        "tissue_canonical": "pancreas",
    },
}


# Tracks which gold datasets required a non-direct canonicalization. Used
# by ``resolve_gold_metadata(strict=True)`` to refuse a substitution.
NONCANONICAL_DATASETS = frozenset(
    {
        "mibi_decidua",   # decidua → uterus
        "vectra_colon",   # vectra (Opal IF) → cycif
        "vectra_pancreas",  # vectra (Opal IF) → cycif
    }
)


def resolve_gold_metadata(
    dataset_key: str,
    dct_config: Optional[Any] = None,
    strict: bool = False,
) -> Tuple[str, str]:
    """Return ``(tissue_canonical, modality_canonical)`` for a Pan-M dataset.

    Args:
        dataset_key: One of the keys in ``GOLD_DATASET_METADATA`` (e.g.
            ``"codex_colon"``).
        dct_config: Optional ``TissueNetConfig`` instance. When provided,
            the resolved canonical names are validated against
            ``dct_config.tissue2idx`` and ``dct_config.domain2idx``;
            unknown names raise ``ValueError``.
        strict: When ``True``, refuse to canonicalize datasets in
            ``NONCANONICAL_DATASETS`` (i.e. ``mibi_decidua``,
            ``vectra_colon``, ``vectra_pancreas``). Caller should pass
            ``strict=False`` (the default) only after explicitly
            acknowledging the substitution.

    Raises:
        KeyError: If ``dataset_key`` is not a known Pan-M subset.
        ValueError: If ``strict=True`` and the dataset requires a
            non-direct canonicalization, or if the resolved canonical
            names are not in the model's vocab.
    """
    try:
        meta = GOLD_DATASET_METADATA[dataset_key]
    except KeyError as exc:
        raise KeyError(
            f"{dataset_key!r} is not a known Pan-M Gold-Standard subset; "
            f"valid keys: {sorted(GOLD_DATASET_METADATA)}"
        ) from exc

    if strict and dataset_key in NONCANONICAL_DATASETS:
        raise ValueError(
            f"Pan-M subset {dataset_key!r} requires a non-direct "
            f"canonicalization "
            f"(paper: tissue={meta['tissue_paper']!r}, "
            f"modality={meta['modality_paper']!r}; "
            f"canonical: tissue={meta['tissue_canonical']!r}, "
            f"modality={meta['modality_canonical']!r}). "
            f"Pass strict=False to accept the substitution, or override "
            f"the metadata at the call site."
        )

    tissue = meta["tissue_canonical"]
    modality = meta["modality_canonical"]

    if dct_config is not None:
        if tissue not in getattr(dct_config, "tissue2idx", {}):
            raise ValueError(
                f"Resolved tissue {tissue!r} for {dataset_key!r} is not in "
                f"dct_config.tissue2idx; available: "
                f"{sorted(getattr(dct_config, 'tissue2idx', {}))}"
            )
        if modality not in getattr(dct_config, "domain2idx", {}):
            raise ValueError(
                f"Resolved modality {modality!r} for {dataset_key!r} is not "
                f"in dct_config.domain2idx; available: "
                f"{sorted(getattr(dct_config, 'domain2idx', {}))}"
            )

    return tissue, modality
