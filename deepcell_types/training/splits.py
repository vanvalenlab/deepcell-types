"""FOV split generation, stratification, and the per-cell index record.

Extracted from ``deepcell_types.training.dataset`` for modularity. These
symbols are re-exported from ``dataset`` for backward compatibility.

``CellIndexRecord`` lives here (rather than in ``dataset``) so that both the
core dataset and the split/stratification helpers can share it without a
circular import — ``dataset`` imports it from this module, and this module's
functions only take a ``dataset`` instance as a parameter (no import of
``dataset`` itself).
"""

import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Tuple

logger = logging.getLogger(__name__)

# Provenance fields that are *not* treated as strict invariants when
# loading a split file. ``zarr_path`` is intentionally portable across
# mount points and symlinks; mismatches are reported but never raise.
# ``archive_fingerprint`` is a whole-archive metadata hash, so purely additive
# growth (appending new datasets/FOVs) changes it even when every FOV named in
# the split is byte-for-byte unchanged. The per-FOV roster checks in
# ``load_fov_splits`` already raise on destructive drift (a split FOV missing
# from the live archive), so the fingerprint is advisory rather than strict.
# Every other field in ``_split_metadata_for_dataset`` is strict.
_ADVISORY_SPLIT_METADATA_KEYS = {
    "zarr_path",
    "archive_fingerprint",
}


class CellIndexRecord(NamedTuple):
    """One per-cell entry in ``FullImageDataset.indices``.

    Replaces a positional 8-tuple. Pickles compactly (NamedTuple is
    serialized as a regular tuple), so existing cell-data caches that
    stored raw 8-tuples still deserialize correctly — and code that
    treats this as a tuple (indexing by integer, unpacking by position)
    continues to work too. The named accessors prevent the
    "positional magic number" footgun called out by complexity H8.

    Field 5 (``fov_name``) and field 6 (``dataset_name``) both currently
    hold ``dataset_key`` because the v8 archive layout encodes the FOV
    path in the dataset key itself; the two attributes are kept distinct
    so that downstream code can later differentiate them without another
    rename.
    """

    ds_idx: int
    ct_label: str
    ct_label_standard: str
    domain: str
    cell_idx: int
    fov_name: str
    dataset_name: str
    centroid: Tuple[float, ...]


def _find_sole_source_fovs(dataset, fov_to_indices):
    """Find FOVs that are the sole source of a cell type class.

    These FOVs must go to train so the model can learn rare classes.
    Without this, single-FOV classes randomly land in val, creating
    classes with 0 train support (guaranteed 0% accuracy on those cells).

    Returns:
        set of fov_keys that must be in train
    """
    # Map: class -> set of FOV keys containing it
    class_to_fovs = defaultdict(set)
    for fov_key, indices in fov_to_indices.items():
        fov_classes = set()
        for idx in indices:
            ct_label = dataset.indices[idx].ct_label_standard
            fov_classes.add(ct_label)
        for ct in fov_classes:
            class_to_fovs[ct].add(fov_key)

    # FOVs that are the only source of a class
    forced_train = set()
    for ct, fovs in class_to_fovs.items():
        if len(fovs) == 1:
            forced_train.update(fovs)

    if forced_train:
        forced_classes = [ct for ct, fovs in class_to_fovs.items() if len(fovs) == 1]
        logger.info(
            "Rare-class stratification: %d FOVs forced to train "
            "(sole source of %d classes: %s)",
            len(forced_train),
            len(forced_classes),
            sorted(forced_classes),
        )

    return forced_train


def _build_fov_strata(dataset, fov_to_indices, stratify_by):
    """For each fov_key, compute its stratum tuple from `stratify_by` keys.

    A stratum is the (modality, tissue) bucket a FOV belongs to. Callers
    force single-FOV strata to train (cannot evaluate a held-out FOV from a
    bucket that has only one) and split multi-FOV strata at train_ratio.

    Returns:
        dict mapping fov_key -> stratum tuple (e.g. ("mibi", "lymphnode"))
    """
    fov_to_stratum = {}
    for fov_key, idxs in fov_to_indices.items():
        sample_i = idxs[0]
        record = dataset.indices[sample_i]
        ds_idx = record.ds_idx
        modality = record.domain
        zf_entry = dataset.zarr_files[ds_idx]
        tissue = zf_entry.get("tissue", "")
        parts = []
        for key in stratify_by:
            if key == "modality":
                parts.append(modality)
            elif key == "tissue":
                parts.append(tissue)
            else:
                raise ValueError(f"Unsupported stratify_by key: {key}")
        fov_to_stratum[fov_key] = tuple(parts)
    return fov_to_stratum


def parse_patient_id(fov_key):
    """Best-effort patient/donor id parsed from a ``fov_key`` naming convention.

    There is no structured patient/donor metadata in the archive (only
    ``tissue``/``modality`` zarr attrs, see ``_build_fov_strata``) — patient
    identity, where recoverable at all, is embedded in the dataset/FOV name
    string itself. This currently only recognizes the known
    ``...Patient<N>...`` convention used by e.g. the McCaffrey TB MIBI
    dataset (``mccaffrey_tb_mibi_lung_Patient3-2`` -> patient id
    ``"mccaffrey_tb_mibi_lung_Patient3"``).

    The returned id is anchored to everything up to and including
    ``Patient<N>`` (not just the bare number), so that two unrelated source
    datasets that both happen to have a "Patient3" are never merged into the
    same group.

    Args:
        fov_key: (dataset_name, fov_name) tuple, as used elsewhere in this
            module.

    Returns:
        The parsed patient id string, or None if neither field matches the
        known convention (caller should fall back to per-FOV handling).
    """
    for candidate in fov_key:
        match = re.search(r"^.*?Patient\d+", str(candidate))
        if match:
            return match.group(0)
    return None


def _group_fov_keys_by_patient(fov_keys):
    """Group ``fov_keys`` sharing a ``parse_patient_id`` result into units.

    Returns a list of "units", where each unit is a list of one or more
    fov_keys that must be assigned to the same split (train xor val) as a
    whole. FOVs with no parseable patient id become their own singleton
    unit — identical to the ungrouped, per-FOV behavior.
    """
    patient_to_unit = defaultdict(list)
    singleton_units = []
    for fov_key in fov_keys:
        patient_id = parse_patient_id(fov_key)
        if patient_id is None:
            singleton_units.append([fov_key])
        else:
            patient_to_unit[patient_id].append(fov_key)
    return list(patient_to_unit.values()) + singleton_units


def create_fov_splits(
    dataset, train_ratio=0.8, seed=42, stratify_by=(), group_by_patient=False
):
    """Split dataset by FOV (no spatial leakage).

    Groups cells by FOV, then assigns entire FOVs to train or val.
    FOVs that are the sole source of a cell type class are forced into
    train to prevent classes with 0 train support.

    When ``stratify_by`` is non-empty (e.g. ``("modality", "tissue")``), the
    remaining FOVs are bucketed by that key tuple and the train/val split is
    applied within each bucket. Single-FOV buckets are forced to train, since
    a single FOV cannot support both train and val.

    Args:
        dataset: FullImageDataset instance
        train_ratio: Fraction of FOVs for training
        seed: Random seed
        stratify_by: Tuple of stratification keys, e.g. ("modality", "tissue").
            Empty tuple disables stratification (legacy global shuffle).
        group_by_patient: If True, FOVs whose key matches the known
            ``...Patient<N>...`` naming convention (see ``parse_patient_id``)
            are grouped so every FOV from the same patient lands entirely in
            train or entirely in val, instead of being shuffled
            independently — prevents patient-level leakage across the split.
            FOVs with no parseable patient id keep the per-FOV behavior.
            Only applies within the ``stratify_by`` per-stratum split (a
            no-op when ``stratify_by`` is empty). Default False: existing
            callers and split files are unaffected unless this is
            explicitly requested.

    Returns:
        train_indices: List of integer indices into dataset
        val_indices: List of integer indices into dataset
    """
    rng = random.Random(seed)

    # Group indices by (dataset_name, fov_name)
    fov_to_indices = defaultdict(list)
    for i, idx_tuple in enumerate(dataset.indices):
        fov_key = (idx_tuple[6], idx_tuple[5])  # (dataset_name, fov_name)
        fov_to_indices[fov_key].append(i)

    # Force sole-source FOVs into train
    forced_train_fovs = _find_sole_source_fovs(dataset, fov_to_indices)

    train_indices = []
    val_indices = []
    for fov_key in forced_train_fovs:
        train_indices.extend(fov_to_indices[fov_key])

    remaining_fov_keys = sorted(
        k for k in fov_to_indices.keys() if k not in forced_train_fovs
    )

    if stratify_by:
        # Per-stratum split: ensures (modality, tissue) buckets with ≥2 FOVs
        # have both train and val coverage. Single-FOV buckets go to train.
        fov_to_stratum = _build_fov_strata(dataset, fov_to_indices, stratify_by)
        by_stratum = defaultdict(list)
        for fov_key in remaining_fov_keys:
            by_stratum[fov_to_stratum[fov_key]].append(fov_key)
        single_fov_strata = []
        for stratum in sorted(by_stratum.keys()):
            keys = list(by_stratum[stratum])
            # Collapse FOVs sharing a parsed patient id into single "units"
            # that are assigned to a split as a whole (see
            # _group_fov_keys_by_patient). When group_by_patient=False every
            # unit is a singleton [fov_key] — identical to the old,
            # ungrouped per-FOV list — so behavior is unchanged by default.
            units = (
                _group_fov_keys_by_patient(keys)
                if group_by_patient
                else [[fk] for fk in keys]
            )
            rng.shuffle(units)
            if len(units) == 1:
                # Whole stratum is one unsplittable unit — either a single
                # FOV, or (with group_by_patient) a single patient group
                # spanning every remaining FOV in this stratum. Can't
                # evaluate a held-out portion either way, so force to train
                # (same rule as the pre-existing single-FOV case).
                single_fov_strata.append(stratum)
                for fk in units[0]:
                    train_indices.extend(fov_to_indices[fk])
                continue
            # Round to nearest, clamp so neither side is empty. Ratio is
            # computed over units, not raw FOV counts: with group_by_patient
            # this trades exact train_ratio adherence for never splitting a
            # patient across train/val.
            n_train = max(1, min(len(units) - 1, int(round(len(units) * train_ratio))))
            for unit in units[:n_train]:
                for fk in unit:
                    train_indices.extend(fov_to_indices[fk])
            for unit in units[n_train:]:
                for fk in unit:
                    val_indices.extend(fov_to_indices[fk])
        if single_fov_strata:
            logger.info(
                "stratified split: %d single-FOV strata forced to train (cannot eval): %s",
                len(single_fov_strata),
                single_fov_strata,
            )
        return train_indices, val_indices

    # Legacy non-stratified path (global random shuffle)
    rng.shuffle(remaining_fov_keys)
    target_train = int(len(fov_to_indices) * train_ratio)
    requested_remaining = target_train - len(forced_train_fovs)
    n_train_remaining = max(0, requested_remaining)
    if requested_remaining < 0:
        logger.warning(
            "rare-class forcing clamp: %d FOVs forced to train exceeds target %d "
            "(requested %d remaining, satisfied 0); train pool saturated by sole-source FOVs",
            len(forced_train_fovs),
            target_train,
            requested_remaining,
        )

    for fov_key in remaining_fov_keys[:n_train_remaining]:
        train_indices.extend(fov_to_indices[fov_key])
    for fov_key in remaining_fov_keys[n_train_remaining:]:
        val_indices.extend(fov_to_indices[fov_key])

    return train_indices, val_indices


def _split_metadata_for_dataset(dataset):
    """Return provenance fields that make split reuse auditable."""
    marker2idx = getattr(dataset, "marker2idx", {})
    ct2idx = getattr(dataset, "ct2idx", {})
    zarr_path = getattr(dataset, "_zarr_path", None)

    return {
        "max_channels": getattr(dataset, "max_channels", None),
        "num_marker_channels": len(marker2idx) if marker2idx is not None else None,
        "num_cell_types": len(ct2idx) if ct2idx is not None else None,
        # Kept for auditability only. Split files must remain portable across
        # mount points and symlinks, so load_fov_splits never treats this as
        # strict provenance.
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        # Advisory: a whole-archive hash that drifts under additive growth even
        # when the split's FOVs are unchanged (see _ADVISORY_SPLIT_METADATA_KEYS).
        "archive_fingerprint": getattr(dataset, "archive_fingerprint", None),
    }


def _format_fov_examples(fov_keys, limit=5):
    examples = sorted(fov_keys)[:limit]
    suffix = "" if len(fov_keys) <= limit else f", ... (+{len(fov_keys) - limit} more)"
    return ", ".join(f"{ds}/{fov}" for ds, fov in examples) + suffix


def save_fov_splits(
    dataset,
    split_file,
    train_ratio=0.8,
    seed=42,
    stratify_by=(),
    group_by_patient=False,
):
    """Generate FOV splits and save to a JSON file for reproducibility.

    Delegates to ``create_fov_splits`` for the actual partitioning logic
    (rare-class sole-source forcing, optional stratification, train/val
    bucket assignment) and adds JSON serialization with metadata.

    Args:
        dataset: FullImageDataset instance.
        split_file: Path to write the JSON split file.
        train_ratio: Fraction of FOVs for training.
        seed: Random seed.
        stratify_by: Tuple of stratification keys, e.g. ``("modality", "tissue")``.
            Empty tuple disables stratification (legacy global shuffle, used by
            v9 splits for benchmark continuity).
        group_by_patient: See ``create_fov_splits``. Default False (no change
            to existing split files unless explicitly requested).

    Returns:
        train_indices, val_indices (same as ``create_fov_splits``).
    """
    train_indices, val_indices = create_fov_splits(
        dataset,
        train_ratio=train_ratio,
        seed=seed,
        stratify_by=stratify_by,
        group_by_patient=group_by_patient,
    )

    # Reconstruct (dataset_name -> [fov_names]) groupings from the per-cell
    # index lists. Each integer in train_indices/val_indices points at a row
    # of dataset.indices, whose fields 6 and 5 are dataset_name and fov_name.
    train_split: dict = {}
    val_split: dict = {}
    train_fov_keys: set = set()
    for i in train_indices:
        record = dataset.indices[i]
        fov_key = (record.dataset_name, record.fov_name)
        if fov_key in train_fov_keys:
            continue
        train_fov_keys.add(fov_key)
        train_split.setdefault(fov_key[0], []).append(fov_key[1])
    val_fov_keys: set = set()
    for i in val_indices:
        record = dataset.indices[i]
        fov_key = (record.dataset_name, record.fov_name)
        if fov_key in val_fov_keys:
            continue
        val_fov_keys.add(fov_key)
        val_split.setdefault(fov_key[0], []).append(fov_key[1])

    # Re-derive single-unit stratum count for metadata transparency. A "unit"
    # is a single FOV, or (group_by_patient=True) a whole patient group — see
    # create_fov_splits. Mirrors that function's own single-unit forced-to-
    # train rule so this count stays accurate whether or not grouping is on.
    num_single_fov_strata = 0
    if stratify_by:
        fov_to_indices = defaultdict(list)
        for i, record in enumerate(dataset.indices):
            fov_to_indices[(record.dataset_name, record.fov_name)].append(i)
        forced = _find_sole_source_fovs(dataset, fov_to_indices)
        fov_to_stratum = _build_fov_strata(dataset, fov_to_indices, stratify_by)
        by_stratum: dict = defaultdict(list)
        for fov_key in fov_to_indices:
            if fov_key in forced:
                continue
            by_stratum[fov_to_stratum[fov_key]].append(fov_key)
        for stratum_keys in by_stratum.values():
            n_units = (
                len(_group_fov_keys_by_patient(stratum_keys))
                if group_by_patient
                else len(stratum_keys)
            )
            if n_units == 1:
                num_single_fov_strata += 1

    split_data = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "stratify_by": list(stratify_by),
            "group_by_patient": bool(group_by_patient),
            "num_train_fovs": sum(len(v) for v in train_split.values()),
            "num_val_fovs": sum(len(v) for v in val_split.values()),
            "num_datasets": len(set(list(train_split) + list(val_split))),
            "num_single_fov_strata_forced_to_train": num_single_fov_strata,
            "created": datetime.now(timezone.utc).isoformat(),
            **_split_metadata_for_dataset(dataset),
        },
        "train": train_split,
        "val": val_split,
    }

    split_path = Path(split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    logger.info("FOV splits saved to %s", split_path)
    return train_indices, val_indices


def load_fov_splits(dataset, split_file, *, strict=True):
    """Load pre-computed FOV splits from a JSON file.

    Args:
        dataset: FullImageDataset instance.
        split_file: Path to the JSON split file.
        strict: If True (default), raise ValueError on overlap between train/
            val/heldout, FOVs in the JSON missing from the live archive, or
            FOVs in the live archive missing from the JSON. If False, log a
            warning and continue with whatever overlap is resolvable.

    Returns:
        train_indices: List of integer indices into dataset.
        val_indices: List of integer indices into dataset.
    """
    with open(split_file) as f:
        split_data = json.load(f)

    train_fovs_by_ds = split_data["train"]
    val_fovs_by_ds = split_data["val"]

    # Build lookup: (dataset_name, fov_name) -> set membership
    train_fov_set = set()
    for ds_name, fov_names in train_fovs_by_ds.items():
        for fov_name in fov_names:
            train_fov_set.add((ds_name, fov_name))

    val_fov_set = set()
    for ds_name, fov_names in val_fovs_by_ds.items():
        for fov_name in fov_names:
            val_fov_set.add((ds_name, fov_name))

    heldout_fov_set = set()
    for ds_name, fov_names in split_data.get("heldout", {}).items():
        for fov_name in fov_names:
            heldout_fov_set.add((ds_name, fov_name))

    train_indices = []
    val_indices = []
    skipped = 0
    heldout = 0
    dataset_fov_set = set()

    for i, idx_tuple in enumerate(dataset.indices):
        fov_key = (idx_tuple[6], idx_tuple[5])  # (dataset_name, fov_name)
        dataset_fov_set.add(fov_key)
        if fov_key in train_fov_set:
            train_indices.append(i)
        elif fov_key in val_fov_set:
            val_indices.append(i)
        elif fov_key in heldout_fov_set:
            heldout += 1
        else:
            skipped += 1

    if heldout > 0:
        logger.info("%d samples intentionally held out by split file", heldout)

    if skipped > 0:
        msg = f"{skipped} samples not found in split file (dataset/FOV mismatch)"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    train_val_overlap = train_fov_set & val_fov_set
    if train_val_overlap:
        msg = (
            f"{len(train_val_overlap)} FOVs appear in both train and val splits: "
            f"{_format_fov_examples(train_val_overlap)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    heldout_overlap = heldout_fov_set & (train_fov_set | val_fov_set)
    if heldout_overlap:
        msg = (
            f"{len(heldout_overlap)} heldout FOVs also appear in train/val: "
            f"{_format_fov_examples(heldout_overlap)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    missing_split_fovs = (train_fov_set | val_fov_set) - dataset_fov_set
    if missing_split_fovs:
        msg = (
            f"{len(missing_split_fovs)} split FOVs are not present in the current "
            f"dataset after filters: {_format_fov_examples(missing_split_fovs)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    missing_heldout_fovs = heldout_fov_set - dataset_fov_set
    if missing_heldout_fovs:
        msg = (
            f"{len(missing_heldout_fovs)} heldout FOVs are not present in the "
            f"current dataset after filters: {_format_fov_examples(missing_heldout_fovs)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    meta = split_data.get("metadata", {})
    current_meta = _split_metadata_for_dataset(dataset)
    metadata_mismatches = []
    for key, current_value in current_meta.items():
        saved_value = meta.get(key)
        if current_value is None:
            continue
        if saved_value is None:
            msg = f"{key}: file is missing, current dataset has {current_value!r}"
            if key in _ADVISORY_SPLIT_METADATA_KEYS:
                logger.warning("split metadata missing advisory %s", msg)
            else:
                metadata_mismatches.append(msg)
        elif saved_value != current_value:
            msg = f"{key}: file has {saved_value!r}, current dataset has {current_value!r}"
            if key in _ADVISORY_SPLIT_METADATA_KEYS:
                logger.warning("split metadata advisory mismatch: %s", msg)
            else:
                metadata_mismatches.append(msg)

    if metadata_mismatches:
        msg = "split metadata mismatch: " + "; ".join(metadata_mismatches)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    logger.info(
        "Loaded FOV splits from %s (created %s): %d train, %d val",
        split_file,
        meta.get("created", "unknown"),
        len(train_indices),
        len(val_indices),
    )

    return train_indices, val_indices
