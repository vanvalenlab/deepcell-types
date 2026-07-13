"""Unit tests for group_by_patient (deepcell_types/training/splits.py).

Bug this guards against: create_fov_splits() stratifies only by
(modality, tissue) and shuffles FOVs within a stratum independently, so a
single patient's FOVs can straddle train and val. Verified in
splits/fov_split_valsubset.json: mccaffrey_tb_mibi Patient2/Patient10/
Patient13 all appear in both train and val.

parse_patient_id()/group_by_patient=True add an OPTIONAL grouping mechanism
(default off, so existing split files/callers are unaffected) that keeps
every FOV sharing a parsed "...Patient<N>..." id on the same side of the
split.
"""
from deepcell_types.training.dataset import CellIndexRecord, create_fov_splits
from deepcell_types.training.splits import parse_patient_id


class _MockDatasetWithStrata:
    def __init__(self, indices, zarr_files):
        self.indices = indices
        self.zarr_files = zarr_files


def _build_dataset(fov_specs):
    """fov_specs: list of (fov_key, modality, tissue, ct_label) tuples.

    Each entry creates 1 cell in 1 FOV whose dataset_name/fov_name are both
    `fov_key` — mirrors the real archive layout (see CellIndexRecord
    docstring: the v8 layout encodes the FOV path in the dataset key).
    """
    indices = []
    zarr_files = []
    for i, (fov_key, modality, tissue, ct_label) in enumerate(fov_specs):
        indices.append(
            CellIndexRecord(i, ct_label, ct_label, modality, i, fov_key, fov_key, (10, 10))
        )
        zarr_files.append(
            {
                "name": fov_key,
                "channel_names": ["DAPI"],
                "dataset_key": fov_key,
                "tissue": tissue,
                "modality": modality,
            }
        )
    return _MockDatasetWithStrata(indices, zarr_files)


def _make_patient_fovs(prefix, patient_ns, fovs_per_patient, modality, tissue, ct_label):
    """Build fov_specs for `fovs_per_patient` FOVs each, for patients in patient_ns."""
    specs = []
    for p in patient_ns:
        for j in range(fovs_per_patient):
            specs.append((f"{prefix}_Patient{p}-{j}", modality, tissue, ct_label))
    return specs


class TestParsePatientId:
    def test_known_convention_parsed(self):
        fov_key = ("mccaffrey_tb_mibi_lung_Patient3-2", "mccaffrey_tb_mibi_lung_Patient3-2")
        assert parse_patient_id(fov_key) == "mccaffrey_tb_mibi_lung_Patient3"

    def test_different_source_datasets_not_merged(self):
        """Two unrelated datasets both having a 'Patient3' must not collide."""
        a = parse_patient_id(("datasetA_Patient3-1", "datasetA_Patient3-1"))
        b = parse_patient_id(("datasetB_Patient3-1", "datasetB_Patient3-1"))
        assert a != b

    def test_no_match_returns_none(self):
        fov_key = ("dryadb004_intestine_codex_B004_CL_reg001", "dryadb004_intestine_codex_B004_CL_reg001")
        assert parse_patient_id(fov_key) is None


class TestGroupByPatientNoLeakage:
    def test_shared_patient_fovs_land_on_one_side(self):
        """Several FOVs sharing a patient id, within the same stratum: with
        grouping enabled, no patient id appears in both train and val.
        """
        # 6 patients x 2 FOVs = 12 FOVs, all (MIBI, lung).
        fov_specs = _make_patient_fovs(
            "mccaffrey_tb_mibi_lung", range(1, 7), 2, "MIBI", "lung", "CD4T"
        )
        ds = _build_dataset(fov_specs)
        # Break sole-source forcing: add a second class present in every FOV.
        ds.indices.extend(
            CellIndexRecord(i, "Bcell", "Bcell", "MIBI", 1000 + i, spec[0], spec[0], (5, 5))
            for i, spec in enumerate(fov_specs)
        )

        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
            group_by_patient=True,
        )
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}

        train_patients = {parse_patient_id((f, f)) for f in train_fovs}
        val_patients = {parse_patient_id((f, f)) for f in val_fovs}

        assert not (train_patients & val_patients), (
            f"patient leaked across split: {train_patients & val_patients}"
        )
        # Sanity: both sides are non-empty (train_ratio=0.5, 6 patient units).
        assert train_fovs
        assert val_fovs

    def test_every_fov_of_a_patient_stays_together(self):
        """A patient with 3 FOVs must have all 3 (not a subset) on one side."""
        fov_specs = _make_patient_fovs(
            "ds_lung", [1, 2], 3, "MIBI", "lung", "CD4T"
        )
        ds = _build_dataset(fov_specs)
        ds.indices.extend(
            CellIndexRecord(i, "Bcell", "Bcell", "MIBI", 1000 + i, spec[0], spec[0], (5, 5))
            for i, spec in enumerate(fov_specs)
        )

        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.5, seed=1, stratify_by=("modality", "tissue"),
            group_by_patient=True,
        )
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}

        patient1_fovs = {f"ds_lung_Patient1-{j}" for j in range(3)}
        patient2_fovs = {f"ds_lung_Patient2-{j}" for j in range(3)}

        for patient_fovs in (patient1_fovs, patient2_fovs):
            in_train = patient_fovs & train_fovs
            in_val = patient_fovs & val_fovs
            assert in_train == patient_fovs or in_val == patient_fovs, (
                f"patient FOVs split across train/val: train={in_train}, val={in_val}"
            )


class TestGroupByPatientBackCompat:
    def test_disabled_matches_ungrouped_behavior(self):
        """group_by_patient=False (the default) must reproduce the exact
        same split as calling create_fov_splits without the parameter at
        all — i.e. this PR does not change any existing split file.
        """
        fov_specs = _make_patient_fovs(
            "mccaffrey_tb_mibi_lung", range(1, 5), 2, "MIBI", "lung", "CD4T"
        )
        ds = _build_dataset(fov_specs)
        ds.indices.extend(
            CellIndexRecord(i, "Bcell", "Bcell", "MIBI", 1000 + i, spec[0], spec[0], (5, 5))
            for i, spec in enumerate(fov_specs)
        )

        train_default, val_default = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
        )
        train_explicit_off, val_explicit_off = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
            group_by_patient=False,
        )

        assert train_default == train_explicit_off
        assert val_default == val_explicit_off

    def test_disabled_can_still_split_a_single_patient_across_train_and_val(self):
        """Without grouping, a patient's FOVs are free to land on both
        sides (the pre-existing, leaky behavior) — confirms the default
        truly leaves old behavior alone rather than always-grouping.
        """
        fov_specs = _make_patient_fovs(
            "ds_lung", range(1, 9), 2, "MIBI", "lung", "CD4T"
        )
        ds = _build_dataset(fov_specs)
        ds.indices.extend(
            CellIndexRecord(i, "Bcell", "Bcell", "MIBI", 1000 + i, spec[0], spec[0], (5, 5))
            for i, spec in enumerate(fov_specs)
        )

        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
            group_by_patient=False,
        )
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}
        train_patients = {parse_patient_id((f, f)) for f in train_fovs}
        val_patients = {parse_patient_id((f, f)) for f in val_fovs}

        # With 8 patients x 2 FOVs shuffled per-FOV (not per-patient) at
        # seed=0, at least one patient should straddle the split — this
        # pins the pre-fix behavior so the "off" path is verified distinct
        # from the "on" path above, not just structurally identical code.
        assert train_patients & val_patients
