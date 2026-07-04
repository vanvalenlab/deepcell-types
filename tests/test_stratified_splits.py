"""Unit tests for (modality, tissue) stratified FOV splits.

An earlier global random shuffle let ``(codex, tonsil)`` end up val-only /
untrainable. The current stratified split forces every multi-FOV stratum
to have both train and val coverage and forces single-FOV strata to train.
"""
from deepcell_types.training.dataset import (
    CellIndexRecord,
    create_fov_splits,
    _build_fov_strata,
)


class _MockDatasetWithStrata:
    """Minimal stub providing both `indices` and `zarr_files` (needed by
    `_build_fov_strata`).

    indices tuple: (ds_idx, ct_label, ct_label_standard, domain, cell_idx,
                    fov_name, dataset_name, centroid)
    zarr_files entry: {"name", "channel_names", "dataset_key", "tissue", "modality"}
    """

    def __init__(self, indices, zarr_files):
        self.indices = indices
        self.zarr_files = zarr_files


def _build_dataset(fov_specs):
    """Construct a minimal dataset from a list of (ds_idx, modality, tissue, ct_label) tuples.

    Each entry creates 1 cell in 1 FOV named "<dataset>_<ds_idx>".
    Returns a _MockDatasetWithStrata.
    """
    indices = []
    zarr_files = []
    for ds_idx, modality, tissue, ct_label in fov_specs:
        ds_name = f"ds_{ds_idx}"
        fov_name = f"fov_{ds_idx}"
        indices.append(
            CellIndexRecord(ds_idx, ct_label, ct_label, modality, ds_idx, fov_name, ds_name, (10, 10))
        )
        zarr_files.append(
            {"name": ds_name, "channel_names": ["DAPI"],
             "dataset_key": ds_name, "tissue": tissue, "modality": modality}
        )
    return _MockDatasetWithStrata(indices, zarr_files)


class TestBuildFovStrata:
    def test_modality_only(self):
        ds = _build_dataset([
            (0, "MIBI", "lung", "CD4T"),
            (1, "MIBI", "skin", "CD4T"),
            (2, "CODEX", "lung", "CD4T"),
        ])
        fov_to_indices = {("ds_0", "fov_0"): [0], ("ds_1", "fov_1"): [1], ("ds_2", "fov_2"): [2]}
        strata = _build_fov_strata(ds, fov_to_indices, ("modality",))
        assert strata[("ds_0", "fov_0")] == ("MIBI",)
        assert strata[("ds_1", "fov_1")] == ("MIBI",)
        assert strata[("ds_2", "fov_2")] == ("CODEX",)

    def test_modality_and_tissue(self):
        ds = _build_dataset([
            (0, "MIBI", "lung", "CD4T"),
            (1, "MIBI", "skin", "CD4T"),
        ])
        fov_to_indices = {("ds_0", "fov_0"): [0], ("ds_1", "fov_1"): [1]}
        strata = _build_fov_strata(ds, fov_to_indices, ("modality", "tissue"))
        assert strata[("ds_0", "fov_0")] == ("MIBI", "lung")
        assert strata[("ds_1", "fov_1")] == ("MIBI", "skin")


class TestStratifiedSplit:
    def test_multi_fov_stratum_has_both_train_and_val(self):
        """Each (modality, tissue) bucket with >=2 FOVs must have train+val FOVs."""
        # 4 FOVs across 2 modalities, 2 each
        ds = _build_dataset([
            (0, "MIBI", "lung", "CD4T"),
            (1, "MIBI", "lung", "CD4T"),  # both MIBI/lung
            (2, "CODEX", "skin", "CD4T"),
            (3, "CODEX", "skin", "CD4T"),  # both CODEX/skin
        ])
        # Multiple cell types so no FOV is "sole source"
        ds.indices.extend([
            CellIndexRecord(0, "Bcell", "Bcell", "MIBI", 100, "fov_0", "ds_0", (10, 10)),
            CellIndexRecord(1, "Bcell", "Bcell", "MIBI", 101, "fov_1", "ds_1", (10, 10)),
            CellIndexRecord(2, "Bcell", "Bcell", "CODEX", 102, "fov_2", "ds_2", (10, 10)),
            CellIndexRecord(3, "Bcell", "Bcell", "CODEX", 103, "fov_3", "ds_3", (10, 10)),
        ])
        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
        )
        # Get FOV names per side
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}
        # Each modality bucket must have ≥1 FOV in train and val
        assert any(f in train_fovs for f in {"fov_0", "fov_1"})
        assert any(f in val_fovs for f in {"fov_0", "fov_1"})
        assert any(f in train_fovs for f in {"fov_2", "fov_3"})
        assert any(f in val_fovs for f in {"fov_2", "fov_3"})

    def test_single_fov_stratum_forced_to_train(self):
        """A modality/tissue combo with only 1 FOV must go to train (cannot eval)."""
        ds = _build_dataset([
            (0, "MIBI", "lung", "CD4T"),
            (1, "MIBI", "lung", "CD4T"),  # MIBI/lung has 2
            (2, "CODEX", "tonsil", "CD4T"),  # CODEX/tonsil has only 1 — forced train
        ])
        # Add other classes to break sole-source forcing
        ds.indices.append(
            CellIndexRecord(0, "Bcell", "Bcell", "MIBI", 100, "fov_0", "ds_0", (10, 10))
        )
        ds.indices.append(
            CellIndexRecord(1, "Bcell", "Bcell", "MIBI", 101, "fov_1", "ds_1", (10, 10))
        )
        ds.indices.append(
            CellIndexRecord(2, "Bcell", "Bcell", "CODEX", 102, "fov_2", "ds_2", (10, 10))
        )
        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.5, seed=0, stratify_by=("modality", "tissue"),
        )
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}
        assert "fov_2" in train_fovs, "single-FOV stratum should be forced to train"
        assert "fov_2" not in val_fovs

    def test_legacy_unstratified_path_still_works(self):
        """Empty stratify_by tuple uses the legacy global-shuffle path."""
        ds = _build_dataset([
            (0, "MIBI", "lung", "CD4T"),
            (1, "MIBI", "lung", "CD4T"),
            (2, "MIBI", "lung", "CD4T"),
            (3, "MIBI", "lung", "CD4T"),
        ])
        ds.indices.extend([
            CellIndexRecord(0, "Bcell", "Bcell", "MIBI", 100, "fov_0", "ds_0", (10, 10)),
            CellIndexRecord(1, "Bcell", "Bcell", "MIBI", 101, "fov_1", "ds_1", (10, 10)),
            CellIndexRecord(2, "Bcell", "Bcell", "MIBI", 102, "fov_2", "ds_2", (10, 10)),
            CellIndexRecord(3, "Bcell", "Bcell", "MIBI", 103, "fov_3", "ds_3", (10, 10)),
        ])
        train_idx, val_idx = create_fov_splits(
            ds, train_ratio=0.75, seed=0, stratify_by=(),
        )
        # 4 FOVs, 0.75 ratio → 3 train, 1 val
        train_fovs = {ds.indices[i][5] for i in train_idx}
        val_fovs = {ds.indices[i][5] for i in val_idx}
        assert len(train_fovs) == 3
        assert len(val_fovs) == 1
