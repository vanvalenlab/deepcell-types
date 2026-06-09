"""Validate the TissueNet zarr archive contract without reading image chunks.

This is a metadata-only guard for repaired archives. It checks that
preprocessed raw/mask arrays are structurally aligned, no repaired FOV has
collapsed back to two channels, split files point at real FOVs, and known
repair sentinel markers are present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))
DEFAULT_ARCHIVE = DATA_DIR / "tissuenet.zarr" if DATA_DIR != Path("") else None

KNOWN_REPAIR_MARKERS = {
    "HBM222_WQKC_382": {"CD20", "HO1", "iNOS"},
    "dryadb005_intestine_codex_B005_SB_reg003": {
        "CD154",
        "CD161",
        "CD25",
        "CD33",
        "NKG2D",
    },
    "dryadb006_intestine_codex_B006_SB_reg002": {
        "CD21",
        "CD25",
        "CDX2",
        "FAP",
        "HLA-Class-2",
        "MUC1",
        "NKG2D",
    },
    "dryadb008_intestine_codex_B008_CL_reg001": {"CD49a", "CD49f", "CDX2"},
}

CONTROL_CHANNEL_PREFIXES = (
    "BLANK",
    "CH2",
    "CH3",
    "DAPI",
    "DNA",
    "DRAQ",
    "EMPTY",
    "EMPYT",
    "HOECHST",
    "IR",
)
CONTROL_CHANNELS = {
    "H3",
    "HISTONEH3",
    "HISTONE H3",
    "RABBIT IGG",
    "GOAT IGG",
    "MOUSE IGG",
}


@dataclass
class ArchiveReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    num_fovs: int = 0
    num_two_channel: int = 0
    num_unknown_channels: int = 0


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _attrs(zarr_json: Path) -> dict:
    return dict(_read_json(zarr_json).get("attributes", {}))


def _shape(zarr_json: Path) -> list[int] | None:
    if not zarr_json.exists():
        return None
    shape = _read_json(zarr_json).get("shape")
    return list(shape) if shape is not None else None


_INTEGER_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}


def _dtype(zarr_json: Path):
    """The zarr v3 ``data_type`` (a str like 'uint32', or a dict for strings)."""
    if not zarr_json.exists():
        return None
    return _read_json(zarr_json).get("data_type")


def _is_control_channel(channel: str) -> bool:
    upper = channel.strip().upper()
    return upper in CONTROL_CHANNELS or upper.startswith(CONTROL_CHANNEL_PREFIXES)


def _resolve_channel(channel: str, root_channels: set[str]) -> str | None:
    # Strict canonical contract: only exact matches against the registered
    # channels resolve. Source-data variants must be canonicalized at
    # ingestion (the archive ingestion pipeline).
    return channel if channel in root_channels else None


def _discover_preprocessed(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("**/preprocessed/zarr.json"))


def _matrix_shape(matrix) -> tuple[int, ...] | None:
    """Best-effort 2D shape of a JSON-stored positivity matrix (list-of-lists).

    Returns ``None`` when the value is missing or not a non-empty 2D list.
    """
    if not isinstance(matrix, list) or not matrix:
        return None
    if not isinstance(matrix[0], list):
        return None
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    return (n_rows, n_cols)


def _fov_key(root: Path, preprocessed: Path) -> str:
    return preprocessed.parent.relative_to(root).as_posix()


def _validate_splits(
    root: Path, split_paths: list[Path], fov_keys: set[str], report: ArchiveReport
) -> None:
    for split_path in split_paths:
        data = _read_json(split_path)
        train = {
            (ds_name, fov_name)
            for ds_name, fov_names in data.get("train", {}).items()
            for fov_name in fov_names
        }
        val = {
            (ds_name, fov_name)
            for ds_name, fov_names in data.get("val", {}).items()
            for fov_name in fov_names
        }
        heldout = {
            (ds_name, fov_name)
            for ds_name, fov_names in data.get("heldout", {}).items()
            for fov_name in fov_names
        }
        overlap = train & val
        if overlap:
            report.errors.append(
                f"{split_path}: {len(overlap)} FOVs appear in train and val"
            )
        heldout_overlap = heldout & (train | val)
        if heldout_overlap:
            report.errors.append(
                f"{split_path}: {len(heldout_overlap)} heldout FOVs also appear in train/val"
            )
        split_keys = {ds for ds, fov in train | val | heldout if ds == fov}
        unsupported = {(ds, fov) for ds, fov in train | val | heldout if ds != fov}
        if unsupported:
            report.errors.append(
                f"{split_path}: {len(unsupported)} entries use unsupported dataset/FOV pairs"
            )
        missing = split_keys - fov_keys
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            report.errors.append(
                f"{split_path}: {len(missing)} split FOVs are absent from {root}: {examples}"
            )


def check_marker_index_order(
    archive_channels: list[str], expected_order: list[str]
) -> list[str]:
    """Verify the archive's marker→index map matches the released model's frozen order.

    ``all_standardized_channels`` IS the released model's marker index map:
    ``config.DCTConfig`` builds ``marker2idx = {ch: i for i, ch in
    enumerate(all_standardized_channels)}``, and the released checkpoint +
    ``svd_512.npz`` marker embeddings are built for that exact order. Reordering
    or resizing it silently misaligns the checkpoint's per-marker weights, or
    trips the ``n_markers`` guard in ``predict._build_model`` so the released
    checkpoint fails to load. This caught a real incident (2026-06-01) where the
    registry was re-accumulated to a 327-channel union and re-sorted.

    Returns a list of human-readable error strings (empty == contract holds).
    """
    archive_channels = list(archive_channels)
    expected_order = list(expected_order)
    if archive_channels == expected_order:
        return []

    errors: list[str] = []
    if len(archive_channels) != len(expected_order):
        errors.append(
            f"all_standardized_channels has {len(archive_channels)} markers but the "
            f"released model expects {len(expected_order)} (marker index-map size "
            f"mismatch — the released checkpoint will fail to load)"
        )
    archive_set, expected_set = set(archive_channels), set(expected_order)
    missing = sorted(expected_set - archive_set)
    extra = sorted(archive_set - expected_set)
    if missing:
        shown = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
        errors.append(
            f"{len(missing)} markers expected by the released model are absent from "
            f"the archive: {shown}"
        )
    if extra:
        shown = ", ".join(extra[:10]) + (" …" if len(extra) > 10 else "")
        errors.append(
            f"{len(extra)} markers in the archive are unknown to the released model: "
            f"{shown}"
        )
    if not missing and not extra:
        # Same set, different order — the dangerous silent case.
        for idx, (got, want) in enumerate(zip(archive_channels, expected_order)):
            if got != want:
                errors.append(
                    f"marker order diverges from the released model at index {idx}: "
                    f"archive has {got!r}, expected {want!r} (same markers, reordered "
                    f"— silently misaligns the checkpoint's per-marker weights)"
                )
                break
    return errors


def _load_expected_marker_order(path: Path) -> list[str]:
    """Load the released model's frozen marker order (source of truth).

    Accepts the released embeddings ``.npz`` (whose ``marker2idx`` object holds
    the ``{marker: index}`` map) or a JSON file containing either a
    ``{marker: index}`` dict or an ordered list of marker names.
    """
    path = Path(path)
    if path.suffix == ".npz":
        import numpy as np

        data = np.load(path, allow_pickle=True)
        marker2idx = data["marker2idx"].item()
        return [m for m, _ in sorted(marker2idx.items(), key=lambda kv: kv[1])]
    obj = _read_json(path)
    if isinstance(obj, dict):
        return [m for m, _ in sorted(obj.items(), key=lambda kv: kv[1])]
    return list(obj)


def validate_archive(
    root: Path,
    *,
    split_paths: list[Path] | None = None,
    allow_two_channel: bool = False,
    fail_on_unknown_channel: bool = False,
    required_markers: dict[str, set[str]] | None = None,
    expected_marker_order: list[str] | None = None,
) -> ArchiveReport:
    report = ArchiveReport()
    root = root.expanduser()
    if not (root / "zarr.json").exists():
        report.errors.append(f"archive root is missing zarr.json: {root}")
        return report

    root_channels_ordered = list(
        _attrs(root / "zarr.json").get("all_standardized_channels", [])
    )
    root_channels = set(root_channels_ordered)
    if expected_marker_order is not None:
        report.errors.extend(
            check_marker_index_order(root_channels_ordered, expected_marker_order)
        )

    # cell_type_mapping is {cell_type_name: index} and IS the model's class
    # index map; its keys must be exactly all_standardized_cell_types and its
    # values a contiguous 0..N-1 range. A drifted mapping silently mislabels
    # every prediction (analogous to the marker index-map guard above).
    root_attrs = _attrs(root / "zarr.json")
    cell_type_mapping = root_attrs.get("cell_type_mapping")
    all_cell_types = root_attrs.get("all_standardized_cell_types")
    if cell_type_mapping is None:
        report.errors.append("root attrs missing 'cell_type_mapping'")
    if all_cell_types is None:
        report.errors.append("root attrs missing 'all_standardized_cell_types'")
    if isinstance(cell_type_mapping, dict) and all_cell_types is not None:
        mapping_keys = set(cell_type_mapping)
        declared = set(all_cell_types)
        if mapping_keys != declared:
            missing = sorted(declared - mapping_keys)
            extra = sorted(mapping_keys - declared)
            report.errors.append(
                "cell_type_mapping keys disagree with all_standardized_cell_types "
                f"(missing from mapping: {missing[:10]}; extra in mapping: {extra[:10]})"
            )
        values = sorted(cell_type_mapping.values())
        if values != list(range(len(cell_type_mapping))):
            report.errors.append(
                "cell_type_mapping values are not a contiguous 0..N-1 index range"
            )

    preprocessed_paths = _discover_preprocessed(root)
    fov_keys = {_fov_key(root, path) for path in preprocessed_paths}
    unknown_channels: dict[str, int] = {}

    for preprocessed in preprocessed_paths:
        report.num_fovs += 1
        fov_key = _fov_key(root, preprocessed)
        raw_shape = _shape(preprocessed / "raw" / "zarr.json")
        mask_shape = _shape(preprocessed / "mask" / "zarr.json")
        if raw_shape is None:
            report.errors.append(f"{fov_key}: missing preprocessed/raw")
            continue
        if mask_shape is None:
            report.errors.append(f"{fov_key}: missing preprocessed/mask")
            continue
        if len(raw_shape) != 3:
            report.errors.append(f"{fov_key}: raw shape is not CYX: {raw_shape}")
        if len(mask_shape) != 2:
            report.errors.append(f"{fov_key}: mask shape is not YX: {mask_shape}")
        mask_dtype = _dtype(preprocessed / "mask" / "zarr.json")
        if mask_dtype is not None and mask_dtype not in _INTEGER_DTYPES:
            report.errors.append(
                f"{fov_key}: preprocessed/mask dtype {mask_dtype!r} is not an "
                "integer type; cell-label matching (`mask == cell_idx`) requires "
                "an integer mask."
            )
        if len(raw_shape) == 3 and len(mask_shape) == 2 and raw_shape[1:] != mask_shape:
            report.errors.append(
                f"{fov_key}: raw spatial shape {raw_shape[1:]} != mask shape {mask_shape}"
            )
        if raw_shape and raw_shape[0] <= 2:
            report.num_two_channel += 1
            if not allow_two_channel:
                report.errors.append(
                    f"{fov_key}: preprocessed/raw has only {raw_shape[0]} channels"
                )

        channel_names = list(
            _attrs(preprocessed / "zarr.json").get("channel_names", [])
        )
        if raw_shape and len(channel_names) != raw_shape[0]:
            report.errors.append(
                f"{fov_key}: {len(channel_names)} channel names for raw shape {raw_shape}"
            )
        for channel in channel_names:
            if channel is None:
                continue
            channel = str(channel)
            if _resolve_channel(
                channel, root_channels
            ) is None and not _is_control_channel(channel):
                unknown_channels[channel] = unknown_channels.get(channel, 0) + 1

        # preprocessed metadata the inference patch generator and the
        # gold-standard pipeline depend on. centroids/scale_factor live on the
        # preprocessed group's attrs (PatchDataset reads scale_factor for MPP
        # resampling; centroids anchor each cell's patch).
        preprocessed_attrs = _attrs(preprocessed / "zarr.json")
        if "centroids" not in preprocessed_attrs:
            report.errors.append(f"{fov_key}: preprocessed attrs missing 'centroids'")
        if "scale_factor" not in preprocessed_attrs:
            report.errors.append(
                f"{fov_key}: preprocessed attrs missing 'scale_factor'"
            )

        # Cell-type ground truth: standardized_source is the SOLE GT source for
        # all code. A FOV that ships a cell_types/annotations group must carry
        # the standardized_source attr, or every consumer silently sees no
        # labels for it. (Not every FOV is annotated, so only check when the
        # annotations group exists.)
        annotations_json = (
            preprocessed.parent / "cell_types" / "annotations" / "zarr.json"
        )
        if annotations_json.exists():
            annotation_attrs = _attrs(annotations_json)
            if "standardized_source" not in annotation_attrs:
                report.errors.append(
                    f"{fov_key}: cell_types/annotations missing "
                    "'standardized_source' (the sole GT source)"
                )

        # Marker positivity: when a FOV provides a marker_positivity group, its
        # positivity_matrix must be shaped (n_cell_types, n_markers) and the
        # 'markers'/'cell_types' label attrs must be present and agree with that
        # shape. A degraded matrix (wrong axis order, stale markers list) would
        # otherwise be read into the MP metrics with silently wrong alignment.
        mp_json = preprocessed.parent / "marker_positivity" / "zarr.json"
        if mp_json.exists():
            mp_attrs = _attrs(mp_json)
            markers = mp_attrs.get("markers")
            mp_cell_types = mp_attrs.get("cell_types")
            matrix_shape = _matrix_shape(mp_attrs.get("positivity_matrix"))
            if markers is None:
                report.errors.append(
                    f"{fov_key}: marker_positivity missing 'markers' attr"
                )
            if "positivity_matrix" not in mp_attrs:
                report.errors.append(
                    f"{fov_key}: marker_positivity missing 'positivity_matrix' attr"
                )
            elif matrix_shape is None:
                report.errors.append(
                    f"{fov_key}: marker_positivity positivity_matrix is not a "
                    "non-empty 2D matrix"
                )
            elif markers is not None and mp_cell_types is not None:
                expected = (len(mp_cell_types), len(markers))
                if matrix_shape != expected:
                    report.errors.append(
                        f"{fov_key}: marker_positivity positivity_matrix shape "
                        f"{matrix_shape} != (n_cell_types, n_markers) {expected}"
                    )

    report.num_unknown_channels = sum(unknown_channels.values())
    if unknown_channels:
        examples = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(
                unknown_channels.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        )
        msg = f"{len(unknown_channels)} channel names are not model-visible: {examples}"
        if fail_on_unknown_channel:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)

    for fov_key, markers in (required_markers or {}).items():
        if fov_key not in fov_keys:
            report.errors.append(f"{fov_key}: required-marker sentinel FOV is absent")
            continue
        channels = {
            _resolve_channel(str(channel), root_channels) or str(channel)
            for channel in _attrs(root / fov_key / "preprocessed" / "zarr.json").get(
                "channel_names", []
            )
        }
        missing = markers - channels
        if missing:
            report.errors.append(
                f"{fov_key}: missing required repaired markers {sorted(missing)}"
            )

    _validate_splits(root, split_paths or [], fov_keys, report)
    return report


def _parse_required_marker(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected FOV_KEY:MARKER")
    fov_key, marker = value.split(":", 1)
    return fov_key, marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, action="append", default=[])
    parser.add_argument("--allow-two-channel", action="store_true")
    parser.add_argument("--fail-on-unknown-channel", action="store_true")
    parser.add_argument(
        "--require-marker",
        type=_parse_required_marker,
        action="append",
        default=[],
        help="Additional required marker sentinel as FOV_KEY:MARKER",
    )
    parser.add_argument(
        "--marker-order-from",
        type=Path,
        default=None,
        help=(
            "Path to the released model's embeddings .npz (with a marker2idx "
            "object) or a JSON marker2idx/ordered-list. When set, assert that the "
            "archive's all_standardized_channels matches that frozen marker→index "
            "order exactly. Guards against the checkpoint-breaking reorder/resize."
        ),
    )
    args = parser.parse_args(argv)

    if args.zarr is None:
        parser.error(
            "--zarr is required when the DATA_DIR environment variable is not set"
        )

    required_markers = {key: set(value) for key, value in KNOWN_REPAIR_MARKERS.items()}
    for fov_key, marker in args.require_marker:
        required_markers.setdefault(fov_key, set()).add(marker)

    expected_marker_order = (
        _load_expected_marker_order(args.marker_order_from)
        if args.marker_order_from is not None
        else None
    )

    report = validate_archive(
        args.zarr,
        split_paths=args.split,
        allow_two_channel=args.allow_two_channel,
        fail_on_unknown_channel=args.fail_on_unknown_channel,
        required_markers=required_markers,
        expected_marker_order=expected_marker_order,
    )

    print(f"Archive: {args.zarr}")
    print(f"FOVs with preprocessed arrays: {report.num_fovs}")
    print(f"Two-channel preprocessed FOVs: {report.num_two_channel}")
    print(f"Non-model-visible channel-name occurrences: {report.num_unknown_channels}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.errors:
        print(f"FAILED: {len(report.errors)} contract errors")
        return 1
    print("PASSED: archive contract checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
