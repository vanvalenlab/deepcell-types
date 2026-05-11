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

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))
DEFAULT_ARCHIVE = DATA_DIR / "tissuenet-caitlin-labels.zarr"

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


def _is_control_channel(channel: str) -> bool:
    upper = channel.strip().upper()
    return upper in CONTROL_CHANNELS or upper.startswith(CONTROL_CHANNEL_PREFIXES)


def _resolve_channel(channel: str, root_channels: set[str]) -> str | None:
    # Strict canonical contract: only exact matches against the registered
    # channels resolve. Source-data variants must be canonicalized at
    # ingestion (hubmap-to-zarr/apply_canonicalization.py).
    return channel if channel in root_channels else None


def _discover_preprocessed(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("**/preprocessed/zarr.json"))


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


def validate_archive(
    root: Path,
    *,
    split_paths: list[Path] | None = None,
    allow_two_channel: bool = False,
    fail_on_unknown_channel: bool = False,
    required_markers: dict[str, set[str]] | None = None,
) -> ArchiveReport:
    report = ArchiveReport()
    root = root.expanduser()
    if not (root / "zarr.json").exists():
        report.errors.append(f"archive root is missing zarr.json: {root}")
        return report

    root_channels = set(_attrs(root / "zarr.json").get("all_standardized_channels", []))
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
    args = parser.parse_args(argv)

    required_markers = {key: set(value) for key, value in KNOWN_REPAIR_MARKERS.items()}
    for fov_key, marker in args.require_marker:
        required_markers.setdefault(fov_key, set()).add(marker)

    report = validate_archive(
        args.zarr,
        split_paths=args.split,
        allow_two_channel=args.allow_two_channel,
        fail_on_unknown_channel=args.fail_on_unknown_channel,
        required_markers=required_markers,
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
