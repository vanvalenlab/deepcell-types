"""Run the tutorial's cell-type inference against a *local* zarr + checkpoint.

This mirrors the inference section of ``docs/site/tutorial.md`` but skips the
remote HuBMAP S3 archive and the ``download_model`` step: it pulls one FOV's
image / segmentation mask / channel names / mpp straight out of a local zarr
and runs ``deepcell_types.predict`` with a checkpoint already on disk. Use it
to confirm the pipeline runs end-to-end in your environment.

Two on-disk FOV layouts are supported:

* **native** (e.g. ``celltype-data-public/*.zarr``): ``<fov>/raw`` holds the
  raw, un-normalized image, channel names live in the *root* ``channel_names``
  attr, and the native resolution is ``<fov>.attrs["mpp"]``. This is the
  faithful path — ``predict`` does its own preprocessing.
* **preprocessed** (e.g. ``gold_standard.zarr``): ``<fov>/preprocessed/raw`` is
  already resampled to 0.5 µm/px and min-max normalized, with channel names in
  ``preprocessed.attrs["channel_names"]``. We feed it with ``mpp=TARGET_MPP`` so
  ``predict``'s resample is a no-op and its min-max step is ~idempotent.

Examples
--------
    python scripts/tutorial_local_demo.py \
        --zarr /data/xwang3/celltype-data-public/Tissue-Breast-Keren_TNBC_MIBI.zarr \
        --fov Point41 \
        --model ~/dct-final-ckpt/deepcell-types_2026-05-17.pt \
        --device cpu
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import zarr

import deepcell_types
from deepcell_types.preprocessing import TARGET_MPP


def load_fov(zarr_path, fov_name):
    """Return ``(img, mask, channel_names, mpp, ground_truth)`` for one FOV.

    ``ground_truth`` is a ``{cell_index: label}`` dict when the archive carries
    a ``cell_type_info`` table, else ``None``.
    """
    z = zarr.open_group(str(zarr_path), mode="r")
    fov = z[fov_name]
    fov_keys = set(fov.group_keys())

    if "preprocessed" in fov_keys:
        # preprocessed layout: already at TARGET_MPP, channel names on the group
        pp = fov["preprocessed"]
        img = pp["raw"][:]
        mask = pp["mask"][:]
        channel_names = list(pp.attrs["channel_names"])
        mpp = TARGET_MPP
        cti_parent = pp
    else:
        # native layout: raw intensities, channel names in root attrs
        img = fov["raw"][:]
        mask = fov["mask"][:]
        channel_names = list(z.attrs["channel_names"])
        mpp = float(fov.attrs["mpp"])
        cti_parent = fov

    mask = np.asarray(mask).squeeze().astype(np.uint32)

    ground_truth = None
    if "cell_type_info" in set(cti_parent.array_keys()) | set(cti_parent.group_keys()):
        cti = cti_parent["cell_type_info"]
        if hasattr(cti, "dtype") and cti.dtype.names:  # structured array
            rows = cti[:]
            ground_truth = {int(r["cell_index"]): str(r["cell_type"]) for r in rows}
        else:  # group with parallel cell_index / cell_type arrays
            idx = cti["cell_index"][:]
            lab = cti["cell_type"][:]
            ground_truth = {int(i): str(name) for i, name in zip(idx, lab)}

    return img, mask, channel_names, mpp, ground_truth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zarr",
        default="/data/xwang3/celltype-data-public/Tissue-Breast-Keren_TNBC_MIBI.zarr",
        help="Path to a local zarr archive.",
    )
    p.add_argument("--fov", default="Point41", help="FOV (group) name within the zarr.")
    p.add_argument(
        "--model",
        default=str(Path.home() / "dct-final-ckpt" / "deepcell-types_2026-05-17.pt"),
        help="Path to the .pt checkpoint.",
    )
    p.add_argument("--device", default="cpu", help='e.g. "cpu", "cuda:0".')
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    img, mask, channel_names, mpp, ground_truth = load_fov(args.zarr, args.fov)
    print(
        f"FOV {args.fov!r}: img={img.shape} mask={mask.shape} "
        f"cells={int(mask.max())} channels={len(channel_names)} mpp={mpp}"
    )

    cell_types = deepcell_types.predict(
        img,
        mask,
        channel_names,
        mpp,
        model_name=args.model,
        device=args.device,
        num_workers=args.num_workers,
    )

    print(f"\nPredicted {len(cell_types)} cells.")
    print("Prediction distribution:")
    for label, n in collections.Counter(cell_types).most_common():
        print(f"  {n:6d}  {label}")

    if ground_truth is not None:
        cell_indices = sorted(int(i) for i in np.unique(mask) if i > 0)
        match = total = 0
        for idx, pred in zip(cell_indices, cell_types):
            gt = ground_truth.get(idx)
            # Skip unharmonized GT and abstained cells for a clean comparison.
            if gt is None or gt == "Failed_Harmonization" or pred == "Unknown":
                continue
            total += 1
            match += pred == gt
        if total:
            print(
                f"\nExact-match vs ground truth "
                f"(excl. Failed_Harmonization & abstained): "
                f"{match}/{total} = {100 * match / total:.1f}%"
            )


if __name__ == "__main__":
    main()
