"""Split an existing val set into a model-selection val_subset and frozen test set.

This is **stage 2** of the two-stage canonical split (stage 1 is
``scripts/generate_splits.py``). A held-out test set that is never used for
model selection is required for an unbiased final number. Rather than
re-splitting the whole archive, this script takes an existing val set and
carves it into val_subset (70%) and test_subset (30%), stratified per dataset
so every dataset still contributes to both subsets.

Output:
    Two new split files that share the same `train` dict as the input (so existing
    models remain comparable for training), but have `val` containing only the
    subset relevant to each use:

      splits/fov_split_v10_valsubset.json  -- train unchanged, val = val_subset (70%)
      splits/fov_split_v10_test.json        -- train unchanged, val = test_subset (30%)

    Each output declares the omitted validation FOVs in `heldout` so strict
    split loading can distinguish intentional exclusions from stale split files.

    Use valsubset during training (`--split_file splits/fov_split_v10_valsubset.json`).
    Use test for the final released number (`--split_file splits/fov_split_v10_test.json`).

Usage:
    python -m scripts.split_val_for_test \\
        --input splits/fov_split_v10.json \\
        --output_prefix splits/fov_split_v10 \\
        --test_ratio 0.3 --seed 42
"""

import click
import json
import random
from pathlib import Path


@click.command()
@click.option(
    "--input",
    "input_path",
    type=str,
    required=True,
    help="Existing 2-way split JSON (e.g. splits/fov_split_v9.json)",
)
@click.option(
    "--output_prefix",
    type=str,
    required=True,
    help="Prefix for output files; writes <prefix>_valsubset.json and <prefix>_test.json",
)
@click.option(
    "--test_ratio",
    type=float,
    default=0.3,
    help="Fraction of val FOVs per dataset that move to the held-out test subset",
)
@click.option("--seed", type=int, default=42)
def main(input_path: str, output_prefix: str, test_ratio: float, seed: int) -> None:
    with open(input_path) as f:
        split_data = json.load(f)

    train_by_ds = split_data["train"]
    val_by_ds = split_data["val"]

    rng = random.Random(seed)

    val_subset_by_ds: dict[str, list[str]] = {}
    test_subset_by_ds: dict[str, list[str]] = {}

    # Detect per-dataset FOV count: canonical splits store each FOV as its own
    # "dataset" key (dataset_name == fov_name), so every dataset has exactly 1 FOV and
    # per-dataset stratification yields an empty test set. Fall back to a global FOV
    # shuffle when >= 90% of datasets have a single FOV in val.
    single_fov_ratio = sum(1 for v in val_by_ds.values() if len(v) < 2) / max(
        len(val_by_ds), 1
    )
    stratify_per_dataset = single_fov_ratio < 0.9

    if stratify_per_dataset:
        for ds_name, fov_names in val_by_ds.items():
            fovs = list(fov_names)
            rng.shuffle(fovs)
            n_test = max(1, round(len(fovs) * test_ratio)) if len(fovs) >= 2 else 0
            test_fovs = fovs[:n_test]
            val_fovs = fovs[n_test:]
            if val_fovs:
                val_subset_by_ds[ds_name] = val_fovs
            if test_fovs:
                test_subset_by_ds[ds_name] = test_fovs
    else:
        print(
            f"INFO: {single_fov_ratio:.0%} of val datasets have a single FOV — using global FOV shuffle instead of per-dataset stratification."
        )
        all_pairs = [(ds, fov) for ds, fovs in val_by_ds.items() for fov in fovs]
        rng.shuffle(all_pairs)
        n_test_total = round(len(all_pairs) * test_ratio)
        test_pairs = all_pairs[:n_test_total]
        val_pairs = all_pairs[n_test_total:]
        for ds, fov in val_pairs:
            val_subset_by_ds.setdefault(ds, []).append(fov)
        for ds, fov in test_pairs:
            test_subset_by_ds.setdefault(ds, []).append(fov)

    base_meta = dict(split_data.get("metadata", {}))

    strategy = "per_dataset" if stratify_per_dataset else "global_fov_shuffle"

    val_output = {
        "metadata": {
            **base_meta,
            "derived_from": str(input_path),
            "subset": "val_subset",
            "test_ratio_split_off": test_ratio,
            "subset_seed": seed,
            "subset_strategy": strategy,
            "num_val_fovs": sum(len(v) for v in val_subset_by_ds.values()),
            "num_heldout_fovs": sum(len(v) for v in test_subset_by_ds.values()),
        },
        "train": train_by_ds,
        "val": val_subset_by_ds,
        "heldout": test_subset_by_ds,
    }

    test_output = {
        "metadata": {
            **base_meta,
            "derived_from": str(input_path),
            "subset": "test_subset",
            "test_ratio_split_off": test_ratio,
            "subset_seed": seed,
            "subset_strategy": strategy,
            "num_val_fovs": sum(len(v) for v in test_subset_by_ds.values()),
            "num_heldout_fovs": sum(len(v) for v in val_subset_by_ds.values()),
        },
        "train": train_by_ds,
        "val": test_subset_by_ds,
        "heldout": val_subset_by_ds,
    }

    val_path = f"{output_prefix}_valsubset.json"
    test_path = f"{output_prefix}_test.json"

    Path(val_path).parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, "w") as f:
        json.dump(val_output, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_output, f, indent=2)

    total_val_before = sum(len(v) for v in val_by_ds.values())
    total_val_after = sum(len(v) for v in val_subset_by_ds.values())
    total_test = sum(len(v) for v in test_subset_by_ds.values())

    print(f"Input val FOVs:   {total_val_before} across {len(val_by_ds)} datasets")
    print(f"Val subset FOVs:  {total_val_after} -> {val_path}")
    print(f"Test subset FOVs: {total_test} -> {test_path}")
    if stratify_per_dataset:
        datasets_without_test = [
            ds for ds, fovs in val_by_ds.items() if ds not in test_subset_by_ds
        ]
        if datasets_without_test:
            print(
                f"WARNING: {len(datasets_without_test)} dataset(s) have only 1 val FOV and contributed no test FOV:"
            )
            for ds in datasets_without_test[:10]:
                print(f"  - {ds}")
            if len(datasets_without_test) > 10:
                print(f"  ... and {len(datasets_without_test) - 10} more")


if __name__ == "__main__":
    main()
