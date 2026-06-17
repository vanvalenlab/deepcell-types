# Dependencies & Packaging Audit — deepcell-types v0.1.0 (PR #41)

(1 blocker, 2 highs, 3 mediums, 0 lows)

## BLOCKER: `tifffile` declared in `[train]` but zero imports exist anywhere
**Location:** `pyproject.toml:68`
No `import tifffile` in `deepcell_types/` or `scripts/`. Comment claims "TIFF-based gold-standard ingest tooling" but nothing imports it.
**Recommendation:** Remove it, or add the actual import to the module that needs it.

## HIGH: `np.ptp` in the inference path breaks on NumPy ≥ 2.0
**Location:** `deepcell_types/preprocessing.py:246`
`_normalize_per_channel` (live inference path via `PatchDataset.__iter__`) uses `np.ptp`, removed as a free function in NumPy 2.0. `numpy>=1.24` base pin permits 2.x. The identical call was already fixed at line 140 but this one was missed → `pip install deepcell-types` + `predict()` raises `AttributeError` on a fresh env today.
**Recommendation:** Replace with `np.max(...) - np.min(...)` (the existing fix); consider `numpy>=1.24,<3` or validate 2.x.

## HIGH: `scikit-image>=0.20` loose lower bound interacts with the np.ptp/NumPy-2 issue
**Location:** `pyproject.toml:27` — correct to be in base (inference uses `rescale`), but pin hygiene weak.

## MEDIUM: `kaleido` pin `<1.0` excludes current stable; `fig.write_image()` ValueError-on-missing-kaleido not caught
**Location:** `pyproject.toml:66`, `training/utils.py:281-330`.
## MEDIUM: `torchinfo`/`torchmetrics` in `[train]` but only used by `scripts/` (not the installed package); `torchmetrics` may pull torchvision (`pyproject.toml:60-61`).
## MEDIUM: package-data globs `training/config/*.yaml` only — embedding `*.json` fallbacks would silently not ship (`pyproject.toml:125-126`, `config.py:509-521`).

## Strengths
Inference/train boundary enforced by a subprocess CI test (`test_inference_deps.py`). Lazy imports for pandas/zarr keep the inference graph numpy-only. `vocab.json`/`channel_mapping.yaml` in package-data; wheel CI installs a real wheel and pokes `DCTConfig`. nimbus py<3.12 marker documented + CI-exercised. MAPS license (Apache + Commons Clause) disclosed in NOTICE; CellSighter is an independent reimplementation. `weights_only=True` with torch>=2.0 floor.
