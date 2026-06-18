# Documentation Audit — deepcell-types v0.1.0 (PR #41)

(2 blockers, 1 high, 2 mediums, 0 lows)

## BLOCKER: README contradicts the v0.1.0 archive-free inference feature
**Location:** `README.md:5-6` and `:42-46` vs `config.py:236-254`, `CHANGELOG.md:22-26`
README still says you *must* provide a multi-GB TissueNet zarr archive before `predict()`. The code falls back to packaged `vocab.json` (archive-free), which is the headline v0.1.0 feature. A new user reads the README, concludes they can't run inference, and abandons.
**Recommendation:** Rewrite intro + "TissueNet zarr archive" section to state the archive is optional.

## BLOCKER: `scripts/train.py` docstring says DANN disabled by default; CLI default enables it
**Location:** `scripts/train.py:11-12` vs `:278-281`
Docstring: `domain:0 (DANN disabled by default)`. CLI: `--domain_weight default=0.1`. Effective default trains with DANN on.
**Recommendation:** Fix docstring to `domain:0.1 (DANN enabled; pass --domain_weight 0 to disable)`.

## HIGH: `preprocessing_ops.py` docstring says "p99 clip"; `DEFAULT_CONFIG` uses p99.9
**Location:** `preprocessing_ops.py:8-9` vs `:21-24` (`_DEFAULT_PERCENTILE = 99.9`)
**Recommendation:** Fix docstring to p99.9.

## MEDIUM: CHANGELOG omits `preprocess_fov` and `PreprocessedFov` from listed top-level exports
**Location:** `CHANGELOG.md:16-17` vs `__init__.py:23-24`.

## MEDIUM: README quickstart imports `download_model` from `deepcell_types.utils` instead of the top-level re-export
**Location:** `README.md:36-37` vs `__init__.py:28`.

## Strengths
`predict()` defaults (num_workers=0, ct_abstention_k=0.2, return_probabilities=False, zarr_path=None) match docs exactly. `PERCENTILE_THRESHOLD=99.9` correct. All 7 CHANGELOG-listed exports present. `vocab.json` correctly in package-data. Baseline READMEs accurate (deviations cited with line numbers). Keyword-only args and `device_num` alias match docs.
