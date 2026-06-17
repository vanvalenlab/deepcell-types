# Security Audit — deepcell-types v0.1.0 (PR #41)

(1 blocker, 1 high, 2 mediums, 0 lows)

## BLOCKER: `weights_only=False` fallback in `_torch_load_weights` enables RCE on old torch
**Location:** `deepcell_types/predict.py:107-122`
On torch <1.13 the fallback executes the checkpoint as arbitrary pickle. Combined with caller-controlled `model_name` paths and the MD5-only `download_model`, a tampered `.pt` = OS-level code execution. Warning is silenceable; no hard error.
**Recommendation:** Replace the fallback with a hard `RuntimeError`; add `torch>=1.13` (effectively already needed) to pyproject.

## HIGH: Baseline training scripts reload their checkpoints with `weights_only=False`
**Location:** `baselines/cellsighter/run.py:441`, `baselines/maps/run.py:468`
`weights_only=False` used only to also read optimizer state / epoch (used for logging). Path is derived from `--model_name` (traversal) and a TOCTOU window on shared nodes → pickle RCE.
**Recommendation:** Save weights separately from optimizer state; load weights with `weights_only=True`.

## MEDIUM: `download_model()` integrity verified with MD5 only
**Location:** `deepcell_types/utils/_auth.py:163-172`, `utils/__init__.py:14-19`
MD5 is collision-broken; server-chosen presigned URL is not host-pinned. Defense-in-depth gap; load-bearing if the BLOCKER isn't fixed.
**Recommendation:** Switch registry to SHA-256.

## MEDIUM: `_model_path` accepts arbitrary filesystem paths with no sandboxing
**Location:** `deepcell_types/predict.py:125-133`
Documented feature, but composes with the BLOCKER (arbitrary path + unsafe deserializer = caller RCE). Low severity once weights_only is enforced.
**Recommendation:** Document trusted-source expectation; optionally restrict bare-string names to `~/.deepcell/models/`.

## Strengths
All YAML uses `safe_load`. The preprocessing op library is genuinely bounded (if/elif dispatch, no getattr/eval; unknown op → ValueError). No subprocess/os.system/eval/exec/pickle.load in library code. Archive extraction has path-traversal + symlink protection. MD5 check deletes-on-failure. The `weights_only` fallback precisely discriminates the TypeError cause. Checkpoint vocab cross-validation at load time.
