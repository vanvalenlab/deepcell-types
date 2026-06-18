# Error Handling & Failure Modes Audit — deepcell-types v0.1.0 (PR #41)

(2 blockers, 3 highs, 3 mediums, 0 lows)

## BLOCKER 1: Resume config check omits `n_heads` and `n_celltypes` — incompatible architecture loaded silently
**Location:** `scripts/train.py:688-698`
Loop only validates `("resnet_channels","d_model")`. `n_heads` is not recoverable from tensor shapes (the code says so), so `--n_heads 4` on resume loads silently with wrong attention config → scientifically invalid run, no error.
**Recommendation:** Add `n_heads` (and `n_celltypes`) to the validation loop.

## BLOCKER 2: Nimbus `main()` returns exit code 0 on fatal setup failures
**Location:** `baselines/nimbus/run.py:323-326, 365-368, 596-599`
Bare `return` in a Click command exits 0. "Nimbus not installed" / "no datasets" / "no predictions" all report success to CI; metrics JSON never written.
**Recommendation:** `raise click.UsageError(...)` / `sys.exit(1)` like xgb/run.py.

## HIGH 1: Resume legacy path loads checkpoint with no vocab/architecture validation
**Location:** `scripts/train.py:668-685` — run the same shape checks before `load_matching_state_dict`.

## HIGH 2: Nimbus FOV-level channel inference skipped silently when `norm_dict` is missing
**Location:** `baselines/nimbus/run.py:440-446, 505-516` — `print` not `logger.warning`; possible silent channel-name mismatch.

## HIGH 3: Nimbus bare `except Exception` swallows per-FOV errors with no count/rate guard
**Location:** `baselines/nimbus/run.py:479-486` — could drop 30% of FOVs and report a metric on the rest. Add a 1% rate guard like `dataset.py:263-272`.

## MEDIUM 1: `_infer_n_domains` KeyErrors on pre-DANN checkpoints with a misleading "not a deepcell-types checkpoint" message (`predict.py:163-164`).
## MEDIUM 2: `_infer_spatial_pool_size` divides by a hardcoded 64; future `d_model` change → cryptic shape mismatch (`predict.py:153-160`). Save `spatial_pool_size` in CKPT_CONFIG.
## MEDIUM 3: Resume `n_celltypes` check is skipped when the key is absent (`train.py:691-692`).

## Strengths
`weights_only=True` with correctly-scoped fallback. ct2idx/canonical_channels ordering validation. Atomic checkpoint/cache writes (tempfile+fsync+rename). Cache ownership / world-writable rejection. 1% failure-rate guard in archive load. Empty-FOV path returns typed empties. `freeze_backbone` zero-trainable guard.
