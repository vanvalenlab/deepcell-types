# Numerical Correctness & Stability Audit — deepcell-types v0.1.0 (PR #41)

(1 blocker, 2 highs, 2 mediums, 2 lows)

## BLOCKER: Default config double-weights rare classes (sampler + FocalLoss simultaneously)
**Location:** `scripts/train.py:614-617`, `deepcell_types/training/samplers.py:42-45`, `scripts/train.py:63-83`
Sampler floors effective count at 1000 (`max(count,1000)`); `compute_class_weights` uses raw `sqrt(total/count)` with NO floor. Both active by default. Net rare-class boost ≈ `total/316` vs intended `sqrt(total/N)`. The code comment says to avoid this, then the default does it. Corrupts class balance; inflates rare-class macro-F1; non-reproducible if a user follows the "use --no_class_weights" advice.
**Recommendation:** Default `no_class_weights=True`, or add the 1000 floor to `compute_class_weights`, or disable the sampler when class weights on. (Maintainer's sampler removal resolves this.)

## HIGH: CellSighter baseline `scatter_` on CUDA has undefined write order when marker index 0 is present
**Location:** `deepcell_types/baselines/cellsighter/model.py:107-111`
Padding slots (`ch_idx==-1`) clamped to 0; `scatter_` without `reduce=` has undefined behavior for duplicate indices on CUDA. When a real channel also maps to index 0, the padding write (0.0) can overwrite the real marker-0 data. TissueNet routinely has a channel at index 0 → silent zeroing of the first canonical marker on CUDA. Corrupts CellSighter training.
**Recommendation:** Use the sink-column pattern (allocate `num_markers+1`, redirect padding to the sink, slice off), as the main model does.

## HIGH: `compat_marker0_zero` silently defaults to True when checkpoint config is absent
**Location:** `deepcell_types/predict.py:225-226`
`config.get("compat_marker0_zero", True)`. A future `False`-trained checkpoint loaded via a path that drops the config dict silently reverts to True → zeros marker-0, degrading predictions. Validator checks values but doesn't assert presence for format_version ≥ 1.1.
**Recommendation:** Warn when the key is absent from a non-empty config; assert presence for v1.1 checkpoints.

## MEDIUM: `HierarchicalLoss` comment/code clamp mismatch (says 1e-8, uses 1e-7)
**Location:** `deepcell_types/training/losses.py:147-164`
Code is correct (fp32 via `autocast(enabled=False)` + `.float()`, clamp 1e-7 safe). Comment text still says 1e-8 → maintainer confusion. Dormant at `hierarchical_weight=0` default.
**Recommendation:** Fix the stale comment.

## MEDIUM: Pretraining `cell_area` from any-nonzero-channel heuristic biases `mean_expression`
**Location:** `deepcell_types/model.py:717-725`
`cell_pixels = (valid_sample != 0).any(dim=1)` underestimates cell area for sparse channels → inflated mean targets. `clamp(min=1.0)` prevents div0 but bias remains. Pretraining-only.
**Recommendation:** Use the self_mask (`spatial_context[:,0]`) as the true cell footprint.

## LOW: dead `if mean_intensity is not None` guard (always True)
**Location:** `deepcell_types/model.py:507` — remove or restructure to gate on the branch being enabled.

## LOW: `F.softmax` on fp16 logits outside autocast for metrics
**Location:** `scripts/train.py:166` — cast `.float()` before softmax (metrics-only; PyTorch's stable kernel likely avoids NaN in practice).

## Checklist resolution (confirmed correct)
- Main-model `scatter_` sink-column fix is correct, with `compat_marker0_zero` gating v0.1.0 bit-for-bit. CellSighter baseline still has the bug (HIGH above).
- Three-layer padding zeroing + marker-embedding mask: correct, no leak.
- Zero-init `intensity_cls_branch[-1]` weight+bias: correct warm-start invariant.
- Class-weight numerics: absent classes → weight 1.0, no div0 (but double-weighting is the BLOCKER).
- Macro-F1 excludes zero-support classes (`_conf_mat_summary`). IQR abstention handles <4 cells / all-equal / non-finite correctly.
- missing_value nan vs 0.0 handled correctly (present-zero vs absent distinguished by mask).

## Strengths
Main-model scatter_ sink-column fix textbook-correct with v0.1.0 compat gating; defense-in-depth padding zeroing; correct zero-init warm-start; macro-F1 zero-support exclusion; robust IQR degenerate-case handling; HierarchicalLoss fp32 wrapping; self-describing checkpoint vocab-ordering guard.
