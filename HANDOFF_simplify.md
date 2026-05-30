# Handoff — PR #41 simplification (branch `refactor/simplify-pr41`)

Status as of this writing. Three cleanup passes are **done, verified (ruff clean,
224 passed / 10 skipped), and pushed**:

1. `refactor: remove dead code and dedup checkpoint warm-start loop`
   — dead `extract_archive`, dead config constants, dead `_build_centroid_tree`
   / `_idx2marker`, and the 3× checkpoint warm-start loop deduped into
   `deepcell_types/training/utils.py::load_matching_state_dict`.
2. `refactor(config): drop config members unused anywhere in the repo`
   — removed in-repo-unused members of `DCTConfig` / `TissueNetConfig`
   (each grep-proven unused).
3. `refactor(training): split dataset god-file into focused modules`
   — `training/dataset.py` 1875 → 884 lines; extracted `transforms.py`,
   `samplers.py`, `splits.py`, `dataloader.py`; `dataset.py` re-exports every
   symbol so the public import surface is unchanged.

> ✅ **Branch is clean and linear** as of this handoff — local == remote
> (`xuefei/refactor/simplify-pr41`, tip `a126fac`), no divergence, no duplicate
> commits. The parallel session's `model.py` work has landed: `eb0ec31`
> "remove unused tumor prediction head and drop tumor_logit from model output"
> and `a74b64b` "rename PreNormTransformerEncoderLayer to
> ChannelWiseTransformerEncoderLayer". `model.py` is still a SINGLE file (it was
> NOT split). B and C below were **NOT started** — `model.py` is unmodified
> relative to `a126fac`. Before starting, re-confirm: `git fetch xuefei &&
> git log --oneline -3 && git diff --stat HEAD -- deepcell_types/model.py`
> (should be empty).

---

## Method contract (applies to every task below)

- **Behavior-preserving.** No change to model numerics, RNG, tensor math,
  normalization, scatter/gather, file formats, or published-checkpoint outputs.
- **Verify each task green before committing:**
  - `ruff check deepcell_types scripts`  → must be "All checks passed!"
  - `python -m pytest tests/ -q -p no:cacheprovider`  → pass/skip counts must
    match the pre-change baseline (currently 224 passed / 10 skipped).
- **Edit via Bash (python str.replace with count assertions) for existing
  files** — there is a PostToolUse formatter hook on Edit/Write that reflows
  whole files and creates huge noisy diffs. Brand-new files via Write are fine.
- **No force-push.** Push fast-forward only; if rejected, fetch + rebase, and if
  the rebase conflicts, abort and reconcile manually.
- Commit-message trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## TODO B — `CellTypeAnnotator.forward()` returns → NamedTuple  (NOT STARTED)

**Goal:** replace the positional 6/7-tuple return of
`CellTypeAnnotator.forward()` with a `typing.NamedTuple` (e.g. `AnnotatorOutput`)
so call sites read by name instead of by position, and the optional
attention-weights field is an explicit `Optional` rather than a length change.

**Current state (verified at `a126fac`).** `forward` has TWO return paths,
both at module-coordinates `deepcell_types/model.py:556` and `:565`:
- non-attn path (`return_attn_weights=False`): returns a 5-tuple
  `(ct_logits, domain_logits, marker_pos_logits, cls_embedding, channel_outputs)`
- attn path (`return_attn_weights=True`): returns the same 5 PLUS
  `cls_to_channels` as the 6th element.
(NOTE: `tumor_logit` was already removed by `eb0ec31`, so there is NO 6/7-tuple
with a tumor field anymore — it's 5 / 6. Re-read lines ~540-566 to confirm exact
field order before coding.)

**Where it's defined:** `deepcell_types/model.py`, `CellTypeAnnotator.forward`
(grep `return (` / `return ct_logits`). Note the two return paths
(`return_attn_weights` True/False).

**Call sites to update (find with):**
`grep -rn "= model(" scripts deepcell_types tests` and
`grep -rn "model.forward\|annotator(" scripts deepcell_types tests`
Known consumers: `deepcell_types/predict.py` (inference unpack),
`scripts/train.py::forward_one_batch`, `scripts/pretrain.py`,
`scripts/predict.py`, `tests/test_v2.py`, `tests/test_canonical_inference.py`.

**Approach (behavior-preserving):**
1. Define `class AnnotatorOutput(NamedTuple):` with the exact current fields in
   order (so existing tuple-unpacking `a, b, c = out` still works — NamedTuple
   IS a tuple). Make the attention field `Optional[...] = None` and ALWAYS return
   the same arity, OR keep the two-arity behavior but document it; pick whichever
   keeps every current unpack site working. The safest: always return the full
   NamedTuple with `cls_to_channels=None` when attn weights aren't requested, and
   update the (few) sites that currently unpack the shorter tuple.
2. Convert unpack sites to attribute access where it improves readability
   (`out.ct_logits`), but this is optional — NamedTuple keeps positional unpack
   valid, so a minimal diff is possible.
3. `ruff` + full `pytest` green; `tests/test_v2.py` exercises the forward output
   directly and is the key guard.

**Risk:** broad (touches every caller) but mechanical. The NamedTuple-is-a-tuple
property means you can do it incrementally without breaking positional unpacks.

---

## TODO C — `**kwargs` → explicit params in model constructor  (NOT STARTED — do this FIRST)

**Goal:** `CellTypeAnnotator.__init__` and `create_model` currently funnel real
architecture params through `**kwargs` (`spatial_pool_size`,
`resnet_base_channels`, `mean_intensity_mode`, `compat_marker0_zero`, possibly
others after the parallel split). Make them explicit keyword params with their
current defaults so they're discoverable, type-hintable, and validated — **no
default-value changes**.

**Where (verified at `a126fac` — re-grep `kwargs.get\|kwargs.pop` before editing
in case line numbers drift):**
`CellTypeAnnotator.__init__(self, d_model=256, n_heads=8, n_layers=4,
n_celltypes=51, n_domains=8, marker_embeddings=None, dropout=0.1, **kwargs)` reads:
- `spatial_pool_size = kwargs.get("spatial_pool_size", 1)`            (~line 278)
- `resnet_base_channels = kwargs.get("resnet_base_channels", 48)`     (~line 286)
- `self.mean_intensity_mode = kwargs.get("mean_intensity_mode", "cls_residual")` (~line 348)
- `self.compat_marker0_zero = bool(kwargs.get("compat_marker0_zero", True))` (~line 356)
  — CRITICAL: keep default `True` (v0.1.0 checkpoint compat; see the long comment
  in `forward`). Do NOT flip.

`create_model(dct_config, marker_embeddings, d_model=256, n_heads=8, n_layers=4,
dropout=0.1, use_conditioned_mp_head=True, **kwargs)` reads:
- `n_celltypes = kwargs.pop("n_celltypes", dct_config.NUM_CELLTYPES)` (~line 656)
- `n_domains = kwargs.pop("n_domains", dct_config.NUM_DOMAINS)`       (~line 657)
then forwards `**kwargs` to `CellTypeAnnotator` (~line 666). Promote all of the
above to explicit params with these EXACT defaults.

**Approach:**
1. Inventory every `kwargs.get(...)` / `kwargs.pop(...)` in `model.py` and the
   exact default each uses. Promote each to an explicit `param=<same default>` in
   the signature; replace the `kwargs.get` body reads with the param name.
2. `create_model` forwards them explicitly (or keeps `**kwargs` passthrough only
   for genuinely-variadic cases — but prefer explicit).
3. CRITICAL: keep `compat_marker0_zero=True` default (v0.1.0 checkpoint compat —
   see the long comment in `forward`). Do not flip any default.
4. Check checkpoint-driven construction in `predict.py::_build_model` still
   passes the same names; `tests/test_checkpoint_roundtrip.py` and
   `tests/test_canonical_inference.py` are the guards.
5. `ruff` + full `pytest` green.

**Risk:** low, behavior-preserving. Lowest-risk of the remaining items —
good first task once `model.py` is stable.

---

## TODO A — split `training/config.py` god-file (948 lines)

**Goal:** same extract-and-re-export pattern as the `dataset.py` split. Reduce
`training/config.py` size by moving cohesive groups into sibling modules while
`training/config.py` re-exports every public symbol (so
`from deepcell_types.training.config import TissueNetConfig, WARMUP_PCT,
CELL_TYPE_HIERARCHY, ...` keeps working unchanged).

**Why deferred:** the parallel session was active in config/model; do this once
that work has landed and the branch is reconciled.

**Suggested module groups (adjust to actual dependency graph):**
- embeddings loading (`load_marker_embeddings_array`, SVD/JSON readers,
  `LazyMarkerPositivityDict`) → `training/embeddings.py`
- hierarchy / cell-type-mapping constants (`CELL_TYPE_HIERARCHY`,
  combined/celltype mapping helpers) → `training/hierarchy.py` or keep in config
- the zarr-archive registry reads that stayed in `TissueNetConfig`
Keep `TissueNetConfig` itself in `config.py`; move only the standalone helpers
it can import back.

**Guards:** import-surface check (every name still importable from
`training.config`), `ruff`, full `pytest`. Watch for circular imports with
`training/archive.py` (already a dependency) and `training/dataset.py`.

**Known live members (do NOT break):** `TissueNetConfig`, `WARMUP_PCT`,
`CELL_TYPE_HIERARCHY`, `load_marker_embeddings_array`, `marker_positivity_labels`,
`tissue2idx`, `celltype_mapping`, `domain_mapping`, `ct2idx`, `domain2idx`,
`marker2idx`, `combined_celltype_mapping` (used by scripts/tests). Re-grep before
moving; `scripts/train.py`, `scripts/pretrain.py`, `baselines/*/run.py`, and
several tests import from here.

---

## TODO D — behavior-changing / judgment items (need explicit sign-off)

These were intentionally NOT done because they change behavior or public surface.
Each needs a human decision, not a blind refactor.

1. **Unify the two preprocessing percentile paths.**
   `deepcell_types/preprocessing.py` has `_percentile_threshold` (NaN-percentile,
   used by `preprocess_fov`) AND `_percentile_threshold_nonzero` (nonzero-indexing,
   used by the legacy `patch_generator` that the published checkpoint was trained
   against). They are deliberately NOT unified — the inference path must stay
   bit-compatible with the trained checkpoint. **Do not merge without retraining.**
   See the comment block above `_normalize_per_channel` in `preprocessing.py`.

2. **Shadowed module-name pairs** — `config.py`/`training/config.py`,
   `dataset.py`/`training/dataset.py`, `abstention.py`/`training/abstention.py`.
   These are an INTENTIONAL inference-vs-training dependency split (keeps the
   inference path numpy-only, no `[train]` extras). Renaming would break public
   import paths and the baselines. **Recommendation: leave as-is.** Filename-only
   navigation hazard; the classes are uniquely named (DCTConfig vs
   TissueNetConfig, PatchDataset vs FullImageDataset).

3. **Remaining public-looking but in-repo-unused config members.** Some
   `TissueNetConfig` members are unused within this repo but read like a stable
   public API external notebooks may depend on. Removing them is a public-API
   break — get sign-off and a deprecation cycle rather than deleting outright.

4. **`forward()` still returns several heads** (domain_logits, marker_pos_logits,
   cls_embedding, channel_outputs) that not all callers use. Trimming the return
   is a behavior/contract change (training needs them; inference doesn't) — only
   do this with a clear caller audit, ideally after TODO B makes the output a
   named structure.

---

## Quick status / sync commands

```bash
cd /data/xwang3/Projects/deepcell-types
git fetch xuefei
git log --oneline --graph -15 HEAD xuefei/refactor/simplify-pr41   # check divergence/dupes
ruff check deepcell_types scripts                                   # must be clean
python -m pytest tests/ -q -p no:cacheprovider                     # baseline 224 passed / 10 skipped
```

**Environment note:** the tool channel in the authoring session was intermittently
degraded (empty/burst tool output), correlated with very high host load
(load avg peaked 45–63). If tool calls stall, check `uptime` and consider
restarting the session / waiting for load to drop.
