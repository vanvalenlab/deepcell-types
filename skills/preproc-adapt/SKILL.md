---
name: preproc-adapt
description: Use when a deepcell-types prediction on a single FOV gives a biologically implausible cell-type composition (e.g. neurons in a lymph node, an epithelial organ called mostly immune, one rare type dominating), especially on sparse / low-plex panels or when a channel looks saturated or high-background. Adapts the per-FOV image preprocessing fed to deepcell_types.predict via its preprocess hook; the model weights are never changed.
---

# Composition-Guided Preprocessing Adaptation

A closed loop that tunes the **image preprocessing** fed to `deepcell-types` on a
single FOV until the predicted cell-type composition is biologically plausible for
the tissue and the markers the panel can actually detect. The model and its weights
are never touched — only the per-channel normalization the model sees, via the
`preprocess` hook on `deepcell_types.predict`.

It works because confident-but-wrong predictions on a hard FOV are often driven by an
**input artifact** (a saturated/high-background channel, a wrong clip percentile, a
single confounding marker), not by the model's decision boundary. Fixing the input is
cheap and reversible; retraining is not. The loop also tells you when a failure is
*not* preprocessing-fixable (a panel/coverage gap), so you don't fabricate a fix.

## When to use
- A FOV's predicted composition is implausible (e.g. neurons in a lymph node, an
  epithelial-free organ called mostly Epithelial, one rare type dominating).
- Sparse / low-plex panels (≤~20 markers) where intensity normalization is fragile.
- A specific channel looks noisy/saturated and you suspect it's steering calls.

Do **not** use it to chase a target composition you don't have independent grounds
to expect — see Guardrails.

## Prerequisites
```bash
pip install "deepcell-types @ git+https://github.com/vanvalenlab/deepcell-types@master"
```
Inputs for one FOV (all you need — no archive required):
- `raw`: `(C, H, W)` numpy array of native intensities.
- `mask`: `(H, W)` integer cell-id label image (`0` = background).
- `channel_names`: length-`C` list of marker names, in `raw`'s channel order. They are
  matched to the model's marker registry; names it doesn't recognize are dropped before
  preprocessing runs.
- `mpp`: microns-per-pixel of the image.
- `tissue`: a string for the tissue (used only by you, to write the expectation).

Baseline prediction (the call the loop wraps):
```python
import deepcell_types as dct
labels = dct.predict(raw, mask, channel_names, mpp,
                     model_name="<released-model>", device="cuda:0")
# labels: per-cell predicted cell type, aligned to mask cell ids.
```

## Adapting preprocessing via the `preprocess` hook
`predict` accepts a `preprocess` hook that **replaces** the built-in per-channel
normalization. Express the adaptation as a short, declarative op-pipeline and build the
hook with `make_preprocessor`. Keeping it declarative and bounded is what makes results
comparable and keeps you honest (no hand-written transforms that quietly fit the answer):

```python
from deepcell_types import predict, make_preprocessor, DEFAULT_CONFIG

config = [{"op": "clip_percentile", "p": 99.9}, {"op": "min_max_normalize"}]  # == built-in default
labels = predict(raw, mask, channel_names, mpp, model_name="<released-model>",
                 device="cuda:0", preprocess=make_preprocessor(config))
```

**Hook contract.** `predict` resamples to the model's target MPP and drops
out-of-vocabulary channels *before* calling the hook. The hook then receives the
resampled, in-vocab `raw` as `(C, H, W)` float32 and the resolved **standard** marker
names aligned to it, and must return `(C, H, W)` in `[0, 1]`. `preprocess=None`
(default) runs the built-in p99.9 clip + min-max; `make_preprocessor(DEFAULT_CONFIG)`
reproduces that bit-for-bit, so the baseline round is a true baseline.

> **Do not wrap `preprocess_fov`.** It is the archive-ingestion path, *not* the
> inference path — `predict` never calls it, so wrapping it changes nothing and you
> will misread "no change" as the model being unfixable. Always go through the
> `preprocess` hook.

`channel_drop` / `channel_weight` reference the model's **standard marker names** (what
the hook receives), which may differ from your raw input names.

Available ops (apply in listed order; always end with a normalize so the model sees
`[0,1]`) — all implemented in `deepcell_types.preprocessing_ops` (`apply_config`,
`make_preprocessor`, `DEFAULT_CONFIG`):

| op | params | effect |
|---|---|---|
| `clip_percentile` | `p` | per-channel clip at the p-th percentile of nonzero pixels (tames bright outliers) |
| `log1p` | — | `log1p(max(x,0))` |
| `background_subtract` | `value` | `clip(x - value, 0, None)` — removes a pervasive background floor (one global value for all channels) |
| `background_subtract_per_channel` | `p` (percentile, default 25), optional `names:[...]` | subtract each channel's own p-th-percentile nonzero floor — removes a high-background *pedestal* on one channel without touching clean channels; the principled fix for a saturated/high-background channel |
| `gamma` | `g` | per-channel gamma on `[0,max]` |
| `denoise` | `kind` (median/gaussian), `size` | spatial denoise |
| `hot_pixel_removal` | `z` | clip per-channel hot pixels above `z` MADs |
| `channel_weight` | `weights:{name:w}` | scale named channels (see no-op note) |
| `channel_drop` | `names:[...]` | zero named channels (never removes; masking handles the rest) |
| `min_max_normalize` | — | terminal per-channel min-max to `[0,1]` |

**Down-weighting gotcha.** `channel_weight` *before* a terminal `min_max_normalize`
is a NO-OP (per-channel min-max cancels uniform pre-scaling). To partially suppress a
channel, put `channel_weight` *after* `min_max_normalize`:
`[clip_percentile, min_max_normalize, {"op":"channel_weight","weights":{"X":0.2}}]`.
Use this when a full `channel_drop` over-corrects.

## The per-FOV loop (do these IN ORDER)

> **GATE — do not skip.** Do **not** call `predict` for judgement until
> `expectations.json` exists and is committed (step 1). If predictions already
> exist (e.g. a cohort was scored before you started), pre-registration is still
> possible and still required: write the expectation as a **pure function of
> inputs** — `f(tissue, panel, model class vocab)` — that reads **no** prediction.
> An expectation derived only from inputs cannot be contaminated by the outputs it
> judges, regardless of when you write it. Inline rules written after eyeballing
> predictions are the failure this gate exists to prevent.

### 1. Pre-register the expectation — FROZEN, before any prediction
**First inspect two things — the panel AND the model's class vocabulary.**
- **Panel** (`channel_names`): a lineage with no marker in the panel is undetectable and
  must NOT be in `expected_present`/`expected_dominant_any_of`.
- **Model classes**: the model can only output the cell types in its head. List its output
  class vocabulary (the labels `predict` can return) and the lineage each maps to; do not
  expect a type/lineage the model has no class for (preprocessing can't conjure a missing
  class), and judge the cell-type ratios at the granularity the model actually supports
  (it may carry one generic class for a lineage rather than fine subtypes).

Base the expectation on tissue biology constrained by BOTH (panel ∩ model classes).
Write `expectations.json` and commit it *before* running anything:
```json
{
  "tissue": "intestine",
  "panel_can_detect": ["Epithelial", "Lymphocyte", "Myeloid", "Stromal", "Endothelial"],
  "expected_present":  ["Epithelial", "Lymphocyte", "Myeloid", "Stromal", "Endothelial"],
  "expected_dominant_any_of": ["Epithelial"],
  "implausible_majority": ["Nerve"],
  "must_be_absent": ["Nerve"]
}
```
This is a **rough plausibility criterion, not target fractions.** It is the contract
you judge every later round against — never a post-hoc story.

### 2. Baseline run + channel-quality scan
Run `predict` with `preprocess=make_preprocessor(DEFAULT_CONFIG)` (identical to passing
no hook) and roll the per-cell labels up to broad lineages (group your model's classes
into lineages — e.g. Epithelial, Lymphocyte, Myeloid, Endothelial, Stromal, Nerve,
Tumor, Other). Record lineage fractions, abstention rate, and mean confidence.

**At the same time, scan per-channel stats** (fraction of positive pixels, median, p99).
This is what lets a later edit be *principled instead of target-steering*: a marker
positive in a large fraction of pixels is a near-certain artifact — e.g. a transcription
factor (FoxP3) positive in 87% of pixels cannot be real signal. Note any such channel now;
you'll likely act on it in step 4.

### 3. Verify at BOTH the lineage AND cell-type level
**Success** when ALL hold:
- every `expected_present` lineage appears (dominant ones in non-trivial fraction);
- no `implausible_majority` lineage is the plurality;
- no `must_be_absent` lineage appears above noise;
- **no marker-absent cell type appears at non-trivial fraction.** Check the cell-type
  ratios, not just the lineage rollup — a spurious type can hide inside a *correct*
  lineage (a panel with no mast marker had a third of cells called Mast, which rolls into
  Myeloid, so a lineage-only check passed silently). Flag any type predicted above a small
  fraction (~10%) whose defining marker is absent from the panel.

  **Split marker-absent calls into two kinds — they are not the same.** A marker-absent
  type that is **not** an expected constituent of this tissue (Mast in a no-mast lymph
  node) is a spurious **candidate** → send it to step 4. A marker-absent type that **is**
  expected tissue biology (trophoblast/EVT in decidua with no HLA-G; enterocytes in gut
  with no CDX2; NK in a node with no CD56) is a **coverage caveat** — preprocessing cannot
  conjure a missing marker, so **record it and do not loop it**. Pre-register these
  expected-but-unconfirmable types per tissue so they don't masquerade as spurious;
  conflating the two over-flags massively.

  **This cell-type-ratio check is the single most useful part of the loop** — it
  distinguishes a fixable input artifact from an unfixable model prior.

If it already passes → **STOP, make no edit.** Most FOVs need none; fabricating edits is
the main failure mode.

### 4. Diagnose → change ONE op (match the op's MECHANISM to the cause)
From the panel scan (step 2) and the failing ratios (step 3), infer the most likely *cause*,
then pick the op whose mechanism targets it:

| Likely cause | Op family (mechanism) |
|---|---|
| A channel positive in most pixels / non-specific high background | suppress it: `background_subtract_per_channel` on that channel (keeps the bright real signal, removes the pedestal), `channel_weight` after `min_max_normalize`, or `channel_drop` for full removal |
| The canonical marker of a lineage the panel shouldn't show, driving that lineage | down-weight or drop that marker (`channel_weight` must come *after* `min_max_normalize` to have any effect; `channel_drop` for full removal) |
| A spurious type with a punctate / granular appearance | spatial denoise: `hot_pixel_removal` or `denoise` |
| Bright outliers dominating after normalization | compress dynamic range: lower `clip_percentile` p, or `log1p` |
| A pervasive low background floor across channels | `background_subtract` at ~the floor |

Apply ONE op and re-run steps 2–3. **Any op can over-correct** — it may trade one spurious
majority for another — so accept it only if *multiple* lineages move toward biology (next
section), and prefer the gentlest setting that achieves it. If an op doesn't help or
over-corrects, **reconsider the diagnosis and try a different mechanism** rather than turning
up the same op's strength.

### 5. Stop
Stop on success, after **≤10 rounds**, or when you hit a floor — whichever comes first:
- **Model-prior floor:** a residual that doesn't move under principled edits (e.g. a kidney's
  last ~6% spurious Tumor, or a liver's last ~10% Mast). Don't chase it — flag it and stop;
  it needs retraining, not preprocessing.
- **Coverage gap:** predictions don't move under drastic input changes (an out-of-distribution
  modality saturates to one lineage regardless). Flag for retraining.
Record the final config + before/after composition.

## Cohort / batch mode (many FOVs)
The loop above is per-FOV. To apply it across a whole cohort (e.g. an archive of
hundreds of FOVs already scored), do **not** eyeball predictions and write rules by
hand — that is exactly how pre-registration gets skipped. Instead:

1. **Generate expectations programmatically as a pure function of inputs.** Write a
   small generator that, for each FOV, emits the step-1 expectation from
   `f(tissue, resolved panel, model class vocab)` only — reading **no** prediction.
   Resolve the panel with the *same* path inference uses (`config.resolve_channel_name`)
   so `panel_can_detect` reflects what the model actually sees, and **assert every
   marker name exists in the model registry** (catches typos / phantom markers). Per
   tissue, pre-register `expected_unconfirmable` — expected constituents whose specific
   marker the panel commonly lacks (trophoblast/EVT, gut enterocytes/muscularis, nodal
   NK). **Commit `expectations.json` before flagging** (a sha/commit is your freeze).
2. **Flag against the frozen file**, splitting each flag into a **candidate** (possible
   fixable artifact) vs a **coverage caveat** (marker-absent but expected biology —
   logged, not looped). A well-behaved model on in-distribution tissue should yield a
   *small* candidate set and a larger, explicitly-recorded caveat set. A flagger that
   marks a large fraction of the cohort is over-flagging — usually an incomplete
   defining-marker list or the candidate/caveat split missing.
3. **Also run a channel-artifact scan — composition gates alone are not enough.** A
   composition flagger can only see problems that distort the lineage/cell-type
   *ratios*. It is **blind** to a high-background / saturated channel inflating a call
   that still rolls up into a lineage the tissue is *expected* to be dominated by (a
   high-background CD15 made spleen FOVs ~40% Neutrophil — Myeloid is expected-dominant
   in spleen, so every composition gate passed). So scan every FOV's channels
   independently of composition: per channel compute how *widespread* the signal is and
   its **dynamic range** (`p99/median`). Flag a channel whose median sits far above the
   FOV's typical channel (a background pedestal) with low dynamic range, especially
   when the cell type that marker defines is also over-called. These are extra
   candidates the ratio gates miss.
4. **The artifact signature is NECESSARY but NOT SUFFICIENT — always confirm with the
   loop.** A bright channel can be a real abundant marker, not an artifact. Test each
   scan-flagged channel by suppressing it (targeted `background_subtract_per_channel`,
   or `channel_weight` after `min_max_normalize`) and re-running: trust it only if the
   call drops *and multiple lineages move toward biology*. In practice a high-background
   CD15 was a real artifact (Neutrophil 41%→~15%, endothelial/macrophage/lymphocyte all
   rose), while equally-bright ASMA (gut muscularis) and OLFM4 (gut epithelium) were
   robust to suppression — real biology. Acting on the scan alone would have fabricated
   those edits; only the loop edit tells artifact from a genuinely bright marker.
5. **Run the per-FOV loop only on the genuine candidates** (composition- and
   scan-flagged). For coverage caveats, panel inspection already settles it (you can't
   `background_subtract` your way to a marker that isn't there) — no run needed; record
   and move on.

This is the same discipline as the single-FOV loop, just with the expectation frozen
in bulk up front. The integrity property is identical: the criterion is a function of
inputs, committed before any prediction is judged.

## Guardrails (load-bearing)
- **Pre-registration first (see the GATE at the top of the loop).** If you didn't
  freeze `expectations.json` before judging any prediction, you can't trust the result —
  redo. If predictions already exist, this is still required: write the expectation as a
  prediction-blind function of inputs and commit it. Rules written after eyeballing
  outputs are not a pre-registration.
- **Reject degenerate edits.** Log abstention + mean confidence every round. An edit
  that only inflates confidence or collapses lineage diversity to hit the expectation,
  without genuinely improving plausibility, is rejected.
- **Prefer artifact-removal over target-steering.** Trust an edit most when it moves
  *multiple* lineages toward biology at once, or fixes the same confound across
  tissues with *opposite* expected compositions — that's removing a real artifact, not
  fitting the answer. Before dropping a channel, get independent evidence it's an
  artifact (e.g. a transcription-factor marker positive in most pixels cannot be real).
- **Judge at BOTH lineage and cell-type level.** A spurious type can hide inside a
  *correct* lineage — e.g. a panel with no mast marker still gets ~a third of cells called
  Mast, which rolls into Myeloid, so a lineage-only check passes silently. Flag any cell
  type predicted above a small fraction whose defining marker is absent from the panel; it
  is almost always spurious.
- **Prefer the gentlest effective op; re-diagnose instead of escalating.** If an edit
  doesn't help or over-corrects, switch to a different mechanism rather than just turning up
  the same op's strength — a stronger version of the wrong op usually makes things worse.
- **Bounded ops only.** If a needed op doesn't exist, add it to
  `deepcell_types.preprocessing_ops` (with a test) rather than hand-writing a one-off
  transform.
- **Some failures are not preprocessing-fixable.** If the panel lacks the markers to
  detect the expected biology, or the model is out-of-distribution for the
  modality/panel (predictions don't move under drastic input changes), STOP and report
  it as a coverage/spec gap — the fix is retraining or a panel change, not preprocessing.

## Generalizable diagnostic patterns (from a multi-modality study)
- **A single confounding marker.** On a sparse MIBI series, one channel (a neuronal
  marker) drove spurious "Nerve" everywhere and spurious stroma in lymphoid organs;
  dropping it fixed tissues with opposite expected compositions at once — the
  opposite-direction test confirmed a genuine artifact, not steering.
- **High-background channel → one type everywhere.** A kidney CODEX FOV called ~52% of
  cells one rare T-cell subtype because its transcription-factor channel (FoxP3) was
  positive in 87% of pixels (impossible for a TF). **Dropping that one channel** flipped it
  to correct Epithelial-dominant kidney (52%→0 Treg, Epithelial 22%→47%); a milder
  background-subtract only got halfway. A second independent model made the same error,
  confirming the artifact was in the image. Lesson: drop the proven-artifact channel
  outright, and pick the op by signature rather than reaching for a global transform.
- **Reduce the artifact, but expect a model-prior floor.** Fixes are often partial: after
  the kidney drop, ~6% spurious Tumor remained and would not move under further principled
  edits; a liver Mast confound fell 2–3× but left a ~10% residual. The loop's job is to
  remove the *artifact-driven* portion and flag the rest — not to force the number to zero
  (which only invites target-steering).
- **Panel can't see the dominant cell.** An immune-focused liver panel had no
  hepatocyte marker, so an "Epithelial-dominant" expectation was naive; the model's
  immune-dominant calls were correct. Always write the expectation panel-aware.
- **Out-of-distribution modality = hard prior.** A modality barely present in training
  saturated to one lineage regardless of input edits (dropping even the relevant
  markers didn't move it). Preprocessing can't fix a coverage gap — flag for retraining.
