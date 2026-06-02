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
| `arcsinh` | `cofactor` (default 5) | compress dynamic range: `arcsinh(x/cofactor)` |
| `log1p` | — | `log1p(max(x,0))` |
| `background_subtract` | `value` | `clip(x - value, 0, None)` — removes a pervasive background floor |
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

### 1. Pre-register the expectation — FROZEN, before any prediction
**First inspect the panel** (`channel_names`). Base the expectation on tissue biology
**as constrained by what the panel can detect**: a lineage with no marker in the panel
is undetectable and must NOT be in `expected_present`/`expected_dominant_any_of`.
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

### 2. Baseline run with `DEFAULT_CONFIG`, then evaluate composition
Run `predict` with `preprocess=make_preprocessor(DEFAULT_CONFIG)` (identical to passing
no hook) and roll the per-cell labels up to broad lineages (group your model's classes
into lineages — e.g. Epithelial, Lymphocyte, Myeloid, Endothelial, Stromal, Nerve,
Tumor, Other). Record lineage fractions, abstention rate, and mean confidence.

### 3. Compare to the FROZEN expectation
**Success** when ALL hold: every `expected_present` lineage appears (dominant ones in
non-trivial fraction); no `implausible_majority` lineage is the plurality; no
`must_be_absent` lineage appears above noise.

### 4. Diagnose → change ONE op
Pick the single op that addresses the dominant mismatch, reasoning from biology and
the marker image:
- epithelium missing despite a bright epithelial marker → lower `clip_percentile` p,
  or `arcsinh` to compress outliers.
- a pervasive high-background channel driving one type everywhere →
  `background_subtract` at ~the channel's background floor.
- a noisy channel → `denoise` / `hot_pixel_removal`.
- a clearly uninformative/saturated/confounding channel → `channel_drop` (or
  down-weight if a full drop over-corrects).
Re-run steps 2–3.

### 5. Stop
On success, or after **6 rounds**, whichever comes first. Record the final config and
the before/after composition.

## Guardrails (load-bearing)
- **Pre-registration first.** If you didn't freeze `expectations.json` before step 2,
  you can't trust the result — redo.
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
  is almost always spurious. (Often driven by hot pixels mimicking granular cells —
  `hot_pixel_removal` cuts it without the dynamic-range distortion `arcsinh` causes.)
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
  cells one rare T-cell subtype because its transcription-factor channel was positive
  in 87% of pixels (impossible for a TF). A background subtraction at the measured
  floor halved it and let epithelium/endothelium re-emerge — and notably a second
  independent model made the same error, confirming the artifact was in the image.
- **Panel can't see the dominant cell.** An immune-focused liver panel had no
  hepatocyte marker, so an "Epithelial-dominant" expectation was naive; the model's
  immune-dominant calls were correct. Always write the expectation panel-aware.
- **Out-of-distribution modality = hard prior.** A modality barely present in training
  saturated to one lineage regardless of input edits (dropping even the relevant
  markers didn't move it). Preprocessing can't fix a coverage gap — flag for retraining.
