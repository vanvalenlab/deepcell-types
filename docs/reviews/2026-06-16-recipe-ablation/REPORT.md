# Recipe ablation: two-stage `resmlp` vs from-scratch — what buys the headline?

**Date:** 2026-06-16/17 · **Archive:** `expanded-tissuenet.zarr` (fingerprint `f5b6ed52`) · **Test set:** held-out 129 FOVs / 486,705 cells (`fov_split_test_current.json`).

All cell-type macro-F1 numbers below are computed on the **same frozen test set**, full-coverage (no abstention) unless a `k` is given, using the repo's own `hierarchical_macro_f1` (parent→child credit) and the shared `_conf_mat_summary` reducer — so they are directly comparable across methods.

## Question

PR #41's headline (`~79.1` full-coverage macro-F1) is produced by the **two-stage `resmlp` recipe**: train a backbone with the weighted sampler on, then *freeze it* and retrain a residual-MLP cell-type head on the natural class distribution (sampler off, plain CE). A natural question for the paper: **is that edge actually due to the two-stage training procedure, or is it confounded by other differences** between the `resmlp` checkpoint and a plain from-scratch model?

## Step 1 — Confound audit (6-agent comparison)

We compared `model_dct_resmlp_best.pt` against a from-scratch single-stage checkpoint across six dimensions (architecture, backbone provenance, recipe, eval harness, preprocessing, data/splits). Findings, all evidence-verified:

**Identical (verified):**
- **Backbone + all auxiliary branches** — 132 shared state-dict keys, 0 shape mismatches (transformer, encoders, fusion, DANN head, intensity-CLS branch, marker heads).
- **Eval harness & test cohort** — same `predict.py`, same 486,705 cells, 100% ground-truth match, same 51-class schema, same full-coverage setting (`--ct_abstention_k 0`).
- **Preprocessing / inputs** — same SVD embeddings (`svd_512.npz`), same archive `f5b6ed52`, same p99.9 clip + normalization + distance transform, 0 eval-dropout; data-path code byte-identical.
- **Data** — same 51-class vocab + ordering, same 1722/431 train/val split, same 129-FOV test cohort, no train/test leakage.

**Two confounds found (NOT just the recipe):**
1. **Cell-type head architecture** — `resmlp` uses a `ResidualMLPHead` (256→512, BatchNorm, 4 residual blocks); from-scratch used the legacy 3-layer MLP (256→256→128→51).
2. **Backbone learning rate** — `resmlp`'s backbone was trained at `lr=1e-3`; the from-scratch model used `lr=3e-4` (3.3×). (Verified: the `resmlp` checkpoint's backbone is bit-identical, 0.0 max-diff over 129 tensors, to the stage-1 backbone checkpoint.)

So the original `resmlp` (80.27 hier) vs from-scratch (74.10 hier) comparison bundled the two-stage procedure **with** a bigger head and a different lr — not a clean recipe-only ablation.

## Step 2 — Controlled experiment

We trained a from-scratch single-stage model with the **two confounds removed**: `--ct_head_arch resmlp --lr 1e-3`, everything else identical to the original from-scratch run (`focal_gamma=2.0`, `domain_weight=0.1` DANN on, `--no_class_weights`, weighted sampler on, `max_samples_per_epoch=500000`, 50 epochs / patience 10, seed 42, same split/archive/SVD). It now differs from the two-stage `resmlp` model **only by the two-stage procedure**.

## Results (test set, hierarchical macro-F1)

| Setup | hier (raw) | flat (raw) | hier @k=0.2 |
|---|---|---|---|
| two-stage **`resmlp`** (head retrain) | **80.27** | 76.82 | 87.28 |
| XGBoost-tuned | 79.03 | 75.62 | — |
| from-scratch, **resmlp-head + lr1e-3** (controlled) | 74.96 | 71.95 | 83.37 |
| from-scratch, MLP-head + lr3e-4 (original) | 74.10 | 71.46 | 83.34 |
| XGBoost-plain | 77.43 | 74.06 | — |
| from-scratch `bnoclsw` (focal 0, no DANN) | 71.71 | 66.83 | 80.57 |

Controlled run full sweep (test, 486,705 cells): raw flat 71.95 / hier 74.96; k=1.5 → 73.12 / 76.11; k=0.5 → 77.18 / 80.29; k=0.2 → 80.17 / 83.37. (Val hier macro-F1 at selection: 0.7493.)

## Conclusions

1. **Removing the two confounds barely moved the from-scratch result: 74.10 → 74.96 hier (+0.86 pp).** The head architecture and the backbone learning rate were **not** the source of the `resmlp` model's lead.
2. **The ~5.3 pp gap (80.27 vs 74.96 hier) is cleanly attributable to the two-stage *procedure* itself** — freeze the sampler-trained backbone, then retrain the residual head on the natural distribution. This corroborates the documented rationale that end-to-end single-stage training "erodes the backbone," now verified with head + lr held constant.
3. **Single-stage DCT does not beat XGBoost-tuned** even with the residual head and matched lr (74.96 < 79.03 hier). **Only the two-stage `resmlp` clears XGBoost-tuned — by ~1.2 pp (80.27 vs 79.03)**, a slim margin.

## Caveats

- The 129 test FOVs are a frozen subset of the same 431-FOV validation pool both models used for checkpoint selection (symmetric across methods, but not a fully independent third holdout).
- All headline numbers here are **full coverage (no abstention)** — the fair comparison, since the baselines do not abstain. DCT's default post-hoc abstention (`k=0.2`, applied only to DCT) is a separate axis flagged as a comparison-fairness concern in the PR #41 deep review (see `../2026-06-15-pr41-deep-review/SYNTHESIS.md`, "Comparison fairness"). With abstention applied to DCT only, the controlled from-scratch reaches 83.37 hier @k=0.2 while XGBoost stays at its full-coverage 79.03 — that gap is the asymmetric-abstention artifact, not a real lead.

## Reproducibility

- **Checkpoints:** two-stage `models/model_dct_resmlp_best.pt`; controlled from-scratch `/tmp/dct_scratch/models/model_dct_scratch_resmlp_best.pt`; original from-scratch `/tmp/dct_scratch/models/model_dct_scratch_nopt_best.pt`. (Backbone shared with `model_dct_{headfix,final_noclsw}_best.pt`, bit-identical.)
- **Controlled run:** `scripts/train.py --ct_head_arch resmlp --lr 1e-3 --focal_gamma 2.0 --domain_weight 0.1 --no_class_weights --resnet_channels 48 --epochs 50 --patience 10 --max_samples_per_epoch 500000 --max_val_samples 200000 --split_file fov_split_current.json` on archive `f5b6ed52`.
- **Eval:** `scripts/predict.py --model_path <ckpt> --split_file <test split, val=129 FOVs> --ct_abstention_k 0`, then `hierarchical_macro_f1` over the prediction CSV. Loading a `resmlp` checkpoint requires the head-agnostic cell-type-count inference added in PR #41 (`b598710`).
- XGBoost test CSVs: `dct-final-ckpt/baseline_xgboost_test_prediction.csv` (tuned) on the bit-identical 129-FOV test set.
