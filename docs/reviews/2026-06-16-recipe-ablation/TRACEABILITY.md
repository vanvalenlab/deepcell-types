# Traceability: reported number → checkpoint → code commit → prediction CSV

Every headline number in `REPORT.md` (and `docs/reviews/2026-06-16-recipe-ablation/`)
maps to a specific checkpoint (sha256), the code commit it was trained AND evaluated
under, and the prediction CSV (sha256) the number was computed from.

**Common to all rows:** archive `expanded-tissuenet.zarr` fingerprint `f5b6ed52`;
test split = 129 FOVs / 486,705 cells (`fov_split_test_eval.json`, val=129 test);
metric = repo `hierarchical_macro_f1` (parent→child credit); full coverage
(`--ct_abstention_k 0`, no abstention). Both `d13fd54` and `b598710` are ancestors
of `xuefei/master` (PR #41), so they resolve in repo history.

| # | Reported (test, hier raw) | Checkpoint (sha256, head, val) | Train commit | Eval commit | Prediction CSV (sha256) |
|---|---|---|---|---|---|
| 1 | two-stage `resmlp` — **80.27** | `model_dct_resmlp_best.pt` `3d8074a8…` (resmlp) | pre-session ✦ | main checkout, pre-`b598710` | `dct_resmlp_test_prediction.csv` `04b1c18b…` |
| 2 | from-scratch resMLP+lr1e-3 — **74.96** | `model_dct_scratch_resmlp_best.pt` `904fb40b…` (resmlp, val 0.7493) | **`b598710`** | **`b598710`** | `scratch_resmlp_test_prediction.csv` `d70fad76…` |
| 3 | from-scratch MLP+lr3e-4 — **74.10** | `model_dct_scratch_nopt_best.pt` `13c8dde9…` (mlp, val 0.7346) | **`d13fd54`** | **`d13fd54`** | `scratch_nopt_test_prediction.csv` `37b96593…` |
| 4 | XGBoost-tuned — **79.03** | XGBoost model (`dct-final-ckpt/`, 2026-05-17) | pre-session | pre-session | `baseline_xgb_tuned_test_prediction.csv` `3bcc7c30…` |

✦ Row 1: the two-stage `resmlp` checkpoint was produced pre-session by `resmlp_final.py`
on pre-dumped frozen-backbone features (not a single clean `train.py`/`retrain_head.py`
commit). Its backbone is bit-identical (0.0 max-diff over 129 tensors) to the stage-1
checkpoints `model_dct_{headfix,final_noclsw}_best.pt`. Its config records `ct_head_arch`
+ `n_celltypes` but **no git hash** — so its exact training commit is not pinned (see gap
below). The eval CSV was generated on the main repo checkout before `b598710`; the resMLP
head still loads there because the checkpoint config carries `n_celltypes`.

## Pin → commit (the code each run executed)
- `/tmp/dct-pin-feat` → `d13fd54043287c2cb57a19c9b16f79ace7cc1ff1` (rows 3; legacy-MLP eval path).
- `/tmp/dct-pin-resmlp` → `b59871019cadb5dd49bf4741810e2518a8bba28f` (row 2; has the
  head-agnostic `n_celltypes` inference needed to load a resMLP checkpoint).

## Code snapshot per number
- Rows 2 & 3 are fully pinned: checkpoint sha256 + train/eval commit + CSV sha256.
- The recipe/data hyperparameters are ALSO embedded in each checkpoint's `config`
  (`*_config.json` here) — `ct_head_arch`, `lr`, `focal_gamma`, `domain_weight`,
  `no_class_weights`, `split_file`, `svd_embeddings_path`, `archive_fingerprint`.

## Self-pinning (gap now closed)
The rows above predate self-pinning, so their train/eval commits were reconstructed from
the pin worktrees' HEADs (this manifest is their source of truth). Going forward this is
automatic: `scripts/train.py` now records `CKPT_CONFIG["git_commit"] = git rev-parse HEAD`
of the running checkout — implemented in **`ef1229f`** (PR #41). Checkpoints trained from
that commit onward carry their own code commit in `config["git_commit"]`.
