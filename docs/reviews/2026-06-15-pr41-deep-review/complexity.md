# Complexity & Maintainability Audit — deepcell-types v0.1.0 (PR #41)

(1 blocker, 4 highs, 4 mediums, 1 low)

## BLOCKER 1: `--no_class_weights` / `use_weighted_sampler` permanently coupled — the sampler can't be disabled
**Location:** `scripts/train.py:408, 290-293, 616-617`
`use_weighted_sampler=True` hardcoded; `--no_class_weights` toggles only FocalLoss alpha. The default (no flag) = class weights ON + sampler ON = the double-weighting the flag's own help warns against. Causal relationship documented in the help/CKPT_CONFIG is permanently false.
**Recommendation:** Plumb `use_weighted_sampler=not no_class_weights`, OR (maintainer's plan) remove the sampler entirely and hardcode the interlock.

## HIGH 2: `tissue_idx` collected, checkpoint-gated, and passed through collation but never used in loss/model
**Location:** `dataset.py:748-814`, `utils.py:163`, `train.py` (no `batch_data.tissue_idx` ref)
The dataset now *requires* valid tissue (raises ValueError) but the tensor is dead in the training loop — incomplete feature or validation scaffolding.
**Recommendation:** Wire it into the model or remove the tensor + document the data-quality gate.

## HIGH 3: `main()` is a ~600-line god function with no seams (`train.py:341-945`)
Setup+data+model+freeze+optim+loop+eval+save in one function; freeze block matches module names as string literals.
**Recommendation:** Extract `_build_model_from_args`, `_build_optimizer_and_scheduler`, `_run_epoch`, `_checkpoint_save/load`.

## HIGH 4: Fragile 3-level dataset unwrapping `train_loader.dataset.dataset.dataset` (`train.py:432-452`)
Guards convert silent AttributeError to TypeError, but any future wrapper change (notably removing the sampler) breaks the isinstance chain.
**Recommendation:** Return the `FullImageDataset` directly from `create_dataloader` as metadata.

## MEDIUM 5: Lazy import of `_discover_fov_keys` inside `_load_tissuenet_archive` — circular-import workaround (`dataset.py:173`; cf branch `fix/dataloader-dataset-circular-import`).
## MEDIUM 6: `DATA_DIR` module-level constant baked into Click defaults at import time (`train.py:57, 203`) — breaks under monkeypatch/spawn. Use `envvar=` or resolve in callback.
## MEDIUM 7: Mean-intensity computation duplicated between model.py and the dataset/predict preprocessing — risks drift (`model.py:451-458`).
## MEDIUM 8: `DataLoaderConfig` missing `fov_grouped_train` field → `create_dataloader_from_config` can't activate it (`dataloader.py:290-322`).

## LOW 9: Freeze/unfreeze matches model module names as string literals (`train.py:496-503`) — assert found-set == expected-set.

## Strengths
Wrapper-depth guards use `raise TypeError` (not assert, survives `-O`). Atomic checkpoint saves everywhere. Cache pickle-injection hardening. `compat_marker0_zero` clean compat flag. `DataLoaderConfig` good API hygiene. Cache fingerprint validation (not mtime). `_build_model` reconstructs most hyperparams from shapes; documents the two un-recoverable ones.
