"""
XGBoost hyperparameter tuning using Optuna.

Performs Bayesian optimization over core XGBoost hyperparameters:
- n_estimators, max_depth, learning_rate
- min_child_weight, subsample, colsample_bytree
"""

import os
import click
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb


# Default data directory from environment
DATA_DIR = Path(
    os.environ.get("DEEPCELL_TYPES_ZARR_PATH") or os.environ.get("DATA_DIR", "")
)

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.baseline_features import (
    extract_features_from_zarr,
    compute_baseline_metrics,
    save_baseline_predictions,
)
from deepcell_types.training.samplers import compute_sample_weights_dct


class XGBoostObjective:
    """Optuna objective for XGBoost hyperparameter tuning."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        metric: str = "macro_accuracy",
        hierarchy: dict = None,
        ct2idx: dict = None,
        device: str = "cpu",
        verbose_trial: bool = False,
        class_balance: str = "dct",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.metric = metric
        self.hierarchy = hierarchy
        self.ct2idx = ct2idx
        self.device = device
        # When True, print the per-trial fit train size + fixed n_classes (used
        # by the canonical-val protocol to evidence subsample size + a stable
        # label space across trials). Default False leaves legacy output intact.
        self.verbose_trial = verbose_trial
        self.class_balance = class_balance
        # Per-row DCT sampler weights (sqrt-inverse-frequency, 1000-count floor),
        # the tree analog of the neural baselines' WeightedRandomSampler; None
        # for faithful unweighted XGBoost. normalize=True (mean 1) so tuning
        # optimizes at the same scale the final model ships with (see run.py).
        self.sample_weight = (
            compute_sample_weights_dct(y_train, normalize=True)
            if class_balance == "dct"
            else None
        )

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to maximize
        """
        if self.verbose_trial:
            print(
                f"  [trial] fit train rows={len(self.X_train)}, "
                f"n_classes={self.num_classes} (fixed = full-train class count)"
            )
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Fixed parameters
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "tree_method": "hist",
            "device": self.device,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Train model with early stopping
        early_stopping_rounds = max(10, params["n_estimators"] // 10)
        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weight,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_prob = model.predict_proba(self.X_val)

        metrics = compute_baseline_metrics(
            self.y_val,
            y_pred,
            y_prob,
            self.num_classes,
            hierarchy=self.hierarchy,
            ct2idx=self.ct2idx,
        )

        return metrics[self.metric]


def run_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    n_trials: int = 100,
    metric: str = "macro_accuracy",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    hierarchy: dict = None,
    ct2idx: dict = None,
    device: str = "cpu",
    verbose_trial: bool = False,
    class_balance: str = "dct",
) -> Tuple[optuna.Study, dict]:
    """
    Run hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_classes: Number of classes
        n_trials: Number of trials to run
        metric: Metric to optimize
        study_name: Name for the Optuna study
        storage: Optional database URL for distributed tuning

    Returns:
        study: Optuna study object
        best_params: Best hyperparameters found
    """
    if study_name is None:
        study_name = f"xgboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Create objective
    objective = XGBoostObjective(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes,
        metric,
        hierarchy=hierarchy,
        ct2idx=ct2idx,
        device=device,
        verbose_trial=verbose_trial,
        class_balance=class_balance,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study, study.best_params


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    num_classes: int,
    device: str = "cpu",
    hierarchy: dict = None,
    ct2idx: dict = None,
    train_fov_names: Optional[np.ndarray] = None,
    eval_set_external: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    class_balance: str = "dct",
) -> Tuple[xgb.XGBClassifier, dict]:
    """
    Train final model with best parameters.

    Early stopping uses a FOV-grouped inner-validation set carved out of the
    training data (or, if FOV names are not provided, a stratified inner-val).
    The held-out (X_test, y_test) is reserved for the reported metric only —
    it must NOT be passed as ``eval_set``, since that would let
    ``early_stopping_rounds`` select ``best_iteration`` to minimise loss on
    the same data we report metrics on (leakage into ``test_metrics``).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (reported metric only — never used for early stop)
        y_test: Test labels (reported metric only — never used for early stop)
        best_params: Best hyperparameters from tuning
        num_classes: Number of classes
        device: Device for training (cpu or cuda:N)
        hierarchy: Cell type hierarchy for hierarchical eval
        ct2idx: Cell type to index mapping
        train_fov_names: Per-row FOV identifier for X_train; when provided,
            the inner-validation split is FOV-grouped (preferred, mirrors
            run.py). If None, falls back to stratified inner-val.

    Returns:
        model: Trained XGBoost model
        metrics: Test set metrics
    """
    params = {
        **best_params,
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
    }

    # Canonical external-val protocol: early-stop on a caller-supplied held-out
    # set (the capped 'val' FOVs of --val_split_file) instead of carving a
    # 10% inner-val. ``X_train``/``y_train`` are already contiguous [0..K-1] over
    # the train classes and ``eval_set_external`` labels are filtered to the same
    # space, so no re-tighten is needed.
    if eval_set_external is not None:
        X_eval, y_eval = eval_set_external
        early_stopping_rounds = max(10, params.get("n_estimators", 100) // 10)
        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        # Per-row DCT sampler weights (sqrt-inverse-frequency, 1000-count floor),
        # the tree analog of the neural baselines' WeightedRandomSampler; None for
        # faithful unweighted XGBoost. Mirrors the legacy path below + run.py.
        sample_weight = (
            compute_sample_weights_dct(y_train, normalize=True)
            if class_balance == "dct"
            else None
        )
        print(f"  Class balancing: {class_balance}")
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_eval, y_eval)],
            verbose=True,
        )
        print(
            f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.6f}"
        )
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_baseline_metrics(
            y_test, y_pred, y_prob, num_classes, hierarchy=hierarchy, ct2idx=ct2idx
        )
        return model, metrics, None

    # Carve an inner-validation set for early stopping. Disjoint from X_test
    # by construction (X_test is never passed in).
    if train_fov_names is not None:
        from sklearn.model_selection import GroupShuffleSplit

        train_fov_array = np.asarray(train_fov_names)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        inner_train_idx, inner_val_idx = next(
            gss.split(X_train, y_train, groups=train_fov_array)
        )
        n_inner_val_fovs = len(np.unique(train_fov_array[inner_val_idx]))
        print(
            f"  Inner-val for early stopping (FOV-grouped): "
            f"{len(inner_val_idx)} samples from {n_inner_val_fovs} FOVs"
        )
    else:
        from sklearn.model_selection import StratifiedShuffleSplit

        # Stratification falls back to a plain shuffle if any class has <2
        # rows, since StratifiedShuffleSplit requires at least one row per
        # class in each partition.
        unique, counts = np.unique(y_train, return_counts=True)
        if (counts >= 2).all():
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            inner_train_idx, inner_val_idx = next(sss.split(X_train, y_train))
        else:
            from sklearn.model_selection import ShuffleSplit

            ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            inner_train_idx, inner_val_idx = next(ss.split(X_train))
        print(
            f"  Inner-val for early stopping (stratified, no FOV names): "
            f"{len(inner_val_idx)} samples"
        )

    X_inner_train = X_train[inner_train_idx]
    y_inner_train = y_train[inner_train_idx]
    X_inner_val = X_train[inner_val_idx]
    y_inner_val = y_train[inner_val_idx]

    # GroupShuffleSplit does not guarantee every class is present in
    # inner-train, and ``XGBClassifier`` rejects targets whose unique
    # values don't equal [0..num_class-1] exactly. Re-tighten the label
    # space to inner-train's class set and drop any inner-val / test
    # rows whose label is absent (test rows dropped here are
    # information loss, not leakage — they're rare classes the model
    # couldn't predict anyway because they never appeared in
    # inner-train). Mirrors the same logic in run.py:178-204. At full
    # scale every class is in inner-train so this is a no-op.
    train_present = np.unique(y_inner_train)
    effective_num_classes = num_classes
    X_test_eff = X_test
    y_test_eff = y_test
    inner_remap = None
    if (
        len(train_present) != num_classes
        or train_present[0] != 0
        or train_present[-1] != len(train_present) - 1
    ):
        inner_remap = {int(orig): i for i, orig in enumerate(train_present)}
        val_mask = np.isin(y_inner_val, train_present)
        test_mask = np.isin(y_test, train_present)
        n_dropped_val = int((~val_mask).sum())
        n_dropped_test = int((~test_mask).sum())
        if n_dropped_val or n_dropped_test:
            print(
                f"  Dropped {n_dropped_val} inner-val + {n_dropped_test} test rows "
                f"with labels absent from inner-train (small-split artifact)."
            )
        X_inner_val = X_inner_val[val_mask]
        y_inner_val = y_inner_val[val_mask]
        X_test_eff = X_test[test_mask]
        y_test_eff = y_test[test_mask]
        remap_fn = np.vectorize(inner_remap.get, otypes=[np.int64])
        y_inner_train = remap_fn(y_inner_train)
        y_inner_val = remap_fn(y_inner_val)
        y_test_eff = remap_fn(y_test_eff)
        effective_num_classes = len(train_present)
        # Update num_class on the params dict so XGBClassifier matches.
        params["num_class"] = effective_num_classes
        if ct2idx is not None:
            ct2idx = {
                name: inner_remap[orig]
                for name, orig in ct2idx.items()
                if orig in inner_remap
            }

    early_stopping_rounds = max(10, params.get("n_estimators", 100) // 10)
    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
    # 'dct' balancing weights each row by the DCT sampler scheme (the tree
    # analog of the neural baselines' WeightedRandomSampler), normalize=True to
    # mean 1 so balancing does not also change effective regularization; 'none'
    # = faithful unweighted XGBoost.
    sample_weight = (
        compute_sample_weights_dct(y_inner_train, normalize=True)
        if class_balance == "dct"
        else None
    )
    model.fit(
        X_inner_train,
        y_inner_train,
        sample_weight=sample_weight,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=True,
    )
    print(
        f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.6f}"
    )

    y_pred = model.predict(X_test_eff)
    y_prob = model.predict_proba(X_test_eff)
    metrics = compute_baseline_metrics(
        y_test_eff,
        y_pred,
        y_prob,
        effective_num_classes,
        hierarchy=hierarchy,
        ct2idx=ct2idx,
    )

    # ``inner_remap`` is None if no class was dropped from inner-train (the
    # common case at full scale); otherwise it maps original compact label →
    # post-GSS contiguous label. Callers need it to write a self-contained
    # remap.json alongside the saved ckpt so out-of-band evaluators can
    # recover the post→orig mapping without replaying the GSS.
    return model, metrics, inner_remap


def _run_canonical_val_tuning(
    *,
    dct_config,
    zarr_dir,
    split_file,
    val_split_file,
    skip_datasets,
    keep_datasets,
    relax_strict,
    X_train_full,
    y_train_full,
    X_test,
    y_test,
    test_dataset_names,
    test_fov_names,
    test_cell_indices,
    train_fov_names_full,
    num_classes,
    n_trials,
    study_name,
    storage,
    device_num,
    max_tuning_samples,
    model_name=None,
    features_cache=None,
    class_balance="dct",
    metric="macro_f1",
):
    """Canonical external-val tuning protocol (mirrors the main DCT model).

    Score each Optuna trial (and early-stop the final model) on the 'val' FOVs
    of --val_split_file capped to 200k cells at seed 42 (mirrors
    dataloader.py:269-273 max_val_samples), with the hierarchical ct_macro_f1
    (LossesAndMetrics.compute, metrics.py:399-419). The reported set is
    --split_file 'val'.

    For speed, each TRIAL fits on a ``max_tuning_samples``-row subsample of the
    full train (mirrors the legacy path's subsample). The contiguous label remap
    is built ONCE from the FULL train's class set, so XGBClassifier's
    ``n_classes`` is fixed = number of full-train classes for every trial and for
    the external val, regardless of which classes a given subsample draws. The
    subsample is class-padded (one representative row per otherwise-missing class)
    to guarantee the subsample's label set equals the full-train [0..K-1]. The
    FINAL best model is fitted on the FULL (un-subsampled) train.
    """
    # Contiguous label space over the TRAIN classes (XGBClassifier requires y in
    # exactly [0..K-1]); selection-val / test rows whose label is absent from
    # train are dropped (smoke-scale no-op at full 1722-FOV scale).
    train_unique = np.sort(np.unique(y_train_full))
    label_to_compact = {int(orig): i for i, orig in enumerate(train_unique)}
    compact_to_label = {i: int(orig) for orig, i in label_to_compact.items()}
    n_classes_compact = len(train_unique)
    compact_ct2idx = {
        name: label_to_compact[idx]
        for name, idx in dct_config.ct2idx.items()
        if idx in label_to_compact
    }
    y_train_full = np.array([label_to_compact[y] for y in y_train_full])

    # Extract + cap the selection-val (the 'val' FOVs of --val_split_file).
    # Reuse the sibling '<features_cache>.valsel.npz' when provided (mirrors
    # xgb/run.py), so the canonical-val tuning skips zarr extraction entirely.
    val_features_cache = (
        str(Path(features_cache).with_suffix(".valsel.npz")) if features_cache else None
    )
    print(f"\nExtracting selection-val features from {val_split_file}...")
    sel_data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=val_split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        missing_value=np.nan,
        cache_path=val_features_cache,
        strict_split=not relax_strict,
    )
    X_sel = sel_data["X_val"]
    y_sel_orig = sel_data["y_val"]
    sel_mask = np.isin(y_sel_orig, train_unique)
    X_sel = X_sel[sel_mask]
    y_sel = np.array([label_to_compact[y] for y in y_sel_orig[sel_mask]])
    # Cap to 200k cells EXACTLY as the main model caps its val set
    # (dataloader.py:269-273: np.random.default_rng(42).choice, no replacement).
    n_val_cells = X_sel.shape[0]
    cap = min(200000, n_val_cells)
    rng = np.random.default_rng(42)
    sel_idx = rng.choice(n_val_cells, size=cap, replace=False)
    X_sel = X_sel[sel_idx]
    y_sel = y_sel[sel_idx]

    # Filter the test (report) rows to train classes + compact-map.
    test_mask = np.isin(y_test, train_unique)
    n_dropped_test = int((~test_mask).sum())
    if n_dropped_test:
        print(
            f"  Dropped {n_dropped_test} test rows with labels absent from "
            f"train (smoke-scale artifact)."
        )
    X_test = X_test[test_mask]
    y_test_orig = y_test[test_mask]  # original ct2idx labels (for CSV)
    y_test = np.array([label_to_compact[y] for y in y_test_orig])
    test_dataset_names = [n for n, k in zip(test_dataset_names, test_mask) if k]
    test_fov_names = [n for n, k in zip(test_fov_names, test_mask) if k]
    test_cell_indices = [c for c, k in zip(test_cell_indices, test_mask) if k]

    n_train_fovs = len(np.unique(np.asarray(train_fov_names_full)))
    n_test_fovs = len(set(test_fov_names))
    print(
        f"Selection protocol: full --split_file train = "
        f"{n_train_fovs} FOVs, {len(X_train_full)} cells (no inner carve); "
        f"selection-val from {val_split_file} capped to {cap} of {n_val_cells} "
        f"cells; test (report) = {n_test_fovs} FOVs, {len(X_test)} cells."
    )

    # Per-trial subsample of the full train for speed (mirrors the legacy
    # path's max_tuning_samples). The label remap was built once over the FULL
    # train, so it is contiguous [0..K-1] across K = n_classes_compact. We
    # class-pad the subsample (one representative row per otherwise-missing
    # class) so its label set equals [0..K-1] every time — that keeps
    # XGBClassifier's n_classes fixed = K across all trials and consistent with
    # the external val, regardless of which classes the random draw happens to
    # contain. The FINAL model below trains on the full un-subsampled train.
    if max_tuning_samples and len(X_train_full) > max_tuning_samples:
        sub_idx = np.random.RandomState(42).choice(
            len(X_train_full), max_tuning_samples, replace=False
        )
        X_tr_sub = X_train_full[sub_idx]
        y_tr_sub = y_train_full[sub_idx]
        present = set(np.unique(y_tr_sub).tolist())
        missing = [c for c in range(n_classes_compact) if c not in present]
        if missing:
            pad_idx = [int(np.where(y_train_full == c)[0][0]) for c in missing]
            X_tr_sub = np.concatenate([X_tr_sub, X_train_full[pad_idx]], axis=0)
            y_tr_sub = np.concatenate([y_tr_sub, y_train_full[pad_idx]], axis=0)
        print(
            f"Per-trial train subsample: {len(X_tr_sub)} rows "
            f"(cap {max_tuning_samples}, +{len(missing)} class-pad rows); "
            f"n_classes fixed = {n_classes_compact} (full-train class count)."
        )
    else:
        X_tr_sub, y_tr_sub = X_train_full, y_train_full
        print(
            f"Per-trial train subsample: {len(X_tr_sub)} rows "
            f"(full train <= cap {max_tuning_samples}); "
            f"n_classes fixed = {n_classes_compact} (full-train class count)."
        )

    # Optuna trials scored by the canonical ct_macro_f1 on the external val.
    print(f"\nStarting hyperparameter tuning ({n_trials} trials)...")
    study, best_params = run_tuning(
        X_tr_sub,
        y_tr_sub,
        X_sel,
        y_sel,
        num_classes=n_classes_compact,
        n_trials=n_trials,
        metric=metric,
        study_name=study_name,
        storage=storage,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
        device=device_num,
        verbose_trial=True,
        class_balance=class_balance,
    )
    print("\nBest trial:")
    print(f"  Value (selection ct_macro_f1): {study.best_trial.value:.4f}")
    print(f"  Params: {best_params}")

    # Final model on full train, early-stopped on the same capped external val.
    print(
        f"\nTraining final model with best parameters on full train "
        f"({len(X_train_full)} cells)..."
    )
    model, test_metrics, _ = train_best_model(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        best_params,
        n_classes_compact,
        device=device_num,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
        eval_set_external=(X_sel, y_sel),
        class_balance=class_balance,
    )

    print("\nFinal Test Results:")
    print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {test_metrics['weighted_accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")

    # Save results (mirrors the legacy save block).
    output_dir = Path("output/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name_safe = study_name or f"xgb_tuning_{timestamp}"

    params_path = output_dir / f"{study_name_safe}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_value": study.best_trial.value,
                "metric": metric,
                "selection": f"canonical external ct_{metric} (capped 200k val)",
                "n_trials": n_trials,
                "test_metrics": {
                    "macro_accuracy": test_metrics["macro_accuracy"],
                    "weighted_accuracy": test_metrics["weighted_accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                },
            },
            f,
            indent=2,
        )
    print(f"\nBest parameters saved to {params_path}")

    model_path = Path(f"models/xgb_tuned_{study_name_safe}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Self-contained post-compact -> original ct mapping (no GSS replay needed:
    # labels are a straight contiguous remap of the train classes).
    idx2ct = {v: k for k, v in dct_config.ct2idx.items()}
    remap_meta = {
        "n_classes": int(model.n_classes_),
        "compact_to_orig_ct_idx": {int(k): int(v) for k, v in compact_to_label.items()},
        "compact_to_ct_name": {
            int(k): idx2ct.get(int(v), "?") for k, v in compact_to_label.items()
        },
        "gss_recipe": (
            "canonical external-val protocol: contiguous remap of train classes; "
            "early stop on capped 'val' FOVs of --val_split_file (no GSS carve)."
        ),
    }
    remap_path = model_path.with_suffix(".remap.json")
    with open(remap_path, "w") as f:
        json.dump(remap_meta, f, indent=2, sort_keys=True)
    print(f"Remap metadata saved to {remap_path}")

    # Prediction CSV on the held-out 129-FOV test (uncapped), in the same schema
    # as the other baselines so the tuned numbers are directly comparable. Map
    # the model's compact probabilities back to ct2idx-sorted columns.
    y_prob_compact = model.predict_proba(X_test)
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob = np.zeros((len(y_test_orig), len(dct_config.ct2idx)), dtype=np.float32)
    for compact_idx, orig_idx in compact_to_label.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]
    pred_name = model_name or study_name_safe
    output_path = Path(f"output/{pred_name}_xgb_tuned_prediction.csv")
    save_baseline_predictions(
        y_test_orig,
        y_prob,
        test_cell_indices,
        test_dataset_names,
        test_fov_names,
        dct_config.ct2idx,
        output_path,
    )

    history_path = output_dir / f"{study_name_safe}_history.csv"
    study.trials_dataframe().to_csv(history_path, index=False)
    print(f"Trial history saved to {history_path}")
    print("\nDone!")


@click.command()
@click.option("--study_name", type=str, default=None, help="Name for the Optuna study")
@click.option("--n_trials", type=int, default=100, help="Number of tuning trials")
@click.option(
    "--metric",
    type=click.Choice(
        ["macro_accuracy", "weighted_accuracy", "macro_f1", "weighted_f1"]
    ),
    default="macro_accuracy",
    help="Metric to optimize. macro_f1 / weighted_f1 read the same key out of "
    "compute_baseline_metrics() — no separate scorer required.",
)
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR),
)
@click.option(
    "--skip_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to skip",
)
@click.option(
    "--keep_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to keep",
)
@click.option(
    "--storage",
    type=str,
    default=None,
    help="Optuna storage URL (e.g., sqlite:///tuning.db) for distributed tuning",
)
@click.option(
    "--split_mode",
    type=click.Choice(["fov", "patch"]),
    default="fov",
    help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)",
)
@click.option(
    "--split_file",
    type=str,
    default=None,
    help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)",
)
@click.option(
    "--val_split_file",
    type=str,
    default=None,
    help="If set, train on the FULL --split_file train and score Optuna trials "
    "(and early-stop the final model) on the 'val' FOVs of THIS file, capped to "
    "200k cells at seed 42 (mirroring dataloader.py:269-273 max_val_samples), "
    "using the canonical hierarchical ct_macro_f1. The reported set stays "
    "--split_file 'val'. When unset, keeps the legacy FOV-grouped inner-val.",
)
@click.option(
    "--max_tuning_samples",
    type=int,
    default=500000,
    help="Max training samples per tuning trial (subsample for speed). Applies "
    "in both the legacy and the --val_split_file canonical protocols; the final "
    "best model always retrains on the full train.",
)
@click.option(
    "--device_num",
    type=str,
    default="cpu",
    help="Device for XGBoost (cpu or cuda:N for GPU acceleration)",
)
@click.option(
    "--features_cache",
    type=str,
    default=None,
    help="Path to a fill-agnostic feature cache (.npz). Reused if its metadata "
    "matches (skips the expensive zarr extraction). The selection-val features "
    "are cached/loaded from a sibling '<features_cache>.valsel.npz'.",
)
@click.option(
    "--class_balance",
    type=click.Choice(["dct", "none"]),
    default="dct",
    help="Training class-balancing for both the Optuna trials and the final "
    "model. 'dct' (default): per-row sample_weight from the DCT sampler "
    "(sqrt-inverse-frequency, 1000-count floor), identical to the main model "
    "and the other baselines. 'none' (faithful XGBoost, ablation): unweighted.",
)
def main(
    study_name: Optional[str],
    n_trials: int,
    metric: str,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    storage: Optional[str],
    split_mode: str,
    split_file: str,
    val_split_file: str,
    max_tuning_samples: int,
    device_num: str,
    features_cache: str,
    class_balance: str,
):
    """Run XGBoost hyperparameter tuning with Optuna."""
    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(f"Optimizing: {metric}")
    print(f"Number of trials: {n_trials}")

    # Convert to lists
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError(
            "--split_file is required. Generate one with: python -m scripts.generate_splits"
        )

    # Relax the strict split-coverage check ONLY for the canonical-val smoke
    # path (--val_split_file with --keep_datasets restricting extraction to a
    # FOV subset). A real run (keep_datasets is None) keeps strict=True.
    _relax_strict = (val_split_file is not None) and (keep_datasets is not None)

    # Extract features directly from zarr (fast path, ~20-50x faster than DataLoader).
    # ``missing_value=np.nan`` so absent markers route through XGBoost's
    # ``missing=NaN`` default-direction logic at every split — without it
    # absent markers would be conflated with real channels whose mean
    # intensity happens to be 0.0.
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        missing_value=np.nan,
        cache_path=features_cache,
        strict_split=not _relax_strict,
    )

    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_test, y_test = data["X_val"], data["y_val"]
    train_fov_names_full = np.asarray(data["train_fov_names"])

    if val_split_file is not None:
        _run_canonical_val_tuning(
            dct_config=dct_config,
            zarr_dir=zarr_dir,
            split_file=split_file,
            val_split_file=val_split_file,
            skip_datasets=skip_datasets,
            keep_datasets=keep_datasets,
            relax_strict=_relax_strict,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            test_dataset_names=data["val_dataset_names"],
            test_fov_names=data["val_fov_names"],
            test_cell_indices=data["val_cell_indices"],
            train_fov_names_full=train_fov_names_full,
            num_classes=num_classes,
            n_trials=n_trials,
            metric=metric,
            study_name=study_name,
            storage=storage,
            device_num=device_num,
            max_tuning_samples=max_tuning_samples,
            features_cache=features_cache,
            class_balance=class_balance,
        )
        return

    print(
        f"Training set: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features"
    )
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Remap labels to contiguous 0-indexed (XGBoost requires this).
    # Train labels first so they get indices [0..n_train_classes-1],
    # then append any test-only labels after.
    train_unique = np.sort(np.unique(y_train_full))
    label_to_compact = {orig: i for i, orig in enumerate(train_unique)}
    next_idx = len(train_unique)
    for label in np.sort(np.unique(y_test)):
        if label not in label_to_compact:
            label_to_compact[label] = next_idx
            next_idx += 1
    n_classes_compact = next_idx
    compact_ct2idx = {
        name: label_to_compact[idx]
        for name, idx in dct_config.ct2idx.items()
        if idx in label_to_compact
    }

    y_train_full = np.array([label_to_compact[y] for y in y_train_full])
    y_test = np.array([label_to_compact[y] for y in y_test])
    print(f"Unique cell types in data: {n_classes_compact} (of {num_classes} total)")

    # Save full training data for final model (before subsampling).
    # FOV names travel alongside so the final fit can carve a FOV-grouped
    # inner-val for early stopping (see ``train_best_model``).
    X_train_all = X_train_full
    y_train_all = y_train_full
    train_fov_names_all = train_fov_names_full

    # Subsample training data for faster tuning trials. FOV names follow
    # the same subsample so the FOV-grouped inner-val split below stays
    # consistent.
    if max_tuning_samples and len(X_train_full) > max_tuning_samples:
        subsample_idx = np.random.RandomState(42).choice(
            len(X_train_full), max_tuning_samples, replace=False
        )
        X_train_full = X_train_full[subsample_idx]
        y_train_full = y_train_full[subsample_idx]
        train_fov_names_subsampled = train_fov_names_full[subsample_idx]
        print(f"Subsampled training data to {max_tuning_samples} samples for tuning")
    else:
        train_fov_names_subsampled = train_fov_names_full

    # Inner train/val split for the Optuna objective. FOV-grouped so that
    # cells from the same FOV cannot appear in both partitions —
    # otherwise hyperparameters that benefit from within-FOV memorisation
    # (e.g. deeper trees) win the tuning, but the FOV-disjoint test set
    # the paper reports is unkind to those choices. Mirrors the split
    # used in ``run.py`` and ``train_best_model``. Test size 0.25 (== 75/25 of
    # subsampled training data) is kept from the previous stratified
    # version for run-to-run continuity.
    from sklearn.model_selection import GroupShuffleSplit

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(
        gss_inner.split(X_train_full, y_train_full, groups=train_fov_names_subsampled)
    )

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    n_inner_val_fovs = len(np.unique(train_fov_names_subsampled[val_idx]))

    # GroupShuffleSplit doesn't guarantee every class is in inner-train
    # (StratifiedShuffleSplit + singleton duplication previously
    # enforced this at the cost of a tiny inner train/val leak).
    # Re-tighten labels to [0..K-1] over inner-train's class set so
    # XGBClassifier accepts the targets; drop inner-val / test rows
    # whose label is absent from inner-train. This mirrors run.py:178-204.
    # ``X_test`` and the original ``n_classes_compact`` are restored before
    # the final ``train_best_model`` call below, which is fitted on the
    # full un-subsampled training data and sees all classes.
    train_present = np.unique(y_train)
    tuning_n_classes = int(n_classes_compact)
    tuning_ct2idx = dict(compact_ct2idx)
    X_val_tuning = X_val
    y_val_tuning = y_val
    if (
        len(train_present) != n_classes_compact
        or train_present[0] != 0
        or train_present[-1] != len(train_present) - 1
    ):
        inner_remap = {int(orig): i for i, orig in enumerate(train_present)}
        val_mask = np.isin(y_val, train_present)
        n_dropped_val = int((~val_mask).sum())
        if n_dropped_val:
            print(
                f"  Dropped {n_dropped_val} inner-val rows whose label is "
                f"absent from FOV-grouped inner-train."
            )
        X_val_tuning = X_val[val_mask]
        y_val_tuning = y_val[val_mask]
        remap_fn = np.vectorize(inner_remap.get, otypes=[np.int64])
        y_train = remap_fn(y_train)
        y_val_tuning = remap_fn(y_val_tuning)
        tuning_ct2idx = {
            name: inner_remap[orig]
            for name, orig in compact_ct2idx.items()
            if orig in inner_remap
        }
        tuning_n_classes = len(train_present)

    print("\nData splits:")
    print(
        f"  Inner-train (FOV-grouped): {len(X_train)} samples, "
        f"{len(np.unique(train_fov_names_subsampled[train_idx]))} FOVs, "
        f"{tuning_n_classes} classes"
    )
    print(
        f"  Inner-val (FOV-grouped):   {len(X_val_tuning)} samples, "
        f"{n_inner_val_fovs} FOVs"
    )
    print(
        f"  Test:                      {len(X_test)} samples (held out, not used in tuning)"
    )

    # Run hyperparameter tuning
    print(f"\nStarting hyperparameter tuning ({n_trials} trials)...")
    study, best_params = run_tuning(
        X_train,
        y_train,
        X_val_tuning,
        y_val_tuning,
        num_classes=tuning_n_classes,
        n_trials=n_trials,
        metric=metric,
        study_name=study_name,
        storage=storage,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=tuning_ct2idx,
        device=device_num,
        class_balance=class_balance,
    )

    print("\nBest trial:")
    print(f"  Value ({metric}): {study.best_trial.value:.4f}")
    print(f"  Params: {best_params}")

    # Train final model on FULL training set (not subsampled) with best params
    print(
        f"\nTraining final model with best parameters on full data ({len(X_train_all)} samples)..."
    )
    X_train_combined = X_train_all
    y_train_combined = y_train_all

    model, test_metrics, inner_remap = train_best_model(
        X_train_combined,
        y_train_combined,
        X_test,
        y_test,
        best_params,
        n_classes_compact,
        device=device_num,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
        train_fov_names=train_fov_names_all,
        class_balance=class_balance,
    )

    print("\nFinal Test Results:")
    print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {test_metrics['weighted_accuracy']:.4f}")

    # Save results
    output_dir = Path("output/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name_safe = study_name or f"xgb_tuning_{timestamp}"

    # Save best parameters
    params_path = output_dir / f"{study_name_safe}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_value": study.best_trial.value,
                "metric": metric,
                "n_trials": n_trials,
                "test_metrics": {
                    "macro_accuracy": test_metrics["macro_accuracy"],
                    "weighted_accuracy": test_metrics["weighted_accuracy"],
                },
            },
            f,
            indent=2,
        )
    print(f"\nBest parameters saved to {params_path}")

    # Save model
    model_path = Path(f"models/xgb_tuned_{study_name_safe}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save the post-GSS → original_ct mapping alongside the ckpt. Mirrors
    # ``run.py``'s same-named sidecar (`<model>.remap.json`). Out-of-band
    # evaluators need this to align ``model.predict()`` outputs (in the
    # contiguous post-GSS space) back to original ``ct2idx`` indices.
    idx2ct = {v: k for k, v in dct_config.ct2idx.items()}
    # Inverse of the post-GSS label_to_compact map: compact idx -> original ct idx.
    compact_to_label = {compact: orig for orig, compact in label_to_compact.items()}
    if inner_remap is not None:
        inv_inner = {v: k for k, v in inner_remap.items()}
        post_to_orig_ct = {
            int(post): int(compact_to_label[int(inv_inner[post])]) for post in inv_inner
        }
    else:
        post_to_orig_ct = {int(c): int(orig) for c, orig in compact_to_label.items()}
    remap_meta = {
        "n_classes": int(model.n_classes_),
        "compact_to_orig_ct_idx": post_to_orig_ct,
        "compact_to_ct_name": {
            int(k): idx2ct.get(int(v), "?") for k, v in post_to_orig_ct.items()
        },
        "gss_recipe": (
            "GroupShuffleSplit(test_size=0.1, random_state=42) on full train, "
            "then optional searchsorted-remap to drop classes absent from "
            "inner-train (see tuning.py:train_best_model)."
        ),
    }
    remap_path = model_path.with_suffix(".remap.json")
    with open(remap_path, "w") as f:
        json.dump(remap_meta, f, indent=2, sort_keys=True)
    print(f"Remap metadata saved to {remap_path}")

    # Save study history
    history_path = output_dir / f"{study_name_safe}_history.csv"
    study.trials_dataframe().to_csv(history_path, index=False)
    print(f"Trial history saved to {history_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
