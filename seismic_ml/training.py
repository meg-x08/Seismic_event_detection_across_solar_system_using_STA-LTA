"""Train, cross-validate and compare all classifiers (report chapter 5).

Evaluation protocol (report section 5.1.6):
* 70 % / 30 % train/test split - made **per recording file** (GroupShuffleSplit)
  so that overlapping windows of the same file never end up on both sides.
* 5-fold cross-validation on the training part (StratifiedGroupKFold).
* Accuracy, precision, recall, F1 and ROC-AUC for every model, plus the
  classical "STA/LTA only" detector evaluated on exactly the same windows.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from . import plotting
from .config import get_config
from .features import FEATURE_NAMES
from .models import get_models

BASELINE = plotting.BASELINE_NAME


# ----------------------------------------------------------------------------- splitting
def group_train_test_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int):
    """Split by recording file; fall back to a stratified row split for tiny datasets."""
    idx = np.arange(len(y))
    if len(np.unique(groups)) >= 4:
        for attempt in range(100):
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
            tr, te = next(gss.split(idx, y, groups))
            if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
                return tr, te, "by recording file (GroupShuffleSplit)"
    warnings.warn("Too few recording files for a group-wise split - using a stratified row split. "
                  "Scores may be optimistic because neighbouring windows can leak across the split.")
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return tr, te, "stratified by row (fallback)"


def make_cv(y: np.ndarray, groups: np.ndarray, folds: int, seed: int):
    if len(np.unique(groups)) >= folds:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed), groups
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), None


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_score) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
    }
    return {k: float(v) for k, v in out.items()}


def baseline_predict(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Classical detector: 'event' if the STA/LTA ratio crossed the dynamic threshold."""
    score = X["sta_lta_peak_over_threshold"].to_numpy(dtype=float)
    return (score >= 1.0).astype(int), score


def positive_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def cross_validate_models(models: dict, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                          folds: int, seed: int, verbose: bool = True) -> pd.DataFrame:
    cv, cv_groups = make_cv(y, groups, folds, seed)
    per_fold: dict[str, list[dict]] = {name: [] for name in list(models) + [BASELINE]}
    for fold, (tr, va) in enumerate(cv.split(X, y, cv_groups), 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        for name, est in models.items():
            m = clone(est).fit(Xtr, y[tr])
            score = positive_scores(m, Xva)
            per_fold[name].append(compute_metrics(y[va], (score >= 0.5).astype(int), score))
        pred, score = baseline_predict(Xva)
        per_fold[BASELINE].append(compute_metrics(y[va], pred, score))
        if verbose:
            print(f"    CV fold {fold}/{folds} done")

    rows = []
    for name, folds_metrics in per_fold.items():
        if not folds_metrics:
            continue
        df = pd.DataFrame(folds_metrics)
        row = {"model": name, "folds": len(df)}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main entry
def train_and_evaluate(df: pd.DataFrame, out_dir: str, model_names: list[str] | None = None,
                       test_size: float = 0.3, cv_folds: int = 5, seed: int = 42,
                       configs: dict | None = None, verbose: bool = True) -> dict:
    """Run the full comparison and write every result into ``out_dir``.

    ``configs`` maps planet -> PlanetConfig dict used to build the features; it is
    stored inside the saved model so detection uses exactly the same settings.
    """
    results_dir = os.path.join(out_dir, "results")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    df = df.dropna(subset=FEATURE_NAMES + ["label"]).reset_index(drop=True)
    X = df[FEATURE_NAMES].astype(float)
    y = df["label"].astype(int).to_numpy()
    groups = df["group"].astype(str).to_numpy() if "group" in df else np.arange(len(df)).astype(str)
    if len(np.unique(y)) < 2:
        raise ValueError("The feature table contains only one class - cannot train a classifier.")

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if verbose:
        print(f"Dataset: {len(y)} windows ({n_pos} events / {n_neg} noise) from "
              f"{len(np.unique(groups))} recordings, {len(FEATURE_NAMES)} features")

    tr, te, split_kind = group_train_test_split(y, groups, test_size, seed)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    if verbose:
        print(f"Train/test split {split_kind}: {len(tr)} train / {len(te)} test windows")

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    models = get_models(random_state=seed, pos_weight=pos_weight, include=model_names)

    # ---- k-fold cross-validation on the training part
    if verbose:
        print(f"Running {cv_folds}-fold cross-validation on the training set ...")
    cv_df = cross_validate_models(models, Xtr, ytr, groups[tr], cv_folds, seed, verbose)
    cv_df.to_csv(os.path.join(results_dir, "cross_validation.csv"), index=False)

    # ---- fit on train, evaluate on the held-out test set
    test_rows, curves, matrices, reports, fitted = [], {}, {}, [], {}
    for name, est in models.items():
        t0 = time.perf_counter()
        model = clone(est).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        score = positive_scores(model, Xte)
        pred = (score >= 0.5).astype(int)
        metrics = compute_metrics(yte, pred, score)
        test_rows.append({"model": name, **metrics, "fit_time_s": fit_s})
        fpr, tpr, _ = roc_curve(yte, score)
        curves[name] = (fpr, tpr, metrics["roc_auc"])
        matrices[name] = confusion_matrix(yte, pred, labels=[0, 1])
        reports.append(f"=== {name} ===\n" + classification_report(
            yte, pred, labels=[0, 1], target_names=["noise", "event"], digits=4, zero_division=0))
        fitted[name] = model
        if verbose:
            print(f"  {name:<16s} acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
                  f"rec={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")

    pred, score = baseline_predict(Xte)
    base_metrics = compute_metrics(yte, pred, score)
    test_rows.append({"model": BASELINE, **base_metrics, "fit_time_s": 0.0})
    fpr, tpr, _ = roc_curve(yte, score)
    curves[BASELINE] = (fpr, tpr, base_metrics["roc_auc"])
    matrices[BASELINE] = confusion_matrix(yte, pred, labels=[0, 1])
    reports.append(f"=== {BASELINE} ===\n" + classification_report(
        yte, pred, labels=[0, 1], target_names=["noise", "event"], digits=4, zero_division=0))
    if verbose:
        print(f"  {BASELINE:<16s} acc={base_metrics['accuracy']:.4f}  "
              f"prec={base_metrics['precision']:.4f}  rec={base_metrics['recall']:.4f}  "
              f"f1={base_metrics['f1']:.4f}  auc={base_metrics['roc_auc']:.4f}")

    test_df = pd.DataFrame(test_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    test_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    with open(os.path.join(results_dir, "classification_reports.txt"), "w") as fh:
        fh.write("\n\n".join(reports))

    # ---- choose the best model by mean cross-validated F1 (never by the test set)
    ml_cv = cv_df[cv_df["model"] != BASELINE].sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
    best_name = ml_cv.iloc[0]["model"] if len(ml_cv) else test_df.iloc[0]["model"]

    # ---- figures
    plotting.plot_roc_curves(curves, os.path.join(results_dir, "roc_curves.png"))
    plotting.plot_metric_comparison(test_df, os.path.join(results_dir, "model_comparison.png"))
    plotting.plot_confusion_matrices(matrices, os.path.join(results_dir, "confusion_matrices.png"))
    importance_model = next((n for n in [best_name, "XGBoost", "Random Forest", "LightGBM"]
                             if n in fitted and hasattr(fitted[n], "feature_importances_")), None)
    if importance_model:
        imp = np.asarray(fitted[importance_model].feature_importances_, dtype=float)
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        plotting.plot_feature_importance(FEATURE_NAMES, imp, importance_model,
                                         os.path.join(results_dir, "feature_importance.png"))
        pd.DataFrame({"feature": FEATURE_NAMES, "importance": imp}).sort_values(
            "importance", ascending=False).to_csv(
            os.path.join(results_dir, "feature_importance.csv"), index=False)
    if "PCA + LogReg" in fitted:
        pca = fitted["PCA + LogReg"].named_steps["pca"]
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        full = PCA().fit(StandardScaler().fit_transform(Xtr))
        plotting.plot_pca_variance(full.explained_variance_ratio_,
                                   os.path.join(results_dir, "pca_explained_variance.png"))
        if verbose:
            print(f"  PCA kept {pca.n_components_} of {len(FEATURE_NAMES)} components (95 % variance)")

    # ---- per-planet breakdown of the best model on the test set
    per_planet = []
    if "planet" in df.columns:
        best_score = positive_scores(fitted[best_name], Xte)
        planets_te = df["planet"].to_numpy()[te]
        for planet in np.unique(planets_te):
            mask = planets_te == planet
            if len(np.unique(yte[mask])) == 2:
                per_planet.append({"planet": planet, "model": best_name, "n_test": int(mask.sum()),
                                   **compute_metrics(yte[mask], (best_score[mask] >= 0.5).astype(int),
                                                     best_score[mask])})
    if per_planet:
        pd.DataFrame(per_planet).to_csv(os.path.join(results_dir, "per_planet_metrics.csv"), index=False)

    # ---- refit the best model on ALL labelled data and save it for detect_events.py
    final_model = clone(models[best_name]).fit(X, y)
    planets = sorted(df["planet"].unique()) if "planet" in df else []
    bundle = {
        "model": final_model,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "planets": planets,
        "configs": {p: (configs or {}).get(p) or get_config(p).to_dict() for p in planets},
        "test_metrics": test_df[test_df["model"] == best_name].iloc[0].to_dict(),
        "cv_metrics": ml_cv.iloc[0].to_dict() if len(ml_cv) else {},
        "n_training_windows": int(len(y)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(bundle, model_path)

    summary = _write_summary(results_dir, test_df, cv_df, best_name, split_kind, len(y), n_pos, n_neg,
                             len(np.unique(groups)), per_planet)
    with open(os.path.join(results_dir, "run_info.json"), "w") as fh:
        json.dump({"best_model": best_name, "split": split_kind, "test_size": test_size,
                   "cv_folds": cv_folds, "seed": seed, "n_windows": int(len(y)),
                   "n_events": n_pos, "n_noise": n_neg, "planets": planets,
                   "features": FEATURE_NAMES}, fh, indent=2)
    if verbose:
        print(f"\nBest model (highest mean CV F1): {best_name}")
        print(f"Saved model  -> {model_path}")
        print(f"Saved results -> {results_dir}")
    return {"best_model": best_name, "test": test_df, "cv": cv_df, "model_path": model_path,
            "summary": summary}


def _fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}%"


def _write_summary(results_dir, test_df, cv_df, best_name, split_kind, n, n_pos, n_neg, n_files,
                   per_planet) -> str:
    lines = [
        "# Model comparison",
        "",
        f"* Windows: **{n}** ({n_pos} seismic events / {n_neg} noise) from **{n_files}** recordings",
        f"* Split: 70/30 {split_kind}; 5-fold cross-validation on the training part",
        f"* Best model (selected by mean cross-validated F1): **{best_name}**",
        "",
        "## Held-out test set",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in test_df.iterrows():
        lines.append(f"| {r['model']} | {_fmt(r['accuracy'])} | {_fmt(r['precision'])} | "
                     f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {r['roc_auc']:.3f} |")
    lines += ["", "## Cross-validation (mean +/- std)", "",
              "| Model | Accuracy | F1-score | ROC-AUC |", "|---|---|---|---|"]
    for _, r in cv_df.sort_values("f1_mean", ascending=False).iterrows():
        lines.append(f"| {r['model']} | {_fmt(r['accuracy_mean'])} +/- {100 * r['accuracy_std']:.1f} | "
                     f"{_fmt(r['f1_mean'])} +/- {100 * r['f1_std']:.1f} | "
                     f"{r['roc_auc_mean']:.3f} +/- {r['roc_auc_std']:.3f} |")
    if per_planet:
        lines += ["", f"## Per-planet test metrics ({best_name})", "",
                  "| Planet | Test windows | Accuracy | Precision | Recall | F1-score |",
                  "|---|---|---|---|---|---|"]
        for r in per_planet:
            lines.append(f"| {r['planet']} | {r['n_test']} | {_fmt(r['accuracy'])} | "
                         f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results_dir, "summary.md"), "w") as fh:
        fh.write(text)
    return text
