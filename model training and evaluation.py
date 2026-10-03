#!/usr/bin/env python
"""Train and evaluate bladder-cancer classifiers with leakage-safe preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "preprocessed_data.xlsx"
RANDOM_STATE = 42
NUM_FEATURES = 15


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_excel(path)
    required = {"genes", "LABEL"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    feature_columns = [c for c in df.columns if c not in required]
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if df[feature_columns].isna().any().any():
        raise ValueError("Feature data contains missing or non-numeric values.")
    labels = set(df["LABEL"].dropna().unique())
    if labels != {0, 1}:
        raise ValueError("LABEL must contain both binary classes 0 and 1.")
    return df


def classification_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return specificity, npv, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_models():
    return {
        "SVM": SVC(C=0.05, kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(C=0.1, penalty="l1", solver="liblinear", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.02, max_depth=2, subsample=0.7, colsample_bytree=0.7, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        "Regularized MLP (DL)": MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=10.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
    }


def evaluate_and_visualize() -> None:
    df = load_data()
    train_pool, holdout = train_test_split(df, test_size=0.2, stratify=df["LABEL"], random_state=RANDOM_STATE)
    feature_columns = [c for c in df.columns if c not in {"genes", "LABEL"}]
    X_train, y_train = train_pool[feature_columns], train_pool["LABEL"]
    X_test, y_test = holdout[feature_columns], holdout["LABEL"]

    selector = SelectKBest(score_func=f_classif, k=min(NUM_FEATURES, len(feature_columns)))
    scaler = StandardScaler()
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]

    # Resampling happens only after the holdout split and training-only fits.
    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)
    models = build_models()
    thresholds = {name: 0.10 for name in models}
    thresholds.update({"SVM": 0.15, "Logistic Regression": 0.05})
    results = {}
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (probabilities >= thresholds[name]).astype(int)
        specificity, npv, matrix = classification_stats(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        results[name] = {
            "threshold": thresholds[name],
            "sensitivity": float(recall_score(y_test, predictions, zero_division=0)),
            "specificity": float(specificity),
            "npv": float(npv),
            "f2_score": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
            "roc_auc": float(auc),
            "confusion_matrix": matrix,
        }
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Isolated Holdout Set")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "roc_curves_comparison.png", dpi=250)
    plt.close()

    metrics = ["sensitivity", "specificity", "npv", "f2_score"]
    names = list(models)
    values = np.array([[results[name][metric] for metric in metrics] for name in names])
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        ax.bar(x + i * width - 0.4 + width / 2, values[i], width, label=name)
    ax.set_xticks(x, ["Sensitivity", "Specificity", "NPV", "F2-score"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Holdout Metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "metrics_comparison_bar.png", dpi=250)
    plt.close(fig)

    bundle = {"model": models["XGBoost"], "selector": selector, "scaler": scaler, "input_feature_names": feature_columns, "feature_names": selected_features, "threshold": thresholds["XGBoost"], "model_name": "XGBoost"}
    joblib.dump(bundle, ROOT / "model_bundle.pkl")
    joblib.dump(bundle["model"], ROOT / "best_model.pkl")
    joblib.dump(bundle["model"], ROOT / "XGBoost_best_model.pkl")
    joblib.dump(scaler, ROOT / "scaler.pkl")
    joblib.dump(selected_features, ROOT / "feature_names.pkl")
    with (ROOT / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"best_model_name": "XGBoost", "num_features": len(selected_features), "results": results}, file, indent=2)
    holdout.drop(columns=["LABEL"]).to_csv(ROOT / "holdout_test_patients.csv", index=False)
    holdout[["LABEL"]].to_csv(ROOT / "holdout_test_labels.csv", index=False)
    print(json.dumps(results, indent=2))
    print(f"Saved model bundle and evaluation outputs to {ROOT}")


if __name__ == "__main__":
    evaluate_and_visualize()
