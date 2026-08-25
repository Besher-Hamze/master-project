"""
Comprehensive NIDS Training Pipeline
Addresses IEEE Access reviewer requirements:
  - Multi-dataset (NSL-KDD, UNSW-NB15, CIC-IDS-2017)
  - SMOTE-ENN for class imbalance (Recall target >= 0.85)
  - Baselines: XGBoost, LightGBM, CatBoost, Random Forest, LSTM
  - Ablation study
  - SHAP explainability
  - Macro F1, Precision, Recall metrics

Usage:
  python train_pipeline.py --dataset unsw-nb15
  python train_pipeline.py --dataset all --smote
  python train_pipeline.py --dataset nsl-kdd --ablation
  python train_pipeline.py --dataset unsw-nb15 --shap
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from datasets.loaders import load_dataset, list_datasets

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


# ─────────────────────────────────────────────
#  Model factories
# ─────────────────────────────────────────────

def make_xgboost():
    return xgb.XGBClassifier(
        max_depth=10, learning_rate=0.1, n_estimators=200,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        scale_pos_weight=3.0,
    )


def make_lightgbm():
    if not HAS_LGB:
        return None
    return lgb.LGBMClassifier(
        max_depth=10, learning_rate=0.1, n_estimators=200,
        random_state=42, n_jobs=-1, verbose=-1,
        class_weight="balanced",
    )


def make_catboost():
    if not HAS_CAT:
        return None
    return CatBoostClassifier(
        depth=10, learning_rate=0.1, iterations=200,
        random_state=42, verbose=0, auto_class_weights="Balanced",
    )


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42,
        n_jobs=-1, class_weight="balanced",
    )


def build_lstm(input_dim):
    """Simple LSTM for sequence-based comparison."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
#  Resampling
# ─────────────────────────────────────────────

def apply_smote_enn(X_train, y_train, method="smote-enn", max_samples=100_000):
    if not HAS_IMBLEARN:
        print("  WARNING: imbalanced-learn not installed, skipping resampling")
        return X_train, y_train

    if len(y_train) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_train), max_samples, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    attack_ratio = y_sub.mean()
    print(f"  Before resampling: {len(y_sub):,} samples, attack ratio={attack_ratio:.3f}")

    k = min(5, int((y_sub == 1).sum()) - 1)
    if k < 1:
        print("  Too few attack samples for SMOTE, skipping")
        return X_train, y_train

    if method == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=k)
    else:
        sampler = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k))

    X_res, y_res = sampler.fit_resample(X_sub, y_sub)
    print(f"  After {method}: {len(y_res):,} samples, attack ratio={y_res.mean():.3f}")
    return X_res, y_res


# ─────────────────────────────────────────────
#  Threshold optimization (boost Recall)
# ─────────────────────────────────────────────

def optimize_threshold(model, X_val, y_val, target_recall=0.85, is_keras=False):
    """Find decision threshold that achieves target recall."""
    if is_keras:
        X_3d = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        proba = model.predict(X_3d, verbose=0).flatten()
    else:
        proba = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_recall = 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (proba >= thresh).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        if rec >= target_recall and rec >= best_recall:
            best_recall = rec
            best_thresh = thresh

    if best_recall < target_recall:
        for thresh in np.arange(0.05, 0.95, 0.01):
            preds = (proba >= thresh).astype(int)
            rec = recall_score(y_val, preds, zero_division=0)
            if rec > best_recall:
                best_recall = rec
                best_thresh = thresh

    return best_thresh, best_recall


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, is_keras=False, threshold=0.5):
    t0 = time.perf_counter()

    if is_keras:
        X_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_proba = model.predict(X_3d, verbose=0).flatten()
    else:
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

    y_pred = (y_proba >= threshold).astype(int)
    ms = (time.perf_counter() - t0) * 1000

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "inference_ms": round(ms, 2),
        "threshold": round(threshold, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def print_metrics_table(all_metrics):
    header = f"{'Model':<18} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'MacroF1':>8} {'AUC':>7} {'ms':>8}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m['model']:<18} "
            f"{m['accuracy']*100:>6.1f}% "
            f"{m['precision']*100:>6.1f}% "
            f"{m['recall']*100:>6.1f}% "
            f"{m['f1']*100:>6.1f}% "
            f"{m['macro_f1']*100:>7.1f}% "
            f"{m['roc_auc']*100:>6.1f}% "
            f"{m['inference_ms']:>7.1f}"
        )


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test, use_smote=False):
    if use_smote:
        X_train, y_train = apply_smote_enn(X_train, y_train)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    models = [
        ("XGBoost", make_xgboost(), False),
        ("Random Forest", make_random_forest(), False),
    ]

    lgbm = make_lightgbm()
    if lgbm:
        models.append(("LightGBM", lgbm, False))

    cat = make_catboost()
    if cat:
        models.append(("CatBoost", cat, False))

    results = []
    trained = {}
    thresholds = {}

    for name, model, is_keras in models:
        print(f"\n  Training {name} ...")
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        print(f"  Trained in {train_s:.1f}s")

        thresh, val_recall = optimize_threshold(model, X_val, y_val, is_keras=is_keras)
        print(f"  Optimal threshold: {thresh:.2f} (val recall: {val_recall*100:.1f}%)")
        thresholds[name] = thresh

        m = evaluate(name, model, X_test, y_test, is_keras, threshold=thresh)
        m["train_time_s"] = round(train_s, 1)
        m["val_recall"] = round(val_recall, 4)
        results.append(m)
        trained[name] = model

    # LSTM (separate due to different input shape)
    print("\n  Training LSTM ...")
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        lstm = build_lstm(X_tr.shape[1])
        X_3d_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_3d_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        t0 = time.perf_counter()
        lstm.fit(
            X_3d_tr, y_tr,
            epochs=10, batch_size=256, validation_split=0.1,
            verbose=0,
        )
        train_s = time.perf_counter() - t0
        thresh, val_recall = optimize_threshold(lstm, X_val, y_val, is_keras=True)
        print(f"  Optimal threshold: {thresh:.2f} (val recall: {val_recall*100:.1f}%)")
        m = evaluate("LSTM", lstm, X_test, y_test, is_keras=True, threshold=thresh)
        m["train_time_s"] = round(train_s, 1)
        m["val_recall"] = round(val_recall, 4)
        results.append(m)
        trained["LSTM"] = lstm
        thresholds["LSTM"] = thresh
        print(f"  Trained in {train_s:.1f}s")
    except Exception as exc:
        print(f"  LSTM skipped: {exc}")

    return results, trained, thresholds


# ─────────────────────────────────────────────
#  Ablation study
# ─────────────────────────────────────────────

def run_ablation(dataset_name, use_smote=True):
    print("\n" + "=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)

    ablation_results = []

    # Full pipeline (baseline)
    print("\n[1/4] Full pipeline (scaling + SMOTE-ENN + tuned XGBoost)")
    X_tr, y_tr, X_te, y_te, _ = load_dataset(dataset_name, scale=True)
    if use_smote:
        X_tr, y_tr = apply_smote_enn(X_tr, y_tr)
    model = make_xgboost()
    model.fit(X_tr, y_tr)
    m = evaluate("full_pipeline", model, X_te, y_te)
    ablation_results.append(m)

    # No scaling
    print("\n[2/4] Without feature scaling")
    X_tr, y_tr, X_te, y_te, _ = load_dataset(dataset_name, scale=False)
    if use_smote:
        X_tr, y_tr = apply_smote_enn(X_tr, y_tr)
    model = make_xgboost()
    model.fit(X_tr, y_tr)
    m = evaluate("no_scaling", model, X_te, y_te)
    ablation_results.append(m)

    # No SMOTE
    print("\n[3/4] Without SMOTE-ENN resampling")
    X_tr, y_tr, X_te, y_te, _ = load_dataset(dataset_name, scale=True)
    model = make_xgboost()
    model.fit(X_tr, y_tr)
    m = evaluate("no_smote", model, X_te, y_te)
    ablation_results.append(m)

    # Top-10 features only
    print("\n[4/4] Top-10 features only")
    X_tr, y_tr, X_te, y_te, feat_names = load_dataset(dataset_name, scale=True)
    if use_smote:
        X_tr, y_tr = apply_smote_enn(X_tr, y_tr)
    full_model = make_xgboost()
    full_model.fit(X_tr, y_tr)
    importances = full_model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    X_tr_top = X_tr[:, top_idx]
    X_te_top = X_te[:, top_idx]
    model = make_xgboost()
    model.fit(X_tr_top, y_tr)
    m = evaluate("top10_features", model, X_te_top, y_te)
    m["selected_features"] = [feat_names[i] for i in top_idx]
    ablation_results.append(m)

    print("\n  Ablation Results:")
    print_metrics_table(ablation_results)

    out = RESULTS_DIR / f"ablation_{dataset_name}.json"
    with open(out, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n  Saved: {out}")

    _plot_ablation(ablation_results, dataset_name)
    return ablation_results


def _plot_ablation(results, dataset_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    width = 0.2
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [r[metric] for r in results]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Ablation Study — {dataset_name}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / f"ablation_{dataset_name}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Plot saved: {path}")


# ─────────────────────────────────────────────
#  SHAP analysis
# ─────────────────────────────────────────────

def run_shap(model, X_test, feature_names, dataset_name, max_samples=500):
    if not HAS_SHAP:
        print("  SHAP not installed. Run: pip install shap")
        return

    print("\n" + "=" * 60)
    print("  SHAP ANALYSIS")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = min(max_samples, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, max_display=20,
    )
    path = RESULTS_DIR / f"shap_summary_{dataset_name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot: {path}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar", show=False, max_display=15,
    )
    path2 = RESULTS_DIR / f"shap_importance_{dataset_name}.png"
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  SHAP importance plot: {path2}")


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_comparison(all_metrics, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [m["model"] for m in all_metrics]
    metric_keys = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Macro F1"]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(names)
    for i, m in enumerate(all_metrics):
        vals = [m[k] for k in metric_keys]
        axes[0].bar(x + i * width, vals, width, label=m["model"])

    axes[0].set_xticks(x + width * (len(names) - 1) / 2)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Model Comparison — {dataset_name}", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    recalls = [m["recall"] for m in all_metrics]
    colors = ["#2ecc71" if r >= 0.85 else "#e74c3c" for r in recalls]
    axes[1].barh(names, recalls, color=colors)
    axes[1].axvline(x=0.85, color="red", linestyle="--", label="Target Recall=0.85")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title("Recall Comparison (target >= 0.85)", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = RESULTS_DIR / f"comparison_{dataset_name}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Comparison plot: {path}")


def save_best_model(trained, all_metrics, dataset_name, thresholds=None):
    best = max(all_metrics, key=lambda m: m["recall"])
    best_name = best["model"]
    model = trained[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if best_name == "LSTM":
        model.save(str(MODEL_DIR / "lstm_model.keras"))
    elif best_name == "XGBoost":
        model.save_model(str(MODEL_DIR / "xgboost_model.json"))
        with open(MODEL_DIR / "xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        with open(MODEL_DIR / f"{best_name.lower().replace(' ', '_')}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    meta = {"dataset": dataset_name, "best_model": best_name, "metrics": best, "thresholds": thresholds}
    with open(MODEL_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Best model saved: {best_name} (Recall={best['recall']*100:.1f}%)")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run_dataset(dataset_name, use_smote=True, do_ablation=False, do_shap=False):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test, feature_names = load_dataset(dataset_name)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Attack ratio — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    all_metrics, trained, thresholds = train_all_models(X_train, y_train, X_test, y_test, use_smote)

    print(f"\n  Results for {dataset_name}:")
    print_metrics_table(all_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"results_{dataset_name}.json"
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {out_json}")

    plot_comparison(all_metrics, dataset_name)
    save_best_model(trained, all_metrics, dataset_name, thresholds)

    if do_ablation:
        run_ablation(dataset_name, use_smote)

    if do_shap and "XGBoost" in trained:
        run_shap(trained["XGBoost"], X_test, feature_names, dataset_name)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="NIDS Training Pipeline (IEEE Access)")
    parser.add_argument("--dataset", default="unsw-nb15",
                        help="Dataset: nsl-kdd, unsw-nb15, cic-ids-2017, or all")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE-ENN (enabled by default)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP plots")
    args = parser.parse_args()

    use_smote = not args.no_smote  # SMOTE-ENN on by default

    print(f"\nNIDS Training Pipeline — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  SMOTE-ENN: {'ON' if use_smote else 'OFF'}")
    print(f"  Ablation:  {'ON' if args.ablation else 'OFF'}")
    print(f"  SHAP:      {'ON' if args.shap else 'OFF'}")

    datasets = list_datasets() if args.dataset == "all" else [args.dataset]
    all_results = {}

    for ds in datasets:
        try:
            all_results[ds] = run_dataset(ds, use_smote, args.ablation, args.shap)
        except FileNotFoundError as exc:
            print(f"\n  SKIPPED {ds}: {exc}")
            print(f"  Run: python -m datasets.download --dataset {ds}")

    if all_results:
        summary_path = RESULTS_DIR / "summary_all_datasets.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Full summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
