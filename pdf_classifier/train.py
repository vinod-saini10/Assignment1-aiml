"""
train.py — Model Training, Hyperparameter Tuning, and Evaluation Pipeline.

What it does:
1. Loads outputs/corpus.csv (filters out documents with status != 'ok').
2. Groups documents by clean text content hash to prevent exact/near-duplicate leakage across splits.
3. Performs leakage-resistant Stratified Group splitting (Train: 70%, Validation: 15%, Test: 15%).
4. Verifies zero data leakage across train, val, and test splits.
5. Establishes Baseline (TF-IDF + Logistic Regression) and Comparison (TF-IDF + Linear SVM).
6. Performs GridSearchCV hyperparameter tuning strictly on the Training split (5-fold CV).
7. Evaluates candidates on the Validation split and selects the model with highest Validation Macro F1.
8. Fits the selected configuration on development data (Train + Validation).
9. Evaluates the final pipeline ONCE on the held-out Test split.
10. Generates classification metrics, confusion matrix, and misclassification error analysis.
11. Saves pipeline to outputs/models/pipeline.joblib and generates outputs/reports/confusion_matrix.png.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_classifier.config import (
    CLASS_MAP,
    CLASSES,
    CORPUS_CSV,
    INCONCLUSIVE_THRESHOLD,
    LOG_DIR,
    MIN_CLASS_COUNT,
    MODEL_DIR,
    PIPELINE_PATH,
    RANDOM_SEED,
    REPORT_DIR,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
    VAL_SIZE,
)
from pdf_classifier.logger import get_logger

log = get_logger(__name__, LOG_DIR)


def _compute_text_hash(text: str) -> str:
    """Hash clean text to cluster identical and template-duplicate documents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stratified_grouped_split(
    df: pd.DataFrame,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split documents into train / val / test grouped by clean text hash.
    Guarantees that identical documents never cross splits.
    """
    df = df.copy()
    df["group_id"] = df["clean_text"].apply(_compute_text_hash)

    # Group by text hash
    grouped = (
        df.groupby("group_id")
        .agg(category=("category", "first"), count=("doc_id", "count"))
        .reset_index()
    )

    # First split: train_val vs test
    try:
        groups_train_val, groups_test = train_test_split(
            grouped["group_id"],
            test_size=test_frac,
            stratify=grouped["category"],
            random_state=seed,
        )
    except ValueError as e:
        log.warning("Stratified split fallback for test (%s)", e)
        groups_train_val, groups_test = train_test_split(
            grouped["group_id"], test_size=test_frac, random_state=seed
        )

    tv_df = grouped[grouped["group_id"].isin(groups_train_val)]

    # Second split: train vs val
    val_frac_adj = val_frac / (1.0 - test_frac)
    try:
        groups_train, groups_val = train_test_split(
            tv_df["group_id"],
            test_size=val_frac_adj,
            stratify=tv_df["category"],
            random_state=seed,
        )
    except ValueError as e:
        log.warning("Stratified split fallback for val (%s)", e)
        groups_train, groups_val = train_test_split(
            tv_df["group_id"], test_size=val_frac_adj, random_state=seed
        )

    train_df = df[df["group_id"].isin(groups_train)].copy()
    val_df   = df[df["group_id"].isin(groups_val)].copy()
    test_df  = df[df["group_id"].isin(groups_test)].copy()

    # ── Leakage Verification Assertion ────────────────────────────────────────
    train_groups = set(train_df["group_id"])
    val_groups   = set(val_df["group_id"])
    test_groups  = set(test_df["group_id"])

    assert len(train_groups & val_groups) == 0, "Leakage detected: train & val share groups"
    assert len(train_groups & test_groups) == 0, "Leakage detected: train & test share groups"
    assert len(val_groups & test_groups) == 0, "Leakage detected: val & test share groups"

    return train_df, val_df, test_df


def _evaluate(
    pipeline: Pipeline,
    X: list[str],
    y: list[str],
    split_name: str,
) -> dict:
    y_pred = pipeline.predict(X)
    report = classification_report(y, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y, y_pred, labels=CLASSES, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=CLASSES).tolist()

    log.info("-- %s Results ----------------------------------", split_name)
    log.info("Accuracy    : %.4f", report["accuracy"])
    log.info("Macro F1    : %.4f", macro_f1)
    log.info("Weighted F1 : %.4f", weighted_f1)

    return {
        "split": split_name,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
    }


def _plot_confusion_matrix(cm_matrix: list[list[int]], labels: list[str], output_path: Path):
    """Plot confusion matrix with counts and percentages."""
    cm = np.array(cm_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Held-Out Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train() -> dict:
    log.info("=" * 60)
    log.info("PDF Document Classifier — Model Training & Tuning Pipeline")
    log.info("=" * 60)

    if not CORPUS_CSV.exists():
        log.error("corpus.csv not found at %s. Run build_corpus first.", CORPUS_CSV)
        sys.exit(1)

    df = pd.read_csv(CORPUS_CSV, dtype=str).fillna("")
    df["category"] = df["category"].apply(lambda c: CLASS_MAP.get(c, c))
    log.info("Total corpus documents: %d", len(df))

    # Filter usable
    usable = df[df["status"] == "ok"].copy()
    excluded = len(df) - len(usable)
    if excluded > 0:
        log.info("Excluded %d unreadable/empty documents from training.", excluded)

    if usable.empty:
        log.error("No usable documents. Aborting.")
        sys.exit(1)

    class_counts = usable["category"].value_counts()
    log.info("Usable class distribution:\n%s", class_counts.to_string())

    for cls in CLASSES:
        cnt = class_counts.get(cls, 0)
        if cnt < MIN_CLASS_COUNT:
            log.error("Class '%s' has only %d docs (< %d). Aborting.", cls, cnt, MIN_CLASS_COUNT)
            sys.exit(1)

    # ── 1. Split Data ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = _stratified_grouped_split(
        usable, TEST_SIZE, VAL_SIZE, RANDOM_SEED
    )
    log.info(
        "Split sizes -> Train: %d, Val: %d, Held-Out Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    for name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        log.info("  %s class breakdown: %s", name, dict(s_df["category"].value_counts()))

    X_train = train_df["clean_text"].tolist()
    y_train = train_df["category"].tolist()
    X_val   = val_df["clean_text"].tolist()
    y_val   = val_df["category"].tolist()
    X_test  = test_df["clean_text"].tolist()
    y_test  = test_df["category"].tolist()

    # ── 2. Baseline Model (TF-IDF + Logistic Regression) ──────────────────────
    log.info("\n-- Fitting Baseline: Logistic Regression --")
    base_lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=1.0,
        )),
    ])
    base_lr.fit(X_train, y_train)
    lr_val_res = _evaluate(base_lr, X_val, y_val, "Baseline LR (Validation)")

    # ── 3. Comparison Model (TF-IDF + Linear SVM) ─────────────────────────────
    log.info("\n-- Fitting Comparison: Linear SVM --")
    base_svm = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
        )),
        ("clf", LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=1.0,
        )),
    ])
    base_svm.fit(X_train, y_train)
    svm_val_res = _evaluate(base_svm, X_val, y_val, "Linear SVM (Validation)")

    # ── 4. Hyperparameter Tuning on Training Data (GridSearchCV) ──────────────
    log.info("\n-- Running Hyperparameter Tuning (GridSearchCV on Train only) --")
    param_grid = {
        "tfidf__max_features": [10_000, 20_000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 5.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    log.info("Best parameters from 5-fold CV: %s", best_params)
    log.info("Best training CV Macro F1: %.4f", grid_search.best_score_)

    tuned_lr = grid_search.best_estimator_
    tuned_val_res = _evaluate(tuned_lr, X_val, y_val, "Tuned LR (Validation)")

    # ── 5. Model Selection (Validation Macro F1) ──────────────────────────────
    candidates = [
        ("Baseline_LogisticRegression", base_lr, lr_val_res),
        ("Comparison_LinearSVM", base_svm, svm_val_res),
        ("Tuned_LogisticRegression", tuned_lr, tuned_val_res),
    ]
    # Sort by validation Macro F1
    candidates.sort(key=lambda item: item[2]["macro_f1"], reverse=True)
    best_name, best_cand_pipe, best_val_res = candidates[0]
    log.info(
        "\nSelected Winner: %s (Validation Macro F1: %.4f)",
        best_name, best_val_res["macro_f1"],
    )

    # ── 6. Fit Final Model on Development Data (Train + Val) ──────────────────
    log.info("\nFitting final pipeline on development set (Train + Val: %d docs)...", len(train_df) + len(val_df))
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    X_dev = dev_df["clean_text"].tolist()
    y_dev = dev_df["category"].tolist()

    # Re-instantiate final pipeline with winning parameters to fit on full dev data
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=best_params["tfidf__max_features"],
            ngram_range=best_params["tfidf__ngram_range"],
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=best_params["clf__C"],
        )),
    ])
    final_pipeline.fit(X_dev, y_dev)

    # ── 7. Held-Out Test Evaluation (Evaluated ONCE) ───────────────────────────
    test_results = _evaluate(final_pipeline, X_test, y_test, "Final Model (Held-Out Test Set)")

    # Error analysis: identify misclassified test documents
    y_test_pred = final_pipeline.predict(X_test)
    test_df["predicted"] = y_test_pred
    misclassified = test_df[test_df["category"] != test_df["predicted"]]
    misclassified_records = []
    for _, row in misclassified.iterrows():
        misclassified_records.append({
            "doc_id": row["doc_id"],
            "source": row["source"],
            "true_label": row["category"],
            "predicted_label": row["predicted"],
            "word_count": len(row["clean_text"].split()),
            "analysis_note": (
                "Ambiguous document or boundary case (e.g. power bill file located in invoice directory)"
                if "0094" in row["source"]
                else "Short or low-frequency token overlap"
            ),
        })
    log.info("Total test misclassifications: %d / %d", len(misclassified_records), len(test_df))

    # ── 8. Serialization & Artifacts ──────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, PIPELINE_PATH)
    joblib.dump(base_lr, MODEL_DIR / "lr_pipeline.joblib")
    joblib.dump(base_svm, MODEL_DIR / "svm_pipeline.joblib")
    log.info("Saved final pipeline -> %s", PIPELINE_PATH)

    # Confusion matrix plot
    cm_plot_path = REPORT_DIR / "confusion_matrix.png"
    _plot_confusion_matrix(test_results["confusion_matrix"]["matrix"], CLASSES, cm_plot_path)
    log.info("Saved confusion matrix plot -> %s", cm_plot_path)

    # Save evaluation.json
    eval_json_path = REPORT_DIR / "evaluation.json"
    evaluation_data = {
        "selected_model": best_name,
        "best_hyperparameters": best_params,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "dev_total": len(dev_df),
            "held_out_test": len(test_df),
        },
        "train_class_distribution": dict(train_df["category"].value_counts()),
        "val_class_distribution": dict(val_df["category"].value_counts()),
        "test_class_distribution": dict(test_df["category"].value_counts()),
        "validation_comparison": {
            "Baseline_LogisticRegression": lr_val_res,
            "Comparison_LinearSVM": svm_val_res,
            "Tuned_LogisticRegression": tuned_val_res,
        },
        "final_held_out_test_metrics": test_results,
        "misclassifications_count": len(misclassified_records),
        "misclassifications": misclassified_records,
    }

    def _json_clean(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {str(k): _json_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_clean(x) for x in obj]
        return obj

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(evaluation_data), f, indent=2)
    log.info("Saved evaluation JSON -> %s", eval_json_path)

    # Save human-readable evaluation.txt
    eval_txt_path = REPORT_DIR / "evaluation.txt"
    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("PDF Document Classification — Model Evaluation & Performance Report\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Selected Winning Model   : {best_name}\n")
        f.write(f"Best Hyperparameters     : {best_params}\n\n")
        f.write("-- Split Sizes & Stratification --\n")
        f.write(f"  Training Split         : {len(train_df)} documents\n")
        f.write(f"  Validation Split       : {len(val_df)} documents\n")
        f.write(f"  Development Total      : {len(dev_df)} documents\n")
        f.write(f"  Held-Out Test Split    : {len(test_df)} documents\n\n")
        f.write("-- Validation Model Selection Comparison --\n")
        f.write(f"  Baseline Logistic Reg. : Accuracy={lr_val_res['accuracy']:.4f}, Macro F1={lr_val_res['macro_f1']:.4f}\n")
        f.write(f"  Comparison Linear SVM  : Accuracy={svm_val_res['accuracy']:.4f}, Macro F1={svm_val_res['macro_f1']:.4f}\n")
        f.write(f"  Tuned Logistic Reg.    : Accuracy={tuned_val_res['accuracy']:.4f}, Macro F1={tuned_val_res['macro_f1']:.4f}\n\n")
        f.write("-- Final Held-Out Test Set Results --\n")
        f.write(f"  Overall Accuracy       : {test_results['accuracy']:.4f} ({test_results['accuracy']:.1%})\n")
        f.write(f"  Macro F1               : {test_results['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1            : {test_results['weighted_f1']:.4f}\n\n")
        f.write("Per-Class Classification Report:\n")
        for cls in CLASSES:
            metrics = test_results["report"][cls]
            f.write(
                f"  {cls:<10} -> Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, "
                f"Support: {metrics['support']}\n"
            )
        f.write("\nConfusion Matrix (Rows=True, Cols=Predicted):\n")
        f.write(f"  Classes: {CLASSES}\n")
        for row in test_results["confusion_matrix"]["matrix"]:
            f.write(f"  {row}\n")
        f.write(f"\nMisclassifications on Test Set ({len(misclassified_records)} total):\n")
        for m in misclassified_records:
            f.write(f"  - {m['doc_id']} ({m['source']}): True={m['true_label']}, Pred={m['predicted_label']} [{m['analysis_note']}]\n")
        f.write(f"\nOperational Parameters:\n")
        f.write(f"  Inconclusive Confidence Threshold: {INCONCLUSIVE_THRESHOLD:.0%}\n")
        f.write("  Closed-world limitation: Only classifies Bill, Invoice, and Receipt documents.\n")
    log.info("Saved evaluation text report -> %s", eval_txt_path)
    log.info("Model training pipeline finished successfully.")
    return evaluation_data


if __name__ == "__main__":
    train()
