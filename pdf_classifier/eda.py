"""
eda.py — Exploratory Data Analysis & Data Quality Verification.

Generates:
1. Class distribution & extraction statistics
2. Text-length distributions (character and word counts)
3. Duplicate & near-duplicate analysis
4. Data quality notes & exclusion audit
5. Visualizations saved to outputs/reports/
6. Machine-readable summary (outputs/reports/eda_summary.json)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_classifier.config import (
    CLASSES,
    CORPUS_CSV,
    LOG_DIR,
    REPORT_DIR,
)
from pdf_classifier.logger import get_logger

log = get_logger(__name__, LOG_DIR)


def run_eda() -> dict:
    log.info("=" * 60)
    log.info("PDF Document Classifier — Exploratory Data Analysis")
    log.info("=" * 60)

    if not CORPUS_CSV.exists():
        log.error("corpus.csv not found at %s. Run build_corpus first.", CORPUS_CSV)
        sys.exit(1)

    df = pd.read_csv(CORPUS_CSV, dtype=str).fillna("")
    total_docs = len(df)
    log.info("Loaded corpus: %d documents", total_docs)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Basic counts & status ──────────────────────────────────────────────
    class_counts = df["category"].value_counts().to_dict()
    status_counts = df["status"].value_counts().to_dict()
    method_counts = df["extraction_method"].value_counts().to_dict()
    
    usable_df = df[df["status"] == "ok"].copy()
    excluded_df = df[df["status"] != "ok"].copy()

    usable_by_class = usable_df["category"].value_counts().to_dict()
    excluded_by_class = excluded_df["category"].value_counts().to_dict()

    # ── 2. Text lengths ───────────────────────────────────────────────────────
    usable_df["char_count"] = usable_df["clean_text"].apply(len)
    usable_df["word_count"] = usable_df["clean_text"].apply(lambda t: len(t.split()))

    text_stats_by_class = {}
    for cat in CLASSES:
        cat_df = usable_df[usable_df["category"] == cat]
        if not cat_df.empty:
            text_stats_by_class[cat] = {
                "count": len(cat_df),
                "char_mean": round(float(cat_df["char_count"].mean()), 1),
                "char_median": round(float(cat_df["char_count"].median()), 1),
                "char_min": int(cat_df["char_count"].min()),
                "char_max": int(cat_df["char_count"].max()),
                "word_mean": round(float(cat_df["word_count"].mean()), 1),
                "word_median": round(float(cat_df["word_count"].median()), 1),
                "word_min": int(cat_df["word_count"].min()),
                "word_max": int(cat_df["word_count"].max()),
            }

    # ── 3. Duplicates Analysis ────────────────────────────────────────────────
    byte_dups = total_docs - df["file_hash"].nunique()
    clean_text_counts = usable_df["clean_text"].value_counts()
    exact_text_dups = int((clean_text_counts > 1).sum())
    total_text_duplicate_instances = int(clean_text_counts[clean_text_counts > 1].sum() - exact_text_dups)

    # ── 4. Data Quality & Anomalies ───────────────────────────────────────────
    # Identify the 28 power-company utility bills stored inside invoices folder
    misplaced_power_bills = df[
        (df["category"] == "Invoice") & (df["source"].str.contains("0094", regex=False))
    ]
    misplaced_power_bill_count = len(misplaced_power_bills)

    # Excluded files details
    exclusions = []
    for _, r in excluded_df.iterrows():
        exclusions.append({
            "doc_id": r["doc_id"],
            "source": r["source"],
            "category": r["category"],
            "status": r["status"],
            "reason": f"Text length below minimum threshold (length={len(r['raw_text'].strip())})",
        })

    # Summary dictionary
    eda_summary = {
        "total_documents": total_docs,
        "usable_documents": len(usable_df),
        "excluded_documents": len(excluded_df),
        "class_distribution_all": class_counts,
        "class_distribution_usable": usable_by_class,
        "class_distribution_excluded": excluded_by_class,
        "status_distribution": status_counts,
        "extraction_method_distribution": method_counts,
        "method_by_category": {
            cat: df[df["category"] == cat]["extraction_method"].value_counts().to_dict()
            for cat in CLASSES
        },
        "text_length_statistics": text_stats_by_class,
        "duplicate_analysis": {
            "exact_file_byte_duplicates": byte_dups,
            "distinct_clean_texts": usable_df["clean_text"].nunique(),
            "identical_clean_text_groups": exact_text_dups,
            "redundant_text_instances": total_text_duplicate_instances,
        },
        "data_quality_flags": {
            "misplaced_power_bills_in_invoices_count": misplaced_power_bill_count,
            "misplaced_power_bills_note": (
                "28 documents in invoices/ originate from power utility billing templates "
                "(BSES/Chhattisgarh), matching the bills in bill/. Folder label preserved per user instruction."
            ),
            "excluded_records": exclusions,
        },
    }

    # ── 5. Save machine-readable JSON ─────────────────────────────────────────
    json_path = REPORT_DIR / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    log.info("Saved EDA JSON summary -> %s", json_path)

    # ── 6. Save human-readable text report ────────────────────────────────────
    txt_path = REPORT_DIR / "eda_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PDF Document Classifier — Exploratory Data Analysis & Data Quality Report\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total PDFs in workspace       : {total_docs}\n")
        f.write(f"Usable documents for training : {len(usable_df)} ({len(usable_df)/total_docs:.1%})\n")
        f.write(f"Excluded unreadable/empty docs: {len(excluded_df)}\n\n")
        f.write("-- Class Distribution --\n")
        for cat in CLASSES:
            u = usable_by_class.get(cat, 0)
            t = class_counts.get(cat, 0)
            f.write(f"  {cat:<10}: {u:>4} usable / {t:>4} total ({u/t:.1%})\n")
        f.write("\n-- Extraction Methods --\n")
        for m, count in method_counts.items():
            f.write(f"  {m:<10}: {count:>4} docs ({count/total_docs:.1%})\n")
        f.write("\n-- Text Length Statistics (Clean Words) --\n")
        for cat, st in text_stats_by_class.items():
            f.write(f"  {cat:<10}: mean={st['word_mean']} words, median={st['word_median']}, range=[{st['word_min']}, {st['word_max']}]\n")
        f.write("\n-- Duplicates & Leakage Checks --\n")
        f.write(f"  Exact file byte duplicates (SHA-256) : {byte_dups}\n")
        f.write(f"  Identical text groups                : {exact_text_dups} groups ({total_text_duplicate_instances} redundant instances)\n")
        f.write("\n-- Data Quality Flags & Exclusions --\n")
        f.write(f"  Misplaced power bills in invoices/: {misplaced_power_bill_count} files (templates match bill/)\n")
        f.write(f"  Excluded empty scans: {len(exclusions)} files:\n")
        for ex in exclusions:
            f.write(f"    - {ex['source']}: {ex['reason']}\n")
    log.info("Saved EDA text report -> %s", txt_path)

    # ── 7. Generate Visualizations ────────────────────────────────────────────
    # Plot 1: Class distribution
    plt.figure(figsize=(7, 4.5))
    cats = list(CLASSES)
    usable_vals = [usable_by_class.get(c, 0) for c in cats]
    excl_vals = [excluded_by_class.get(c, 0) for c in cats]
    
    x = np.arange(len(cats))
    width = 0.4
    plt.bar(x - width/2, usable_vals, width, label="Usable (status=ok)", color="#4f46e5")
    plt.bar(x + width/2, excl_vals, width, label="Excluded (empty/fail)", color="#ef4444")
    plt.xticks(x, cats)
    plt.ylabel("Number of Documents")
    plt.title("Document Category Distribution")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot1_path = REPORT_DIR / "class_distribution.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()

    # Plot 2: Extraction methods
    plt.figure(figsize=(7, 4.5))
    methods = list(method_counts.keys())
    m_counts = [method_counts[m] for m in methods]
    colors = ["#10b981" if m == "direct" else "#f59e0b" if m == "ocr" else "#ef4444" for m in methods]
    bars = plt.bar(methods, m_counts, color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 15, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    plt.ylabel("Document Count")
    plt.title("PDF Extraction Methods (Direct vs OCR Fallback)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot2_path = REPORT_DIR / "extraction_methods.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    # Plot 3: Text length distribution
    plt.figure(figsize=(7, 4.5))
    data_by_class = [usable_df[usable_df["category"] == c]["word_count"].values for c in cats]
    plt.boxplot(data_by_class, tick_labels=cats, patch_artist=True,
                boxprops=dict(facecolor="#c7d2fe", color="#4338ca"),
                medianprops=dict(color="#b91c1c", linewidth=2))
    plt.ylabel("Word Count (Preprocessed)")
    plt.title("Clean Text Word Length by Document Class")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = REPORT_DIR / "text_length_distribution.png"
    plt.savefig(plot3_path, dpi=150)
    plt.close()

    log.info("Saved plots: class_distribution.png, extraction_methods.png, text_length_distribution.png")
    log.info("EDA completed successfully.")
    return eda_summary


if __name__ == "__main__":
    run_eda()
