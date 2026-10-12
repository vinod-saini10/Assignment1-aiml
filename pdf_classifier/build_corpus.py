"""
build_corpus.py — Discover all PDFs, extract text, and write corpus.csv.

Run from the project root:
    python -m pdf_classifier.build_corpus

What it does
────────────
1. Discovers PDFs in the three source folders (originals, never moved).
2. Extracts text per document (direct → OCR fallback).
3. Detects duplicates by SHA-256 file hash.
4. Runs data quality checks.
5. Writes outputs/corpus.csv with columns:
       doc_id, path, category, raw_text, clean_text, extraction_method,
       file_hash, status, source, page_count, warnings
6. Prints a data inventory summary.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_classifier.config import (
    CLASS_MAP,
    CLASSES,
    CORPUS_CSV,
    LOG_DIR,
    OUTPUT_DIR,
    SOURCE_BILLS_DIR,
    SOURCE_INVOICES_DIR,
    SOURCE_RECEIPTS_DIR,
)
from pdf_classifier.extractor import extract_pdf
from pdf_classifier.logger import get_logger
from pdf_classifier.preprocessor import preprocess

log = get_logger(__name__, LOG_DIR)


# ── Source folder → (Path, canonical category) mapping ─────────────────────────
SOURCES: dict[str, tuple[Path, str]] = {
    "bill":     (SOURCE_BILLS_DIR, "Bill"),
    "invoices": (SOURCE_INVOICES_DIR, "Invoice"),
    "receipts": (SOURCE_RECEIPTS_DIR, "Receipt"),
}


def discover_pdfs() -> list[tuple[Path, str]]:
    """Return list of (pdf_path, category) for all PDFs in source folders."""
    entries: list[tuple[Path, str]] = []
    for key, (folder, cat_name) in SOURCES.items():
        if not folder.exists():
            log.error("Source folder not found: %s", folder)
            continue
        pdfs = sorted(folder.glob("*.pdf"))
        log.info("  %-8s  %4d PDFs  in  %s", cat_name, len(pdfs), folder)
        for p in pdfs:
            entries.append((p, cat_name))
    return entries


def build_corpus() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("PDF Document Classifier — build_corpus")
    log.info("=" * 60)

    entries = discover_pdfs()
    if not entries:
        log.error("No PDFs found. Check source folder paths in config.py.")
        sys.exit(1)

    total = len(entries)
    log.info("Total PDFs discovered: %d", total)

    # Check for existing corpus to avoid re-extracting already-successful docs
    existing_cache: dict[str, dict] = {}
    if CORPUS_CSV.exists():
        try:
            prev_df = pd.read_csv(CORPUS_CSV, dtype=str).fillna("")
            for _, r in prev_df.iterrows():
                if r.get("status") == "ok":
                    existing_cache[str(Path(r["path"]).resolve())] = r.to_dict()
            if existing_cache:
                log.info("Loaded %d previously extracted 'ok' documents from cache.", len(existing_cache))
        except Exception as exc:
            log.warning("Could not load previous corpus cache: %s", exc)

    rows: list[dict] = []
    seen_hashes: dict[str, str] = {}  # hash -> first doc_id
    duplicate_count = 0

    for idx, (pdf_path, category) in enumerate(entries, start=1):
        doc_id = f"{category[:3].upper()}_{idx:05d}"
        source_label = f"{category}/{pdf_path.name}"
        resolved_path = str(pdf_path.resolve())

        if resolved_path in existing_cache:
            row = existing_cache[resolved_path]
            row["category"] = CLASS_MAP.get(row.get("category", ""), category)
            fhash = row.get("file_hash", "")
            is_dup = False
            if fhash in seen_hashes:
                is_dup = True
                duplicate_count += 1
            else:
                seen_hashes[fhash] = row.get("doc_id", doc_id)
            row["is_duplicate"] = is_dup
            rows.append(row)
            continue

        if idx % 20 == 0 or idx == 1 or len(rows) % 20 == 0:
            log.info("Processing %d / %d  (%s)", idx, total, pdf_path.name[:60])

        result = extract_pdf(pdf_path, doc_id, category, source_label)

        # Duplicate detection
        is_duplicate = False
        if result.file_hash in seen_hashes:
            is_duplicate = True
            duplicate_count += 1
            result.warnings.append(f"duplicate_of:{seen_hashes[result.file_hash]}")
            log.debug("Duplicate: %s == %s", doc_id, seen_hashes[result.file_hash])
        else:
            seen_hashes[result.file_hash] = doc_id

        clean_text = preprocess(result.text)

        rows.append({
            "doc_id":            result.doc_id,
            "path":              result.path,
            "category":          CLASS_MAP.get(result.category, result.category),
            "raw_text":          result.text,
            "clean_text":        clean_text,
            "extraction_method": result.method,
            "file_hash":         result.file_hash,
            "status":            result.status,
            "source":            result.source,
            "page_count":        result.page_count,
            "is_duplicate":      is_duplicate,
            "warnings":          "|".join(result.warnings),
        })

    df = pd.DataFrame(rows)

    # -- Data Quality Report ---------------------------------------------------
    log.info("")
    log.info("-- Data Quality Report ----------------------------------")
    log.info("Total documents   : %d", len(df))
    log.info("Duplicates        : %d", duplicate_count)
    log.info("Status breakdown  :")
    for status, cnt in df["status"].value_counts().items():
        log.info("  %-18s %d", status, cnt)
    log.info("Extraction method :")
    for meth, cnt in df["extraction_method"].value_counts().items():
        log.info("  %-18s %d", meth, cnt)
    log.info("Category counts   :")
    for cat in CLASSES:
        n = (df["category"] == cat).sum()
        log.info("  %-8s %d", cat, n)

    empty_mask = df["status"].isin(["empty", "ocr_unavailable", "corrupt", "encrypted"])
    if empty_mask.any():
        log.warning("%d documents have no usable text - they will be excluded from training.",
                    empty_mask.sum())

    # ── Save (with backup preservation) ───────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS_CSV.exists():
        backup_path = OUTPUT_DIR / "corpus_backup.csv"
        try:
            import shutil
            shutil.copy2(CORPUS_CSV, backup_path)
            log.info("Preserved existing corpus backup -> %s", backup_path)
        except Exception as exc:
            log.warning("Could not create corpus backup: %s", exc)

    df.to_csv(CORPUS_CSV, index=False, encoding="utf-8")
    log.info("")
    log.info("Corpus saved -> %s", CORPUS_CSV)
    return df


if __name__ == "__main__":
    build_corpus()
