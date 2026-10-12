"""
extractor.py — PDF text extraction with OCR fallback.

Strategy
────────
1. Open the PDF with PyMuPDF.
2. For each page, try direct text extraction (get_text).
3. If the extracted text is below MIN_TEXT_LENGTH chars, render the page to
   a Pillow image and run Tesseract OCR.
4. Aggregate page texts into a single document-level string.
5. Track: extraction method ('direct' | 'ocr' | 'mixed' | 'failed'),
           status ('ok' | 'empty' | 'encrypted' | 'corrupt' | 'ocr_unavailable'),
           and any per-page warnings.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF

from pdf_classifier.config import (
    MIN_TEXT_LENGTH,
    MAX_PAGES_PER_DOC,
    OCR_DPI,
    OCR_LANG,
    TESSERACT_CMD,
    TESSDATA_DIR,
    LOG_DIR,
)
from pdf_classifier.logger import get_logger

log = get_logger(__name__, LOG_DIR)


# ── Lazy Tesseract / PyMuPDF OCR setup ────────────────────────────────────────
_tesseract_available: Optional[bool] = None


def _check_tesseract() -> bool:
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    # 1. Check PyMuPDF built-in OCR with local tessdata
    if TESSDATA_DIR.exists() and any(TESSDATA_DIR.glob("*.traineddata")):
        _tesseract_available = True
        log.info("PyMuPDF embedded Tesseract OCR available via %s.", TESSDATA_DIR)
        return True

    # 2. Check external Tesseract binary via pytesseract
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        log.info("External Tesseract OCR available.")
    except Exception as exc:
        log.warning("OCR engine not available: %s — scanned pages will be empty.", exc)
        _tesseract_available = False

    return _tesseract_available


def _ocr_page(page: pymupdf.Page) -> str:
    """Run OCR via PyMuPDF native Tesseract engine or pytesseract fallback."""
    if not _check_tesseract():
        return ""

    # Strategy A: PyMuPDF native OCR (fastest, self-contained)
    if TESSDATA_DIR.exists() and any(TESSDATA_DIR.glob("*.traineddata")):
        try:
            tp = page.get_textpage_ocr(
                dpi=OCR_DPI,
                full=True,
                language=OCR_LANG,
                tessdata=str(TESSDATA_DIR),
            )
            text = tp.extractText()
            if text and text.strip():
                return text
        except Exception as exc:
            log.debug("PyMuPDF OCR error: %s, falling back to pytesseract", exc)

    # Strategy B: External pytesseract fallback
    try:
        import pytesseract
        from PIL import Image

        mat = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except Exception as exc:
        log.debug("pytesseract error: %s", exc)
        return ""


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ExtractionResult:
    doc_id: str
    path: str
    category: str
    text: str
    method: str          # 'direct' | 'ocr' | 'mixed' | 'failed'
    status: str          # 'ok' | 'empty' | 'encrypted' | 'corrupt' | 'ocr_unavailable'
    file_hash: str
    page_count: int
    source: str          # logical source label, e.g. 'bill/0094…'
    warnings: list[str] = field(default_factory=list)


# ── File hash ─────────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main extraction function ──────────────────────────────────────────────────
def extract_pdf(
    pdf_path: Path,
    doc_id: str,
    category: str,
    source_label: str,
) -> ExtractionResult:
    """
    Extract text from a single PDF.  Returns an ExtractionResult regardless
    of success or failure — callers should check result.status.
    """
    path_str = str(pdf_path)
    fhash = _sha256(pdf_path)
    warnings: list[str] = []

    # ── Open document ─────────────────────────────────────────────────────────
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        log.error("Cannot open %s: %s", pdf_path.name, exc)
        return ExtractionResult(
            doc_id=doc_id, path=path_str, category=category,
            text="", method="failed", status="corrupt",
            file_hash=fhash, page_count=0, source=source_label,
            warnings=[f"open_error: {exc}"],
        )

    # ── Encryption check ──────────────────────────────────────────────────────
    if doc.is_encrypted:
        doc.close()
        log.warning("Encrypted: %s", pdf_path.name)
        return ExtractionResult(
            doc_id=doc_id, path=path_str, category=category,
            text="", method="failed", status="encrypted",
            file_hash=fhash, page_count=0, source=source_label,
            warnings=["encrypted"],
        )

    page_count = doc.page_count
    n_pages = min(page_count, MAX_PAGES_PER_DOC)
    if page_count > MAX_PAGES_PER_DOC:
        warnings.append(f"truncated_to_{MAX_PAGES_PER_DOC}_pages")

    page_texts: list[str] = []
    methods_used: set[str] = set()

    for i in range(n_pages):
        try:
            page = doc[i]
            raw = page.get_text("text").strip()

            if len(raw) >= MIN_TEXT_LENGTH:
                page_texts.append(raw)
                methods_used.add("direct")
            else:
                # Fallback: OCR
                ocr_text = _ocr_page(page).strip()
                if len(ocr_text) >= MIN_TEXT_LENGTH:
                    page_texts.append(ocr_text)
                    methods_used.add("ocr")
                else:
                    if not _tesseract_available:
                        methods_used.add("ocr_unavailable")
                    else:
                        warnings.append(f"page_{i}_empty_after_ocr")
        except Exception as exc:
            warnings.append(f"page_{i}_error: {exc}")
            log.debug("Page %d error in %s: %s", i, pdf_path.name, exc)

    doc.close()

    full_text = "\n\n".join(page_texts)

    # ── Determine method label ─────────────────────────────────────────────────
    if "direct" in methods_used and "ocr" in methods_used:
        method = "mixed"
    elif "ocr" in methods_used:
        method = "ocr"
    elif "direct" in methods_used:
        method = "direct"
    elif "ocr_unavailable" in methods_used:
        method = "failed"
    else:
        method = "failed"

    # ── Determine status ──────────────────────────────────────────────────────
    if "ocr_unavailable" in methods_used and len(full_text.strip()) == 0:
        status = "ocr_unavailable"
    elif len(full_text.strip()) < MIN_TEXT_LENGTH:
        status = "empty"
    else:
        status = "ok"

    if status != "ok":
        log.warning("%-12s  %s  [%s]", status, pdf_path.name, method)

    return ExtractionResult(
        doc_id=doc_id, path=path_str, category=category,
        text=full_text, method=method, status=status,
        file_hash=fhash, page_count=page_count, source=source_label,
        warnings=warnings,
    )
