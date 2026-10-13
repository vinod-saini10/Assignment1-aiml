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
import shutil
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


def _resolve_tesseract_exe() -> Optional[str]:
    """Return the best available Tesseract executable path, or None."""
    # Prefer the explicitly configured path — checked first so the result is
    # stable even when Tesseract is absent from the process PATH (e.g. Streamlit).
    if Path(TESSERACT_CMD).exists():
        return TESSERACT_CMD
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    return None


def _check_tesseract() -> bool:
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    # Both PyMuPDF native OCR and pytesseract require the Tesseract executable.
    try:
        import pytesseract

        executable = _resolve_tesseract_exe()
        if not executable:
            raise FileNotFoundError(
                f"Tesseract executable not found at {TESSERACT_CMD!r} and not on PATH. "
                "Install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        # Set the command path explicitly so it works even when Tesseract is
        # not on the system PATH (common when launched via Streamlit or IDEs).
        pytesseract.pytesseract.tesseract_cmd = executable
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        log.info(
            "Tesseract OCR available — exe: %s  tessdata: %s",
            executable,
            TESSDATA_DIR,
        )
    except FileNotFoundError as exc:
        log.warning(
            "tesseract_not_installed: %s",
            exc,
        )
        _tesseract_available = False
    except Exception as exc:
        log.warning(
            "tesseract_not_configured: %s — check TESSERACT_CMD in config.py.",
            exc,
            exc_info=True,
        )
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

        # Re-apply the exe path — Streamlit / IDE launchers may not have
        # Tesseract on their PATH even though the binary is installed.
        exe = _resolve_tesseract_exe()
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe

        mat = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except Exception as exc:
        log.debug("pytesseract error: %s", exc, exc_info=True)
        return ""


def _ocr_image(image_path: Path) -> str:
    """Preprocess an image and extract text with the configured Tesseract."""
    if not _check_tesseract():
        return ""
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        import pytesseract

        # Always resolve and set the executable explicitly — Streamlit and other
        # launchers may not inherit Tesseract on their PATH even when the binary
        # exists at the configured location.
        executable = _resolve_tesseract_exe()
        if not executable:
            log.error(
                "ocr_execution_failed: Tesseract executable lost between _check_tesseract() "
                "and _ocr_image() for %s. Check TESSERACT_CMD in config.py.",
                image_path.name,
            )
            return ""
        pytesseract.pytesseract.tesseract_cmd = executable

        with Image.open(image_path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            scale = 2 if max(image.size) < 1800 else 1
            if scale > 1:
                image = image.resize((image.width * scale, image.height * scale))

            gray = ImageOps.grayscale(image)
            gray = ImageEnhance.Contrast(gray).enhance(1.5).filter(ImageFilter.SHARPEN)
            threshold = gray.point(lambda pixel: 255 if pixel > 180 else 0)
            candidates = (image, gray, threshold)
            texts = [
                pytesseract.image_to_string(candidate, lang=OCR_LANG, config="--psm 6")
                for candidate in candidates
            ]
            # Keep the richest result; thresholding can harm receipts with faint text.
            best = max(texts, key=lambda t: len(t.strip()), default="")
            if not best.strip():
                log.warning(
                    "empty_ocr_result: Tesseract ran on %s but returned no text.",
                    image_path.name,
                )
            return best
    except Exception as exc:
        log.error(
            "ocr_execution_failed for %s: %s",
            image_path.name,
            exc,
            exc_info=True,
        )
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

    @property
    def character_count(self) -> int:
        """Number of characters available to preprocessing/classification."""
        return len(self.text)


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
                        if "tesseract_unavailable: install Tesseract OCR and set TESSERACT_CMD" not in warnings:
                            warnings.append(
                                "tesseract_unavailable: install Tesseract OCR and set TESSERACT_CMD"
                            )
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


def extract_image(
    image_path: Path,
    doc_id: str,
    category: str,
    source_label: str,
) -> ExtractionResult:
    """Validate an image and extract its text with Tesseract OCR."""
    path_str = str(image_path)
    try:
        fhash = _sha256(image_path)
    except OSError:
        fhash = ""
    warnings: list[str] = []

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        log.warning("Cannot read image %s: %s", image_path.name, exc)
        return ExtractionResult(
            doc_id=doc_id, path=path_str, category=category, text="",
            method="failed", status="corrupt", file_hash=fhash,
            page_count=1, source=source_label, warnings=["unreadable_image"],
        )

    text = _ocr_image(image_path).strip()
    if len(text) < MIN_TEXT_LENGTH:
        if not _check_tesseract():
            warnings.append(
                "tesseract_unavailable: install Tesseract OCR and set TESSERACT_CMD"
            )
            status = "ocr_unavailable"
        else:
            warnings.append("insufficient_text_after_ocr")
            status = "empty"
        method = "failed" if status == "ocr_unavailable" else "ocr"
    else:
        status, method = "ok", "ocr"

    return ExtractionResult(
        doc_id=doc_id, path=path_str, category=category, text=text,
        method=method, status=status, file_hash=fhash, page_count=1,
        source=source_label, warnings=warnings,
    )


def extract_document(
    document_path: Path,
    doc_id: str,
    category: str,
    source_label: str,
) -> ExtractionResult:
    """Extract text from a supported PDF or image document."""
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(document_path, doc_id, category, source_label)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return extract_image(document_path, doc_id, category, source_label)
    return ExtractionResult(
        doc_id=doc_id, path=str(document_path), category=category, text="",
        method="failed", status="corrupt", file_hash="", page_count=0,
        source=source_label, warnings=["unsupported_file_type"],
    )
