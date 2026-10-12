"""
tests/test_extractor.py — Integration tests for the PDF extractor.

These tests use a small synthetic PDF created in-memory by PyMuPDF
(no external files needed).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pymupdf

from pdf_classifier.extractor import ExtractionResult, extract_pdf


def _make_pdf(text: str) -> Path:
    """Create a one-page PDF with the given text and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    p = Path(tmp.name)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(p))
    doc.close()
    return p


def test_digital_pdf_extraction():
    p = _make_pdf("INVOICE #12345\nBill To: Aaron Bergman\nAmount: $500.00")
    try:
        result = extract_pdf(p, "DOC_001", "invoice", "test/invoice")
        assert isinstance(result, ExtractionResult)
        assert result.status == "ok"
        assert result.method == "direct"
        assert "invoice" in result.text.lower() or "INVOICE" in result.text
        assert result.page_count == 1
        assert len(result.file_hash) == 64
    finally:
        p.unlink(missing_ok=True)


def test_empty_pdf_detection():
    """A PDF with no text should have status 'empty' or trigger OCR."""
    doc = pymupdf.open()
    doc.new_page()  # blank page
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    p = Path(tmp.name)
    doc.save(str(p))
    doc.close()
    try:
        result = extract_pdf(p, "DOC_002", "receipt", "test/receipt")
        # Should be empty or ocr_unavailable (no Tesseract in CI)
        assert result.status in ("empty", "ocr_unavailable")
    finally:
        p.unlink(missing_ok=True)


def test_file_hash_consistency():
    """Same file → same hash."""
    p = _make_pdf("Consistent content for hashing test.")
    try:
        r1 = extract_pdf(p, "DOC_A", "bill", "test/bill")
        r2 = extract_pdf(p, "DOC_B", "bill", "test/bill")
        assert r1.file_hash == r2.file_hash
    finally:
        p.unlink(missing_ok=True)


def test_corrupt_pdf():
    """A corrupt (non-PDF) file should return status 'corrupt'."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"NOT A PDF BINARY CONTENT")
    tmp.close()
    p = Path(tmp.name)
    try:
        result = extract_pdf(p, "DOC_CORRUPT", "invoice", "test/corrupt")
        assert result.status == "corrupt"
        assert result.method == "failed"
    finally:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows: file still held by PyMuPDF; temp file will be cleaned by OS


def test_multipage_pdf():
    """Multi-page PDF should aggregate text from all pages."""
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i+1} ELECTRICITY BILL content", fontsize=12)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    p = Path(tmp.name)
    doc.save(str(p))
    doc.close()
    try:
        result = extract_pdf(p, "DOC_MULTI", "bill", "test/bill")
        assert result.page_count == 3
        assert result.status == "ok"
        assert "Page 1" in result.text or "page" in result.text.lower()
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [
        test_digital_pdf_extraction,
        test_empty_pdf_detection,
        test_file_hash_consistency,
        test_corrupt_pdf,
        test_multipage_pdf,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}  -- {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
