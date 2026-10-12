"""
tests/test_pipeline.py — Comprehensive tests for end-to-end pipeline,
model inference, uncertainty handling, category consistency, and edge cases.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf_classifier.config import CLASSES, INCONCLUSIVE_THRESHOLD, PIPELINE_PATH, TESSDATA_DIR
from pdf_classifier.extractor import extract_pdf
from pdf_classifier.predict import PredictionResult, load_pipeline, predict_pdf


def _make_pdf(text: str) -> Path:
    """Create a temporary synthetic PDF."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    p = Path(tmp.name)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(p))
    doc.close()
    return p


def test_pipeline_loading():
    """Verify saved model pipeline exists, loads, and has expected components."""
    pipeline = load_pipeline()
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_category_consistency():
    """Ensure pipeline output classes strictly match config CLASSES in exact casing."""
    pipeline = load_pipeline()
    model_classes = list(pipeline.classes_)
    assert sorted(model_classes) == sorted(CLASSES)
    assert set(model_classes) == {"Bill", "Invoice", "Receipt"}


def test_predict_synthetic_invoice():
    """Synthetic Superstore purchase invoice should be classified as Invoice."""
    p = _make_pdf(
        "INVOICE\nSuperstore Purchase Order CA-2023-99881\n"
        "Customer: Jane Doe\nItem: Office Supplies Paper\n"
        "Rate: $25.00 Qty: 4 Total Amount: $100.00\nPayment Terms: Net 30"
    )
    try:
        res = predict_pdf(p)
        assert isinstance(res, PredictionResult)
        assert res.predicted_class == "Invoice"
        assert res.probabilities["Invoice"] > 0.50
        assert res.extraction.status == "ok"
    finally:
        p.unlink(missing_ok=True)


def test_predict_synthetic_bill():
    """Synthetic power utility bill should be classified as Bill."""
    p = _make_pdf(
        "CENTRAL TRANSMISSION UTILITY OF INDIA LIMITED\n"
        "BILL OF SUPPLY FOR INTER-STATE TRANSMISSION CHARGES\n"
        "DIC Name: BSES Rajdhani Power Limited Delhi Discom\n"
        "Billing Month: April 2024\nBill No: MI11360601260026\n"
        "Total Transmission Charges Payable: Rs. 4,500,000.00\nRTGS Payment to SBI"
    )
    try:
        res = predict_pdf(p)
        assert isinstance(res, PredictionResult)
        assert res.predicted_class == "Bill"
        assert res.probabilities["Bill"] > 0.50
        assert res.extraction.status == "ok"
    finally:
        p.unlink(missing_ok=True)


def test_predict_corrupt_file():
    """Corrupt file must return inconclusive without raising unhandled exception."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"CORRUPT GARBAGE HEADER")
    tmp.close()
    p = Path(tmp.name)
    try:
        res = predict_pdf(p)
        assert res.is_inconclusive is True
        assert res.extraction.status == "corrupt"
        assert res.predicted_class == "Inconclusive"
    finally:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_predict_empty_pdf():
    """Empty PDF must return inconclusive without crashing or predicting positive class."""
    doc = pymupdf.open()
    doc.new_page()  # blank page
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    p = Path(tmp.name)
    doc.save(str(p))
    doc.close()
    try:
        res = predict_pdf(p)
        assert res.is_inconclusive is True
        assert res.extraction.status in ("empty", "ocr_unavailable")
        assert res.predicted_class == "Inconclusive"
    finally:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_inconclusive_threshold():
    """Inconclusive threshold must be a float between 0.5 and 0.9."""
    assert 0.50 <= INCONCLUSIVE_THRESHOLD <= 0.90


def test_ocr_engine_available():
    """Verify PyMuPDF embedded OCR with local tessdata is available."""
    assert TESSDATA_DIR.exists()
    assert (TESSDATA_DIR / "eng.traineddata").exists()
