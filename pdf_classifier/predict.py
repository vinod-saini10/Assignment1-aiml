"""
predict.py — Single-document prediction using the saved pipeline.

Can be imported by the Streamlit app or called from the CLI:
    python -m pdf_classifier.predict path/to/document.pdf
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from pdf_classifier.config import (
    CLASSES,
    INCONCLUSIVE_THRESHOLD,
    PIPELINE_PATH,
)
from pdf_classifier.extractor import ExtractionResult, extract_document
from pdf_classifier.preprocessor import preprocess


@dataclass
class PredictionResult:
    predicted_class: str
    probabilities: dict[str, float]   # class → probability (LR only)
    is_inconclusive: bool
    extraction: ExtractionResult
    clean_text_snippet: str           # first 500 chars of clean text


def load_pipeline():
    """Load the saved joblib pipeline. Raises FileNotFoundError if absent."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {PIPELINE_PATH}. "
            "Please run `python -m pdf_classifier.train` first."
        )
    return joblib.load(PIPELINE_PATH)


def predict_document(document_path: Path) -> PredictionResult:
    """
    Extract text from *pdf_path* and classify it.

    Returns a PredictionResult with predicted class, class probabilities
    (when available), and an inconclusive flag when max probability
    is below INCONCLUSIVE_THRESHOLD.
    """
    pipeline = load_pipeline()

    doc_id = f"PRED_{document_path.stem[:20]}"
    result = extract_document(document_path, doc_id, "unknown", str(document_path))
    clean  = preprocess(result.text)

    # ── Handle unreadable or empty documents safely ───────────────────────────
    if result.status != "ok" or not clean.strip():
        return PredictionResult(
            predicted_class="Inconclusive",
                probabilities={},
            is_inconclusive=True,
            extraction=result,
            clean_text_snippet="",
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    clf = pipeline.named_steps["clf"]
    has_proba = hasattr(clf, "predict_proba")

    if has_proba:
        proba_arr = pipeline.predict_proba([clean])[0]
        proba_dict = {cls: float(p) for cls, p in zip(pipeline.classes_, proba_arr)}
        predicted = pipeline.classes_[int(np.argmax(proba_arr))]
        max_prob = float(np.max(proba_arr))
    else:
        # LinearSVC — use decision function as a proxy (normalised to [0,1])
        dec = pipeline.decision_function([clean])[0]
        exp_dec = np.exp(dec - np.max(dec))
        proba_arr = exp_dec / exp_dec.sum()
        proba_dict = {cls: float(p) for cls, p in zip(pipeline.classes_, proba_arr)}
        predicted = pipeline.predict([clean])[0]
        max_prob = float(np.max(proba_arr))

    if (
        set(proba_dict) != set(CLASSES)
        or not np.isfinite(list(proba_dict.values())).all()
        or any(prob < 0.0 or prob > 1.0 for prob in proba_dict.values())
    ):
        return PredictionResult(
            predicted_class="Inconclusive",
            probabilities={},
            is_inconclusive=True,
            extraction=result,
            clean_text_snippet=clean[:500],
        )

    is_inconclusive = max_prob < INCONCLUSIVE_THRESHOLD

    snippet = clean[:500] if clean else ""

    return PredictionResult(
        predicted_class=predicted,
        probabilities=proba_dict,
        is_inconclusive=is_inconclusive,
        extraction=result,
        clean_text_snippet=snippet,
    )

def predict_pdf(pdf_path: Path) -> PredictionResult:
    """Backward-compatible PDF prediction wrapper."""
    return predict_document(pdf_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pdf_classifier.predict <path_to_document>")
        sys.exit(1)

    document = Path(sys.argv[1])
    if not document.exists():
        print(f"File not found: {document}")
        sys.exit(1)

    pr = predict_document(document)
    print(f"\nPredicted class : {pr.predicted_class}")
    print(f"Probabilities   : { {k: f'{v:.3f}' for k,v in pr.probabilities.items()} }")
    print(f"Inconclusive    : {pr.is_inconclusive}")
    print(f"Extraction      : {pr.extraction.method}  status={pr.extraction.status}")
    if pr.extraction.warnings:
        print(f"Warnings        : {pr.extraction.warnings}")
