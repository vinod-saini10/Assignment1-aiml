"""
app.py — Streamlit web UI for PDF Document Classification.

Run from the project root:
    streamlit run pdf_classifier/app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from pdf_classifier.config import CLASSES, INCONCLUSIVE_THRESHOLD, PIPELINE_PATH, REPORT_DIR
from pdf_classifier.predict import PredictionResult, predict_pdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Document Classifier",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }

    /* Hero title */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.55);
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Class badge */
    .badge {
        display: inline-block;
        padding: 0.35em 0.9em;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-invoice { background: linear-gradient(90deg, #6366f1, #8b5cf6); color: #fff; }
    .badge-bill    { background: linear-gradient(90deg, #f59e0b, #f97316); color: #fff; }
    .badge-receipt { background: linear-gradient(90deg, #10b981, #059669); color: #fff; }

    /* Warning banner */
    .warn-banner {
        background: rgba(239,68,68,0.2);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    /* Info banner */
    .info-banner {
        background: rgba(99,102,241,0.15);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #c7d2fe;
        font-size: 0.9rem;
    }

    /* Progress bar override */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.5) !important; }
    [data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📄 PDF Document Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload a PDF and let the model classify it as an Invoice, Bill, or Receipt</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        This tool uses a **TF-IDF + ML** pipeline trained on your real data:
        - 🟣 **Invoice** — Superstore purchase invoices
        - 🟠 **Bill** — Power/energy utility bills
        - 🟢 **Receipt** — Scanned payment receipts

        **Extraction**: Direct PDF text → OCR fallback (Tesseract)

        **⚠️ Limitations**
        - 3-class classifier only; **cannot** reliably identify unknown document types
        - Confidence < {:.0%} → labelled *inconclusive*
        - OCR quality affects receipt accuracy
        """.format(INCONCLUSIVE_THRESHOLD)
    )

    st.markdown("---")
    st.markdown("## 📊 Model Info")
    if PIPELINE_PATH.exists():
        import joblib
        pipeline = joblib.load(PIPELINE_PATH)
        clf_name = type(pipeline.named_steps["clf"]).__name__
        st.success(f"**Model loaded**: {clf_name}")
    else:
        st.error("No trained model found.\nRun `python -m pdf_classifier.train` first.")

    if REPORT_DIR.exists():
        report_json = REPORT_DIR / "evaluation.json"
        if report_json.exists():
            import json
            with open(report_json, encoding="utf-8") as f:
                rep = json.load(f)
            best = rep.get("selected_model", rep.get("best_model", "?"))
            test_res = rep.get("final_held_out_test_metrics", {})
            if not test_res:
                test_res = rep.get(f"{best.replace('LogisticRegression','LR').replace('LinearSVC','SVM')}_test", {})
            st.markdown(f"**Best model**: {best}")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{test_res.get('accuracy', 0):.1%}")
            col2.metric("Macro F1", f"{test_res.get('macro_f1', 0):.3f}")

# ── Main area ─────────────────────────────────────────────────────────────────
model_ready = PIPELINE_PATH.exists()

if not model_ready:
    st.markdown(
        '<div class="warn-banner">⚠️ No trained model found. '
        'Run <code>python -m pdf_classifier.train</code> before using this app.</div>',
        unsafe_allow_html=True,
    )

uploaded = st.file_uploader(
    "Drop a PDF here or click to browse",
    type=["pdf"],
    disabled=not model_ready,
    help="Only PDF files are supported (max 25MB).",
)

if uploaded and model_ready:
    MAX_FILE_SIZE_MB = 25
    if uploaded.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Uploaded file exceeds {MAX_FILE_SIZE_MB}MB limit. Please upload a smaller PDF.")
        st.stop()

    with st.spinner("Extracting text and classifying …"):
        # Save to temp file (Streamlit gives us a BytesIO-like object)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            pr: PredictionResult = predict_pdf(tmp_path)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Extraction warnings ───────────────────────────────────────────────────
    if pr.extraction.status != "ok":
        st.markdown(
            f'<div class="warn-banner">⚠️ Extraction issue — status: '
            f'<strong>{pr.extraction.status}</strong>. '
            f'Text is insufficient for reliable classification.</div>',
            unsafe_allow_html=True,
        )

    if pr.extraction.warnings:
        with st.expander("🔔 Extraction warnings"):
            for w in pr.extraction.warnings:
                st.write(f"• {w}")

    # ── Result card ───────────────────────────────────────────────────────────
    cls = pr.predicted_class
    badge_cls = f"badge-{cls.lower()}"

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;margin-bottom:1rem;">'
        f'<span class="badge {badge_cls}">{cls}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if pr.is_inconclusive or cls == "Inconclusive":
        max_prob = max(pr.probabilities.values()) if pr.probabilities else 0.0
        st.markdown(
            f'<div class="warn-banner" style="text-align:center;">'
            f'⚠️ <strong>Inconclusive prediction</strong> — '
            f'highest confidence is {max_prob:.1%}, below the '
            f'{INCONCLUSIVE_THRESHOLD:.0%} threshold. '
            f'Manual review is recommended.</div>',
            unsafe_allow_html=True,
        )
    else:
        max_prob = pr.probabilities.get(cls, 0.0)
        st.markdown(
            f'<div class="info-banner" style="text-align:center;">'
            f'✅ Classified as <strong>{cls}</strong> with '
            f'{max_prob:.1%} confidence.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Class probability bars ────────────────────────────────────────────────
    st.markdown("### 📊 Class Scores")
    note_col = st.columns(1)[0]
    note_col.caption(
        "Scores represent model posterior probabilities for the closed-world 3 classes."
    )

    colors = {"Invoice": "#6366f1", "Bill": "#f59e0b", "Receipt": "#10b981"}
    for cat in CLASSES:
        prob = pr.probabilities.get(cat, 0.0)
        col1, col2, col3 = st.columns([2, 5, 1])
        col1.markdown(f"**{cat.title()}**")
        col2.progress(min(prob, 1.0))
        col3.markdown(f"`{prob:.2%}`")

    # ── Document metadata ─────────────────────────────────────────────────────
    with st.expander("📎 Document metadata"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages", pr.extraction.page_count)
        m2.metric("Extraction", pr.extraction.method)
        m3.metric("Status", pr.extraction.status)
        st.caption(f"File hash (SHA-256): `{pr.extraction.file_hash[:24]}…`")

    # ── Extracted text preview ────────────────────────────────────────────────
    with st.expander("📝 Extracted text preview (preprocessed)"):
        if pr.clean_text_snippet:
            st.text(pr.clean_text_snippet + ("…" if len(pr.clean_text_snippet) == 500 else ""))
        else:
            st.warning("No text could be extracted from this document.")

elif not uploaded:
    # Empty state
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
            <div style="color:rgba(255,255,255,0.6);font-size:1rem;">
                Upload a PDF above to get started
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
