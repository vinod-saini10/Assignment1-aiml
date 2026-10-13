"""
config.py — Central configuration for the PDF Document Classifier project.
All paths are derived from this file's location (project root = parent directory).
"""
from pathlib import Path

# ── Project root (two levels up from this file: pdf_classifier/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Raw source data  (originals – NEVER modified by this project)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_BILLS_DIR    = RAW_DATA_DIR / "bills"
RAW_INVOICES_DIR = RAW_DATA_DIR / "invoices"
RAW_RECEIPTS_DIR = RAW_DATA_DIR / "receipts"

# ── Source folders (the original folders you already have)
SOURCE_BILLS_DIR    = PROJECT_ROOT / "bill"
SOURCE_INVOICES_DIR = PROJECT_ROOT / "invoices"
SOURCE_RECEIPTS_DIR = PROJECT_ROOT / "receipts"

# ── Outputs
OUTPUT_DIR    = PROJECT_ROOT / "pdf_classifier" / "outputs"
CORPUS_CSV    = OUTPUT_DIR / "corpus.csv"
REPORT_DIR    = OUTPUT_DIR / "reports"
MODEL_DIR     = OUTPUT_DIR / "models"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
LOG_DIR       = OUTPUT_DIR / "logs"

# ── Class labels (order matters for confusion matrix / report)
CLASSES = ["Bill", "Invoice", "Receipt"]
CLASS_MAP = {
    "bill": "Bill",
    "invoice": "Invoice",
    "receipt": "Receipt",
    "Bill": "Bill",
    "Invoice": "Invoice",
    "Receipt": "Receipt",
}

# ── Extraction settings
MIN_TEXT_LENGTH = 30      # chars; pages below this trigger OCR fallback
MAX_PAGES_PER_DOC = 50   # safety cap for memory
OCR_DPI = 200             # render DPI for scanned pages
OCR_LANG = "eng"
SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
MAX_FILE_SIZE_MB = 25

# ── Tesseract binary and tessdata paths
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = PROJECT_ROOT / "tessdata"

# ── Model / split settings
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15   # fraction of (train+val) remaining after test split
MIN_CLASS_COUNT = 10   # minimum docs per class to proceed with training

# ── TF-IDF settings
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 2
TFIDF_SUBLINEAR_TF = True

# ── Confidence threshold below which the app warns "inconclusive"
INCONCLUSIVE_THRESHOLD = 0.60
