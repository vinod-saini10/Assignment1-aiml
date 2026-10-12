"""
preprocessor.py — Shared text preprocessing pipeline.

The SAME preprocessing is applied during:
  • corpus building (build_corpus.py)
  • model training  (train.py)
  • inference       (predict.py / Streamlit app)

This guarantees no train/serving skew.
"""
from __future__ import annotations

import re
import unicodedata


# Characters/patterns to normalise away
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHA_NUM_RE = re.compile(r"[^a-zA-Z0-9\s]")  # keep uppercase for NUM token
_DIGITS_RE = re.compile(r"\b\d+\b")      # standalone numbers → token 'NUM'


def preprocess(text: str, *, replace_digits: bool = True) -> str:
    """
    Clean and normalise raw document text for TF-IDF vectorisation.

    Steps
    ─────
    1. Unicode NFKC normalisation (handles accented / composed characters)
    2. Lower-case
    3. Remove control characters
    4. Optionally replace standalone digit sequences with 'NUM'
    5. Remove remaining non-alphanumeric characters (keeps a-z, A-Z, 0-9, spaces)
    6. Collapse whitespace

    Parameters
    ──────────
    text          : raw extracted text
    replace_digits: if True, standalone integers become the token 'NUM'
                    (helps generalise across document IDs, amounts, dates)

    Returns
    ───────
    Cleaned string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Lower
    text = text.lower()

    # 3. Strip control characters (keep \n/\t → space below)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" or ch in "\n\t" else " " for ch in text)

    # 4. Digit replacement
    if replace_digits:
        text = _DIGITS_RE.sub("NUM", text)

    # 5. Remove punctuation / symbols
    text = _NON_ALPHA_NUM_RE.sub(" ", text)

    # 6. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
