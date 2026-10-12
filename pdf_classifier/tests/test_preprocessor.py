"""
tests/test_preprocessor.py — Unit tests for the text preprocessor.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf_classifier.preprocessor import preprocess


def test_empty_string():
    assert preprocess("") == ""


def test_none_returns_empty():
    assert preprocess(None) == ""  # type: ignore


def test_lowercase():
    assert preprocess("INVOICE") == "invoice"


def test_digit_replacement():
    result = preprocess("Amount: 1234 Date: 01/01/2023")
    assert "NUM" in result
    assert "1234" not in result


def test_no_digit_replacement():
    result = preprocess("Amount: 1234", replace_digits=False)
    assert "NUM" not in result
    assert "NUM" not in result


def test_punctuation_removed():
    result = preprocess("Hello, World! #@$")
    assert "," not in result
    assert "!" not in result
    assert "#" not in result


def test_whitespace_collapsed():
    result = preprocess("too   many    spaces")
    assert "  " not in result
    assert result == "too many spaces"


def test_unicode_normalised():
    # NFKC: ligature fi → fi
    result = preprocess("ﬁle")
    assert "ﬁ" not in result


def test_typical_invoice_text():
    text = "INVOICE #12345\nBill To: John Doe\nAmount Due: $500.00"
    result = preprocess(text)
    assert "invoice" in result
    assert "bill" in result
    assert "john doe" in result


def test_typical_bill_text():
    text = "ELECTRICITY BILL\nAccount: 9876543210\nTotal Payable: ₹2,500"
    result = preprocess(text)
    assert "electricity" in result
    assert "bill" in result


def test_control_chars_removed():
    result = preprocess("hello\x00world\x01test")
    assert "\x00" not in result
    assert "\x01" not in result


if __name__ == "__main__":
    # Quick self-test without pytest
    tests = [
        test_empty_string, test_none_returns_empty, test_lowercase,
        test_digit_replacement, test_no_digit_replacement, test_punctuation_removed,
        test_whitespace_collapsed, test_unicode_normalised,
        test_typical_invoice_text, test_typical_bill_text, test_control_chars_removed,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}  -- {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
