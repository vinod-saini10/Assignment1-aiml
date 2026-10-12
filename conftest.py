# conftest.py — pytest root configuration
# Makes the project root importable during test collection
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()))
