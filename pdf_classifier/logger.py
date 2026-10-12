"""
logger.py — Consistent logging setup for the entire project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

_INITIALISED: bool = False


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """
    Return a named logger.  On first call, attaches a StreamHandler (INFO)
    and optionally a FileHandler to *log_dir*.
    """
    global _INITIALISED

    logger = logging.getLogger(name)

    if not _INITIALISED:
        logger.setLevel(logging.DEBUG)

        # Console handler (safe encoding on Windows)
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
        ch = logging.StreamHandler(stream)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (if log_dir provided)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(log_dir / f"run_{ts}.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        _INITIALISED = True

    return logger
