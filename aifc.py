"""
Minimal aifc shim for Python 3.13+
This project does not rely on reading AIFF files. Some third-party audio
libraries import `aifc` unconditionally; this stub satisfies the import.
Any attempt to open AIFF will raise a clear error.
"""
from typing import Any

class Error(Exception):
    """Raised when AIFF operations are attempted in this environment."""

def open(file: Any, mode: str | None = None):
    raise Error("AIFF (aifc) is not supported in this environment.")
