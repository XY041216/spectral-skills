"""Reader preview-stage errors and codes."""

from __future__ import annotations


INPUT_PATH_NOT_FOUND = "INPUT_PATH_NOT_FOUND"
UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
PERMISSION_DENIED = "PERMISSION_DENIED"
DECODE_FAILED = "DECODE_FAILED"
EMPTY_FILE = "EMPTY_FILE"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
MALFORMED_FILE = "MALFORMED_FILE"
PREVIEW_LIMIT_REACHED = "PREVIEW_LIMIT_REACHED"


class SpectralReaderError(Exception):
    """Base exception for future spectral-reader implementation errors."""

    code = "SPECTRAL_READER_ERROR"


class PreviewError(SpectralReaderError):
    """Preview-stage exception with a stable error code."""

    def __init__(self, message: str, *, code: str = "PREVIEW_ERROR") -> None:
        super().__init__(message)
        self.code = code
