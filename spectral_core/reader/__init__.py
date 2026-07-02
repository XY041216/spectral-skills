"""Public spectral-reader core API."""

from .version import CORE_VERSION, SCHEMA_VERSION
from .workflow import read_spectral_dataset

__all__ = [
    "CORE_VERSION",
    "SCHEMA_VERSION",
    "read_spectral_dataset",
]
