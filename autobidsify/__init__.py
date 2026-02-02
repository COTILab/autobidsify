"""
autobidsify: Automated BIDS standardization tool powered by LLM-first architecture.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from autobidsify.utils import info, warn, fatal, debug
from autobidsify.constants import BIDS_VERSION, MODALITY_MRI, MODALITY_NIRS

__all__ = [
    "__version__",
    "info",
    "warn",
    "fatal",
    "debug",
    "BIDS_VERSION",
    "MODALITY_MRI",
    "MODALITY_NIRS",
]
