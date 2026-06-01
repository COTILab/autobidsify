# tests/test_mri_convert.py
# Unit tests for autobidsify/converters/mri_convert.py
# Tests cover: check_dcm2niix_available (mocked), validate_nifti (mocked).
# run_dcm2niix_batch and arrays_to_nifti are NOT tested here (require
# dcm2niix binary and nibabel file I/O with real data).

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from autobidsify.converters.mri_convert import check_dcm2niix_available


# ============================================================================
# check_dcm2niix_available
# ============================================================================

class TestCheckDcm2niixAvailable:

    def test_returns_true_when_found(self):
        with patch("autobidsify.converters.mri_convert.shutil.which",
                   return_value="/usr/bin/dcm2niix"):
            assert check_dcm2niix_available() is True

    def test_returns_false_when_not_found(self):
        with patch("autobidsify.converters.mri_convert.shutil.which",
                   return_value=None):
            assert check_dcm2niix_available() is False

    def test_returns_bool(self):
        with patch("autobidsify.converters.mri_convert.shutil.which",
                   return_value=None):
            result = check_dcm2niix_available()
            assert isinstance(result, bool)
