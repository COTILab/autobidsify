# tests/conftest.py
# Shared pytest fixtures and configuration for the entire test suite.
# This file is automatically loaded by pytest before any test module runs.

import pytest
from pathlib import Path


# ============================================================================
# Shared path fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the AUTO_BIDSIFY project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tests_dir():
    """Return the tests/ directory."""
    return Path(__file__).parent


# ============================================================================
# Shared small dataset fixtures (reusable across all test modules)
# ============================================================================

@pytest.fixture
def camcan_file_list():
    """
    Representative CamCAN file paths (relative, no real files needed).
    Multi-site, hierarchical: site_subID / scantype / format / file
    """
    return [
        "Newark_sub41006/anat_mprage_anonymized/BRIK/scan_mprage_anonymized.nii.gz",
        "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
        "Newark_sub41006/anat_mprage_skullstripped/NIfTI/scan_mprage_skullstripped.nii.gz",
        "Newark_sub41006/func_rest/NIfTI/scan_rest.nii.gz",
        "Beijing_sub82352/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
        "Beijing_sub82352/func_rest/NIfTI/scan_rest.nii.gz",
        "Cambridge_sub06272/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
    ]


@pytest.fixture
def vh_file_list():
    """
    Visible Human Project flat file list.
    VHM = male, VHF = female. Both are flat DICOM files (no subdirectories).
    """
    return [
        "VHMCT1mm-Hip (134).dcm",
        "VHMCT1mm-Hip (135).dcm",
        "VHMCT1mm-Head (256).dcm",
        "VHMCT1mm-Shoulder (89).dcm",
        "VHFCT1mm-Hip (45).dcm",
        "VHFCT1mm-Head (120).dcm",
        "VHFCT1mm-Ankle (78).dcm",
    ]


@pytest.fixture
def bids_file_list():
    """
    Already-BIDS standard file paths (sub-XX directories).
    """
    return [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-03/func/sub-03_task-rest_bold.nii.gz",
    ]


@pytest.fixture
def nirs_file_list():
    """
    fNIRS dataset with mixed formats: .snirf, .nirs, .mat
    """
    return [
        "sub-01/nirs/sub-01_task-rest_nirs.snirf",
        "sub-02_run-01.nirs",
        "sub-03_data.mat",
        "sub-04/nirs/sub-04_task-motor_nirs.snirf",
    ]


# ============================================================================
# Minimal BIDS dataset fixture (creates real files on disk)
# ============================================================================

@pytest.fixture
def minimal_bids_dir(tmp_path):
    """
    Create a minimal valid BIDS dataset on disk for integration testing.

    Structure:
        tmp_path/
          dataset_description.json
          README.md
          participants.tsv
          sub-01/anat/sub-01_T1w.nii.gz  (empty placeholder)
    """
    import json

    dd = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.10.0",
        "License": "CC0",
        "DatasetType": "raw"
    }
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(dd, indent=2)
    )
    (tmp_path / "README.md").write_text("# Test Dataset\n")
    (tmp_path / "participants.tsv").write_text("participant_id\nsub-01\n")

    sub_dir = tmp_path / "sub-01" / "anat"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_T1w.nii.gz").write_bytes(b"")  # placeholder

    return tmp_path


# ============================================================================
# Minimal ingest_info fixture (used by evidence and execute stages)
# ============================================================================

@pytest.fixture
def minimal_ingest_info(tmp_path):
    """
    Write a minimal ingest_info.json and return the output directory.
    The actual_data_path points to a small data subdirectory.
    """
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Place one placeholder file so evidence doesn't scan an empty dir
    (data_dir / "sub-01_T1w.nii.gz").write_bytes(b"")

    staging = tmp_path / "_staging"
    staging.mkdir()

    ingest_info = {
        "step": "ingest",
        "input_type": "directory",
        "input_path": str(data_dir),
        "actual_data_path": str(data_dir),
        "staging_dir": None,
        "output_dir": str(tmp_path),
        "status": "complete"
    }
    (staging / "ingest_info.json").write_text(json.dumps(ingest_info))

    return tmp_path
