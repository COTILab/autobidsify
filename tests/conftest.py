# tests/conftest.py
# Shared pytest fixtures for the entire test suite.
# Updated: added eeg_file_list fixture.

import pytest
from pathlib import Path


# ============================================================================
# Shared path fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the autobidsify project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tests_dir():
    """Return the tests/ directory."""
    return Path(__file__).parent


# ============================================================================
# Dataset file-list fixtures (no real files needed)
# ============================================================================

@pytest.fixture
def camcan_file_list():
    """CamCAN: multi-site hierarchical MRI. site_subID/scantype/format/file."""
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
    """Visible Human Project: flat DICOM. VHM=male, VHF=female."""
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
    """Already-BIDS file paths (sub-XX directories)."""
    return [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-03/func/sub-03_task-rest_bold.nii.gz",
    ]


@pytest.fixture
def nirs_file_list():
    """fNIRS dataset with mixed formats: .snirf, .nirs, .mat."""
    return [
        "sub-01/nirs/sub-01_task-rest_nirs.snirf",
        "sub-02_run-01.nirs",
        "sub-03_data.mat",
        "sub-04/nirs/sub-04_task-motor_nirs.snirf",
    ]


@pytest.fixture
def eeg_file_list():
    """
    EEG dataset: flat EDF files, 3 subjects × 2 tasks (_1=rest, _2=arithmetic).
    Mirrors the mental-arithmetic EDF dataset structure.
    """
    return [
        "Subject00_1.edf",
        "Subject00_2.edf",
        "Subject01_1.edf",
        "Subject01_2.edf",
        "Subject02_1.edf",
        "Subject02_2.edf",
        "subject-info.csv",
        "README.txt",
        "SHA256SUMS.txt",
    ]


@pytest.fixture
def eeg_hierarchical_file_list():
    """EEG dataset with sub-XX directory structure."""
    return [
        "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "sub-01/eeg/sub-01_task-rest_eeg.json",
        "sub-01/eeg/sub-01_task-rest_channels.tsv",
        "sub-02/eeg/sub-02_task-rest_eeg.edf",
        "sub-02/eeg/sub-02_task-rest_channels.tsv",
    ]


# ============================================================================
# Minimal on-disk BIDS dataset
# ============================================================================

@pytest.fixture
def minimal_bids_dir(tmp_path):
    """
    Minimal valid BIDS dataset on disk for integration testing.

    Structure:
        tmp_path/
          dataset_description.json
          README.md
          participants.tsv
          sub-01/anat/sub-01_T1w.nii.gz   (empty placeholder)
    """
    import json

    dd = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.10.0",
        "License": "CC0",
        "DatasetType": "raw",
    }
    (tmp_path / "dataset_description.json").write_text(json.dumps(dd, indent=2))
    (tmp_path / "README.md").write_text("# Test Dataset\n")
    (tmp_path / "participants.tsv").write_text("participant_id\nsub-01\n")

    sub_dir = tmp_path / "sub-01" / "anat"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_T1w.nii.gz").write_bytes(b"")

    return tmp_path


@pytest.fixture
def minimal_eeg_bids_dir(tmp_path):
    """
    Minimal valid EEG BIDS dataset on disk.

    Structure:
        tmp_path/
          dataset_description.json
          README.md
          participants.tsv
          sub-00/eeg/sub-00_task-rest_eeg.edf        (empty)
          sub-00/eeg/sub-00_task-rest_eeg.json       (minimal)
          sub-00/eeg/sub-00_task-rest_channels.tsv   (minimal)
    """
    import json

    dd = {
        "Name": "EEG Test",
        "BIDSVersion": "1.10.0",
        "License": "CC0",
        "DatasetType": "raw",
    }
    (tmp_path / "dataset_description.json").write_text(json.dumps(dd, indent=2))
    (tmp_path / "README.md").write_text("# EEG Test\n")
    (tmp_path / "participants.tsv").write_text("participant_id\nsub-00\n")

    eeg_dir = tmp_path / "sub-00" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-00_task-rest_eeg.edf").write_bytes(b"")
    (eeg_dir / "sub-00_task-rest_eeg.json").write_text(
        json.dumps({"TaskName": "rest", "SamplingFrequency": 500})
    )
    (eeg_dir / "sub-00_task-rest_channels.tsv").write_text(
        "name\ttype\tunits\nFp1\tEEG\tmicroV\n"
    )

    return tmp_path


# ============================================================================
# Minimal ingest_info fixture
# ============================================================================

@pytest.fixture
def minimal_ingest_info(tmp_path):
    """
    Write a minimal ingest_info.json and return the output directory.
    """
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
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
        "status": "complete",
    }
    (staging / "ingest_info.json").write_text(json.dumps(ingest_info))
    return tmp_path
