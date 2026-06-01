# tests/test_validators.py
# Unit tests for autobidsify/converters/validators.py
# Tests cover _internal_bids_validation (pure Python, no external tools).
# _run_python_bids_validator and _run_npm_bids_validator are NOT tested here
# (require installed tools or subprocess).

import json
import pytest
from pathlib import Path

from autobidsify.converters.validators import _internal_bids_validation


# ============================================================================
# Helpers
# ============================================================================

def _make_bids_dir(tmp_path: Path, *, name="Test", bids_version="1.10.0",
                   license_="CC0", include_readme=True,
                   include_participants=True, n_subjects=1,
                   invalid_json=False) -> Path:
    """Create a minimal BIDS directory on disk."""
    bids_root = tmp_path / "bids_compatible"
    bids_root.mkdir()

    # dataset_description.json
    dd = {"Name": name, "BIDSVersion": bids_version, "License": license_}
    if invalid_json:
        (bids_root / "dataset_description.json").write_text("{ bad json }")
    else:
        (bids_root / "dataset_description.json").write_text(
            json.dumps(dd, indent=2)
        )

    if include_readme:
        (bids_root / "README.md").write_text("# Test\n")

    if include_participants:
        rows = "\n".join(f"sub-{i:02d}" for i in range(1, n_subjects + 1))
        (bids_root / "participants.tsv").write_text(f"participant_id\n{rows}\n")

    # Subject directories
    for i in range(1, n_subjects + 1):
        sub_dir = bids_root / f"sub-{i:02d}" / "anat"
        sub_dir.mkdir(parents=True)
        (sub_dir / f"sub-{i:02d}_T1w.nii.gz").write_bytes(b"")

    return bids_root


# ============================================================================
# _internal_bids_validation
# ============================================================================

class TestInternalBidsValidation:

    # ── Valid dataset passes ─────────────────────────────────────────────

    def test_valid_dataset_no_errors(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path)
        result = _internal_bids_validation(bids_root)
        assert result["issues"]["errors"] == []

    def test_valid_dataset_validator_field(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path)
        result = _internal_bids_validation(bids_root)
        assert result["validator"] == "internal"
        assert result["available"] is True

    def test_valid_dataset_has_summary(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, n_subjects=3)
        result = _internal_bids_validation(bids_root)
        assert result["summary"]["subjectCount"] == 3

    # ── Missing dataset_description.json ────────────────────────────────

    def test_missing_dataset_description_is_error(self, tmp_path):
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        (bids_root / "README.md").write_text("# Test\n")
        (bids_root / "participants.tsv").write_text("participant_id\nsub-01\n")
        sub = bids_root / "sub-01" / "anat"
        sub.mkdir(parents=True)
        (sub / "sub-01_T1w.nii.gz").write_bytes(b"")

        result = _internal_bids_validation(bids_root)
        codes = [e["code"] for e in result["issues"]["errors"]]
        assert "MISSING_DATASET_DESCRIPTION" in codes

    # ── Invalid JSON in dataset_description.json ─────────────────────────

    def test_invalid_json_is_error(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, invalid_json=True)
        result = _internal_bids_validation(bids_root)
        codes = [e["code"] for e in result["issues"]["errors"]]
        assert "INVALID_JSON" in codes

    # ── Missing required fields ──────────────────────────────────────────

    def test_missing_name_is_error(self, tmp_path):
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        dd = {"BIDSVersion": "1.10.0", "License": "CC0"}
        (bids_root / "dataset_description.json").write_text(json.dumps(dd))
        (bids_root / "README.md").write_text("# Test\n")
        sub = bids_root / "sub-01" / "anat"
        sub.mkdir(parents=True)
        (sub / "sub-01_T1w.nii.gz").write_bytes(b"")

        result = _internal_bids_validation(bids_root)
        codes = [e["code"] for e in result["issues"]["errors"]]
        assert "MISSING_NAME" in codes

    def test_missing_license_is_error(self, tmp_path):
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        dd = {"Name": "Test", "BIDSVersion": "1.10.0"}
        (bids_root / "dataset_description.json").write_text(json.dumps(dd))
        (bids_root / "README.md").write_text("# Test\n")
        sub = bids_root / "sub-01" / "anat"
        sub.mkdir(parents=True)
        (sub / "sub-01_T1w.nii.gz").write_bytes(b"")

        result = _internal_bids_validation(bids_root)
        codes = [e["code"] for e in result["issues"]["errors"]]
        assert "MISSING_LICENSE" in codes

    def test_missing_bids_version_is_warning(self, tmp_path):
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        dd = {"Name": "Test", "License": "CC0"}
        (bids_root / "dataset_description.json").write_text(json.dumps(dd))
        (bids_root / "README.md").write_text("# Test\n")
        sub = bids_root / "sub-01" / "anat"
        sub.mkdir(parents=True)
        (sub / "sub-01_T1w.nii.gz").write_bytes(b"")

        result = _internal_bids_validation(bids_root)
        w_codes = [w["code"] for w in result["issues"]["warnings"]]
        assert "MISSING_BIDS_VERSION" in w_codes

    # ── Missing README ───────────────────────────────────────────────────

    def test_missing_readme_is_warning(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, include_readme=False)
        result = _internal_bids_validation(bids_root)
        w_codes = [w["code"] for w in result["issues"]["warnings"]]
        assert "MISSING_README" in w_codes

    def test_readme_txt_accepted(self, tmp_path):
        # README.txt is a valid README variant
        bids_root = _make_bids_dir(tmp_path, include_readme=False)
        (bids_root / "README.txt").write_text("# Test\n")
        result = _internal_bids_validation(bids_root)
        w_codes = [w["code"] for w in result["issues"]["warnings"]]
        assert "MISSING_README" not in w_codes

    # ── Missing participants.tsv ─────────────────────────────────────────

    def test_missing_participants_is_warning(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, include_participants=False)
        result = _internal_bids_validation(bids_root)
        w_codes = [w["code"] for w in result["issues"]["warnings"]]
        assert "MISSING_PARTICIPANTS" in w_codes

    # ── No subject directories ───────────────────────────────────────────

    def test_no_subjects_is_error(self, tmp_path):
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        dd = {"Name": "Empty", "BIDSVersion": "1.10.0", "License": "CC0"}
        (bids_root / "dataset_description.json").write_text(json.dumps(dd))
        (bids_root / "README.md").write_text("# Empty\n")
        (bids_root / "participants.tsv").write_text("participant_id\n")

        result = _internal_bids_validation(bids_root)
        codes = [e["code"] for e in result["issues"]["errors"]]
        assert "NO_SUBJECTS" in codes

    # ── Subject count in summary ─────────────────────────────────────────

    def test_subject_count_correct(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, n_subjects=5)
        result = _internal_bids_validation(bids_root)
        assert result["summary"]["subjectCount"] == 5

    def test_total_files_positive(self, tmp_path):
        bids_root = _make_bids_dir(tmp_path, n_subjects=2)
        result = _internal_bids_validation(bids_root)
        assert result["summary"]["totalFiles"] > 0
