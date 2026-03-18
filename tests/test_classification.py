# tests/test_classification.py
# Unit tests for autobidsify/stages/classification.py
# Tests cover: _detect_extension, classify_and_stage (logic only),
#              _stage_files_to_pools (with real files on disk),
#              classify_files (end-to-end with staging dir).
# No LLM calls. All tests are pure Python / file I/O.

import json
import pytest
from pathlib import Path

from autobidsify.stages.classification import (
    _detect_extension,
    classify_and_stage,
    classify_files,
    MRI_EXTS,
    NIRS_EXTS,
    CLASSIFICATION_PLAN_FILENAME,
)
from autobidsify.constants import NIRS_POOL, MRI_POOL, UNKNOWN_POOL, STAGING_DIR


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp(tmp_path):
    """Clean temporary directory for each test."""
    return tmp_path


@pytest.fixture
def minimal_output_dir(tmp_path):
    """
    Create a minimal pipeline output directory with:
      - _staging/
      - a small fake dataset under data/
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # MRI files
    (data_dir / "VHMCT1mm-Hip (134).dcm").write_bytes(b"fake dcm")
    (data_dir / "VHMCT1mm-Hip (135).dcm").write_bytes(b"fake dcm")
    (data_dir / "VHFCT1mm-Hip (45).dcm").write_bytes(b"fake dcm")
    (data_dir / "sub-01_T1w.nii.gz").write_bytes(b"fake nii")
    (data_dir / "sub-01_T1w.nii").write_bytes(b"fake nii")

    # fNIRS files
    (data_dir / "sub-01_nirs.snirf").write_bytes(b"fake snirf")
    (data_dir / "sub-02_data.nirs").write_bytes(b"fake nirs")
    (data_dir / "sub-03_data.mat").write_bytes(b"fake mat")

    # Auxiliary / unknown files
    (data_dir / "dataset_description.json").write_text('{"Name": "Test"}')
    (data_dir / "README.md").write_text("# Test dataset")
    (data_dir / "participants.tsv").write_text("participant_id\nsub-01\n")
    (data_dir / "notes.txt").write_text("some notes")

    # Staging dir
    staging = tmp_path / STAGING_DIR
    staging.mkdir()

    return tmp_path, data_dir


@pytest.fixture
def minimal_bundle(minimal_output_dir):
    """
    Build a minimal evidence bundle dict matching the files in minimal_output_dir.
    """
    output_dir, data_dir = minimal_output_dir
    files = [
        str(p.relative_to(data_dir)).replace("\\", "/")
        for p in data_dir.rglob("*")
        if p.is_file()
    ]
    return {
        "root":      str(data_dir),
        "all_files": sorted(files),
    }


# ============================================================================
# _detect_extension
# ============================================================================

class TestDetectExtension:

    @pytest.mark.parametrize("relpath, expected", [
        # Compound extension must be handled before suffix split
        ("sub-01/anat/scan.nii.gz",          ".nii.gz"),
        ("VHMCT1mm-Hip (134).nii.gz",        ".nii.gz"),
        # Standard single extensions
        ("VHMCT1mm-Hip (134).dcm",           ".dcm"),
        ("sub-01_T1w.nii",                   ".nii"),
        ("sub-01_signal.snirf",              ".snirf"),
        ("sub-02_data.nirs",                 ".nirs"),
        ("sub-03_homer.mat",                 ".mat"),
        ("sub-01.jnii",                      ".jnii"),
        ("sub-01.bnii",                      ".bnii"),
        # Auxiliary / unknown extensions
        ("dataset_description.json",         ".json"),
        ("README.md",                        ".md"),
        ("participants.tsv",                 ".tsv"),
        ("notes.txt",                        ".txt"),
        ("protocol.pdf",                     ".pdf"),
        # Case insensitivity
        ("SCAN.DCM",                         ".dcm"),
        ("SCAN.NII.GZ",                      ".nii.gz"),
        ("SIGNAL.SNIRF",                     ".snirf"),
        # No extension
        ("Makefile",                         ""),
        # Nested path
        ("sub-01/anat/sub-01_T1w.nii.gz",   ".nii.gz"),
    ])
    def test_extension_detection(self, relpath, expected):
        assert _detect_extension(relpath) == expected

    def test_nii_gz_takes_priority_over_suffix(self):
        # Path.suffix would return ".gz" — must be overridden
        from pathlib import Path as _Path
        assert _Path("scan.nii.gz").suffix == ".gz"          # confirm stdlib behavior
        assert _detect_extension("scan.nii.gz") == ".nii.gz" # our fix is correct


# ============================================================================
# Module-level constants
# ============================================================================

class TestModuleConstants:

    def test_mri_exts_contains_expected(self):
        for ext in [".dcm", ".nii", ".nii.gz", ".jnii", ".bnii"]:
            assert ext in MRI_EXTS

    def test_nirs_exts_contains_expected(self):
        for ext in [".snirf", ".nirs", ".mat"]:
            assert ext in NIRS_EXTS

    def test_no_overlap_between_mri_and_nirs(self):
        assert MRI_EXTS.isdisjoint(NIRS_EXTS), \
            f"Overlap detected: {MRI_EXTS & NIRS_EXTS}"

    def test_mat_only_in_nirs(self):
        assert ".mat" in NIRS_EXTS
        assert ".mat" not in MRI_EXTS

    def test_classification_plan_filename(self):
        assert CLASSIFICATION_PLAN_FILENAME == "classification_plan.json"


# ============================================================================
# classify_and_stage — classification logic (no real files needed)
# ============================================================================

class TestClassifyAndStageLogic:
    """
    Tests that focus on the classification logic only.
    We use a bundle whose 'root' points to a real directory (tmp_path),
    but we only care about the returned plan — not the copied files.
    """

    def _make_bundle(self, tmp_path, file_list):
        """Helper: create bundle pointing to tmp_path with given file list."""
        # Create empty placeholder files so root.exists() passes
        # and _stage_files_to_pools doesn't log missing warnings
        for relpath in file_list:
            p = tmp_path / relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        return {"root": str(tmp_path), "all_files": file_list}

    def test_mri_files_classified_correctly(self, tmp_path):
        files = ["scan.dcm", "scan.nii", "scan.nii.gz", "scan.jnii", "scan.bnii"]
        bundle = self._make_bundle(tmp_path, files)
        staging = tmp_path / STAGING_DIR
        staging.mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert set(plan["mri_files"]) == set(files)
        assert plan["nirs_files"] == []
        assert plan["unknown_files"] == []

    def test_nirs_files_classified_correctly(self, tmp_path):
        files = ["signal.snirf", "data.nirs", "homer.mat"]
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert set(plan["nirs_files"]) == set(files)
        assert plan["mri_files"] == []
        assert plan["unknown_files"] == []

    def test_auxiliary_files_go_to_unknown(self, tmp_path):
        files = [
            "dataset_description.json",
            "README.md",
            "participants.tsv",
            "notes.txt",
            "protocol.pdf",
            "sub-01_T1w.json",   # BIDS sidecar
        ]
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert set(plan["unknown_files"]) == set(files)
        assert plan["mri_files"] == []
        assert plan["nirs_files"] == []

    def test_mixed_dataset_split_correctly(self, tmp_path):
        mri   = ["scan1.dcm", "scan2.nii.gz"]
        nirs  = ["signal.snirf", "data.mat"]
        aux   = ["README.md", "participants.tsv"]
        files = mri + nirs + aux
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert set(plan["mri_files"])     == set(mri)
        assert set(plan["nirs_files"])    == set(nirs)
        assert set(plan["unknown_files"]) == set(aux)

    def test_empty_all_files_returns_empty_plan(self, tmp_path):
        bundle = {"root": str(tmp_path), "all_files": []}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert plan["mri_files"]     == []
        assert plan["nirs_files"]    == []
        assert plan["unknown_files"] == []

    def test_counts_field_matches_lists(self, tmp_path):
        files = ["scan.dcm", "signal.snirf", "README.md"]
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        counts = plan["counts"]
        assert counts["mri_files"]     == len(plan["mri_files"])
        assert counts["nirs_files"]    == len(plan["nirs_files"])
        assert counts["unknown_files"] == len(plan["unknown_files"])
        assert counts["all_files"]     == len(files)

    def test_plan_contains_required_keys(self, tmp_path):
        bundle = {"root": str(tmp_path), "all_files": []}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        for key in ["mri_files", "nirs_files", "unknown_files",
                    "classification_method", "classification_rules", "counts"]:
            assert key in plan, f"Missing key: {key}"

    def test_classification_method_value(self, tmp_path):
        bundle = {"root": str(tmp_path), "all_files": []}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert plan["classification_method"] == "extension_based"

    def test_classification_rules_is_list_of_strings(self, tmp_path):
        bundle = {"root": str(tmp_path), "all_files": []}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert isinstance(plan["classification_rules"], list)
        for rule in plan["classification_rules"]:
            assert isinstance(rule, str)

    def test_mat_classified_as_nirs_not_mri(self, tmp_path):
        files = ["homer_data.mat"]
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert "homer_data.mat" in plan["nirs_files"]
        assert "homer_data.mat" not in plan["mri_files"]

    def test_nii_gz_classified_as_mri_not_unknown(self, tmp_path):
        files = ["sub-01_T1w.nii.gz"]
        bundle = self._make_bundle(tmp_path, files)
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        plan = classify_and_stage(bundle, tmp_path)
        assert "sub-01_T1w.nii.gz" in plan["mri_files"]
        assert "sub-01_T1w.nii.gz" not in plan["unknown_files"]


# ============================================================================
# classify_and_stage — defensive checks
# ============================================================================

class TestClassifyAndStageDefensiveChecks:

    def test_missing_root_field_calls_fatal(self, tmp_path, capsys):
        """bundle without 'root' should call fatal() and return empty dict."""
        bundle = {"all_files": ["scan.dcm"]}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        # fatal() calls sys.exit — catch SystemExit
        with pytest.raises(SystemExit):
            classify_and_stage(bundle, tmp_path)

    def test_nonexistent_root_calls_fatal(self, tmp_path):
        """bundle with root pointing to a non-existent directory."""
        bundle = {
            "root":      str(tmp_path / "does_not_exist"),
            "all_files": ["scan.dcm"],
        }
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        with pytest.raises(SystemExit):
            classify_and_stage(bundle, tmp_path)

    def test_empty_all_files_warns_but_continues(self, tmp_path, recwarn):
        """Empty all_files should warn but not crash."""
        bundle = {"root": str(tmp_path), "all_files": []}
        (tmp_path / STAGING_DIR).mkdir(exist_ok=True)
        # Should not raise
        plan = classify_and_stage(bundle, tmp_path)
        assert isinstance(plan, dict)


# ============================================================================
# classify_and_stage — plan file written to disk
# ============================================================================

class TestClassifyAndStagePlanFile:

    def test_plan_file_created(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        classify_and_stage(minimal_bundle, output_dir)
        plan_path = output_dir / STAGING_DIR / CLASSIFICATION_PLAN_FILENAME
        assert plan_path.exists()

    def test_plan_file_is_valid_json(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        classify_and_stage(minimal_bundle, output_dir)
        plan_path = output_dir / STAGING_DIR / CLASSIFICATION_PLAN_FILENAME
        with open(plan_path) as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)

    def test_plan_file_counts_match(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        plan = classify_and_stage(minimal_bundle, output_dir)
        plan_path = output_dir / STAGING_DIR / CLASSIFICATION_PLAN_FILENAME
        with open(plan_path) as f:
            saved = json.load(f)
        assert saved["counts"]["mri_files"]     == len(plan["mri_files"])
        assert saved["counts"]["nirs_files"]    == len(plan["nirs_files"])
        assert saved["counts"]["unknown_files"] == len(plan["unknown_files"])


# ============================================================================
# _stage_files_to_pools — real file copying
# ============================================================================

class TestStageFilesToPools:

    def test_mri_files_copied_to_mri_pool(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        plan = classify_and_stage(minimal_bundle, output_dir)
        mri_pool = output_dir / MRI_POOL
        for relpath in plan["mri_files"]:
            assert (mri_pool / relpath).exists(), \
                f"Expected MRI file in pool: {relpath}"

    def test_nirs_files_copied_to_nirs_pool(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        plan = classify_and_stage(minimal_bundle, output_dir)
        nirs_pool = output_dir / NIRS_POOL
        for relpath in plan["nirs_files"]:
            assert (nirs_pool / relpath).exists(), \
                f"Expected fNIRS file in pool: {relpath}"

    def test_unknown_files_copied_to_unknown_pool(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        plan = classify_and_stage(minimal_bundle, output_dir)
        unknown_pool = output_dir / UNKNOWN_POOL
        for relpath in plan["unknown_files"]:
            assert (unknown_pool / relpath).exists(), \
                f"Expected auxiliary file in unknown pool: {relpath}"

    def test_mri_files_not_in_nirs_pool(self, minimal_output_dir, minimal_bundle):
        output_dir, _ = minimal_output_dir
        plan = classify_and_stage(minimal_bundle, output_dir)
        nirs_pool = output_dir / NIRS_POOL
        for relpath in plan["mri_files"]:
            assert not (nirs_pool / relpath).exists(), \
                f"MRI file should not be in NIRS pool: {relpath}"

    def test_directory_structure_preserved_in_pool(self, tmp_path):
        """Files in subdirectories preserve their relative path in the pool."""
        data_dir = tmp_path / "data"
        (data_dir / "sub-01" / "anat").mkdir(parents=True)
        (data_dir / "sub-01" / "anat" / "scan.dcm").write_bytes(b"dcm")
        (tmp_path / STAGING_DIR).mkdir()

        bundle = {
            "root":      str(data_dir),
            "all_files": ["sub-01/anat/scan.dcm"],
        }
        classify_and_stage(bundle, tmp_path)
        expected = tmp_path / MRI_POOL / "sub-01" / "anat" / "scan.dcm"
        assert expected.exists()

    def test_missing_source_file_warns_but_does_not_crash(self, tmp_path):
        """A file listed in all_files but absent on disk should warn, not crash."""
        (tmp_path / STAGING_DIR).mkdir()
        bundle = {
            "root":      str(tmp_path),
            "all_files": ["ghost_file.dcm"],   # does not exist on disk
        }
        # Should not raise
        plan = classify_and_stage(bundle, tmp_path)
        assert "ghost_file.dcm" in plan["mri_files"]
        # Pool should be empty (file was missing)
        mri_pool = tmp_path / MRI_POOL
        assert not list(mri_pool.rglob("*.dcm"))


# ============================================================================
# classify_files — end-to-end CLI entry point
# ============================================================================

class TestClassifyFiles:

    def _write_evidence_bundle(self, output_dir, data_dir, file_list):
        """Write a minimal evidence_bundle.json to _staging/."""
        bundle = {
            "root":      str(data_dir),
            "all_files": file_list,
        }
        staging = output_dir / STAGING_DIR
        staging.mkdir(parents=True, exist_ok=True)
        bundle_path = staging / "evidence_bundle.json"
        bundle_path.write_text(json.dumps(bundle))

    def test_classify_files_creates_plan(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scan.dcm").write_bytes(b"dcm")

        self._write_evidence_bundle(
            tmp_path, data_dir, ["scan.dcm"]
        )
        classify_files(tmp_path)
        plan_path = tmp_path / STAGING_DIR / CLASSIFICATION_PLAN_FILENAME
        assert plan_path.exists()

    def test_classify_files_missing_bundle_calls_fatal(self, tmp_path):
        """No evidence_bundle.json → fatal()."""
        (tmp_path / STAGING_DIR).mkdir()
        with pytest.raises(SystemExit):
            classify_files(tmp_path)

    def test_classify_files_full_pipeline(self, tmp_path):
        """Full end-to-end: write bundle, run classify, check all pools."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scan.dcm").write_bytes(b"dcm")
        (data_dir / "signal.snirf").write_bytes(b"snirf")
        (data_dir / "README.md").write_text("readme")

        self._write_evidence_bundle(
            tmp_path, data_dir, ["scan.dcm", "signal.snirf", "README.md"]
        )
        classify_files(tmp_path)

        assert (tmp_path / MRI_POOL  / "scan.dcm").exists()
        assert (tmp_path / NIRS_POOL / "signal.snirf").exists()
        assert (tmp_path / UNKNOWN_POOL / "README.md").exists()