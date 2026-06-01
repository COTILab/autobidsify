# tests/test_universal_core.py
# Unit tests for autobidsify/universal_core.py
# Updated to match current codebase.
# Tests cover: FileStructureAnalyzer, UniversalFileMatcher, SmartFileGrouper.

import pytest
from autobidsify.universal_core import (
    FileStructureAnalyzer,
    UniversalFileMatcher,
    SmartFileGrouper,
)


# ============================================================================
# FileStructureAnalyzer — analyze_directory_structure
# ============================================================================

class TestAnalyzeDirectoryStructure:

    def test_flat_structure_max_depth_zero(self):
        files = ["VHMCT1mm-Hip.dcm", "VHFCT1mm-Hip.dcm"]
        fsa = FileStructureAnalyzer(files)
        result = fsa.analyze_directory_structure()
        assert result["max_depth"] == 0

    def test_bids_structure_template(self):
        files = [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-02_T1w.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.analyze_directory_structure()
        assert result["max_depth"] >= 1
        assert "sub" in result["structure_template"].lower() or \
               "subject" in result["structure_template"].lower()

    def test_camcan_three_level_structure(self):
        files = [
            "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan.nii.gz",
            "Beijing_sub82352/anat_mprage_anonymized/NIfTI/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.analyze_directory_structure()
        assert result["max_depth"] == 3

    def test_returns_unique_dir_names(self):
        files = [
            "sub-01/anat/scan.nii.gz",
            "sub-01/func/scan.nii.gz",
            "sub-02/anat/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.analyze_directory_structure()
        assert "anat" in result["unique_dir_names"]
        assert "func" in result["unique_dir_names"]

    def test_empty_file_list(self):
        fsa = FileStructureAnalyzer([])
        result = fsa.analyze_directory_structure()
        assert result["max_depth"] == 0


# ============================================================================
# FileStructureAnalyzer — detect_subject_identifiers
# ============================================================================

class TestDetectSubjectIdentifiers:

    def test_camcan_site_sub_pattern(self):
        files = [
            "Newark_sub41006/anat/scan.nii.gz",
            "Beijing_sub82352/anat/scan.nii.gz",
            "Cambridge_sub06272/anat/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.detect_subject_identifiers()
        assert result["best_candidate"] is not None
        assert result["best_candidate"]["pattern_name"] == "site_sub_id"
        assert result["best_candidate"]["count"] == 3

    def test_bids_standard_sub_pattern(self):
        files = [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-02_T1w.nii.gz",
            "sub-10/anat/sub-10_T1w.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.detect_subject_identifiers()
        assert result["best_candidate"] is not None
        assert result["best_candidate"]["count"] == 3

    def test_user_hint_improves_confidence(self):
        files = [
            "sub-01/anat/scan.nii.gz",
            "sub-02/anat/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result_with_hint = fsa.detect_subject_identifiers(user_hint=2)
        result_without   = fsa.detect_subject_identifiers()
        # With correct hint, score should be >= without hint
        if result_with_hint["best_candidate"] and result_without["best_candidate"]:
            assert result_with_hint["best_candidate"]["score"] >= \
                   result_without["best_candidate"]["score"]

    def test_confidence_values_are_valid(self):
        files = ["sub-01/anat/scan.nii.gz", "sub-02/anat/scan.nii.gz"]
        fsa = FileStructureAnalyzer(files)
        result = fsa.detect_subject_identifiers()
        assert result["confidence"] in ("high", "medium", "low", "none")

    def test_flat_files_no_directory_pattern(self):
        files = ["VHMCT1mm-Hip.dcm", "VHFCT1mm-Hip.dcm"]
        fsa = FileStructureAnalyzer(files)
        result = fsa.detect_subject_identifiers()
        # No directory-based pattern since files are flat
        if result["best_candidate"]:
            assert result["best_candidate"]["type"] != "directory_pattern" or \
                   result["best_candidate"]["count"] == 0

    def test_site_sub_pattern_has_site_metadata(self):
        files = [
            "NewYork_sub001/anat/scan.nii.gz",
            "London_sub002/anat/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.detect_subject_identifiers()
        if result["best_candidate"]:
            assert result["best_candidate"].get("metadata", {}).get("has_site") is True


# ============================================================================
# FileStructureAnalyzer — detect_duplicate_filenames
# ============================================================================

class TestDetectDuplicateFilenames:

    def test_detects_duplicates_across_paths(self):
        files = [
            "sub-01/BRIK/scan.nii.gz",
            "sub-01/NIfTI/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        dupes = fsa.detect_duplicate_filenames()
        assert "scan.nii.gz" in dupes
        assert len(dupes["scan.nii.gz"]) == 2

    def test_no_duplicates_when_unique(self):
        files = [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-02_T1w.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        dupes = fsa.detect_duplicate_filenames()
        # Different filenames — no duplicates
        assert "sub-01_T1w.nii.gz" not in dupes

    def test_empty_file_list_no_duplicates(self):
        fsa = FileStructureAnalyzer([])
        assert fsa.detect_duplicate_filenames() == {}


# ============================================================================
# FileStructureAnalyzer — build_directory_tree_summary
# ============================================================================

class TestBuildDirectoryTreeSummary:

    def test_returns_required_keys(self):
        files = [
            "sub-01/anat/scan.nii.gz",
            "sub-02/func/rest.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.build_directory_tree_summary()
        assert "subject_structure_samples" in result
        assert "total_subjects_detected" in result
        assert "sampled_subjects" in result

    def test_total_subjects_correct(self):
        files = [
            "sub-01/anat/scan.nii.gz",
            "sub-02/anat/scan.nii.gz",
            "sub-03/anat/scan.nii.gz",
        ]
        fsa = FileStructureAnalyzer(files)
        result = fsa.build_directory_tree_summary()
        assert result["total_subjects_detected"] == 3

    def test_sampling_respects_max_subjects(self):
        files = [f"sub-{i:03d}/anat/scan.nii.gz" for i in range(1, 101)]
        fsa = FileStructureAnalyzer(files)
        result = fsa.build_directory_tree_summary(max_subjects=10)
        assert result["sampled_subjects"] <= 10


# ============================================================================
# UniversalFileMatcher — parse_pattern_features
# ============================================================================

class TestParsePatternFeatures:

    def test_double_star_nii_gz(self):
        feat = UniversalFileMatcher.parse_pattern_features("**/*.nii.gz")
        assert feat["extension"] == ".nii.gz"

    def test_double_star_with_path_keyword(self):
        feat = UniversalFileMatcher.parse_pattern_features(
            "**/anat_mprage_anonymized/*.nii.gz"
        )
        assert feat["extension"] == ".nii.gz"
        assert any("anat" in kw for kw in feat["path_keywords"])

    def test_returns_dict_with_expected_keys(self):
        feat = UniversalFileMatcher.parse_pattern_features("**/*.dcm")
        for key in ["type", "path_keywords", "filename_prefix",
                     "filename_keywords", "extension", "exclude_keywords"]:
            assert key in feat


# ============================================================================
# UniversalFileMatcher — match_file
# ============================================================================

class TestMatchFile:

    def test_extension_match(self):
        feat = {"extension": ".nii.gz", "path_keywords": [],
                "filename_prefix": None, "filename_keywords": [],
                "exclude_keywords": []}
        assert UniversalFileMatcher.match_file("sub-01/anat/scan.nii.gz", feat)

    def test_extension_no_match(self):
        feat = {"extension": ".nii.gz", "path_keywords": [],
                "filename_prefix": None, "filename_keywords": [],
                "exclude_keywords": []}
        assert not UniversalFileMatcher.match_file("sub-01/anat/scan.dcm", feat)

    def test_path_keyword_required(self):
        feat = {"extension": ".nii.gz", "path_keywords": ["nifti"],
                "filename_prefix": None, "filename_keywords": [],
                "exclude_keywords": []}
        assert UniversalFileMatcher.match_file(
            "sub-01/NIfTI/scan.nii.gz", feat)
        assert not UniversalFileMatcher.match_file(
            "sub-01/BRIK/scan.nii.gz", feat)

    def test_exclude_keyword_blocks_match(self):
        feat = {"extension": ".nii.gz", "path_keywords": [],
                "filename_prefix": None, "filename_keywords": [],
                "exclude_keywords": ["brik"]}
        assert not UniversalFileMatcher.match_file(
            "sub-01/BRIK/scan.nii.gz", feat)
        assert UniversalFileMatcher.match_file(
            "sub-01/NIfTI/scan.nii.gz", feat)

    def test_filename_prefix_match(self):
        feat = {"extension": ".dcm", "path_keywords": [],
                "filename_prefix": "vhm", "filename_keywords": [],
                "exclude_keywords": []}
        assert UniversalFileMatcher.match_file("VHMCT1mm-Hip.dcm", feat)
        assert not UniversalFileMatcher.match_file("VHFCT1mm-Hip.dcm", feat)


# ============================================================================
# UniversalFileMatcher — match_files_batch
# ============================================================================

class TestMatchFilesBatch:

    def test_matches_all_nii_gz(self):
        files = [
            "sub-01/anat/scan.nii.gz",
            "sub-01/func/rest.nii.gz",
            "sub-02/anat/scan.nii.gz",
            "sub-01/anat/scan.dcm",     # should not match
        ]
        matched = UniversalFileMatcher.match_files_batch(
            files, ["**/*.nii.gz"]
        )
        assert len(matched) == 3
        assert all(f.endswith(".nii.gz") for f in matched)

    def test_excludes_brik(self):
        files = [
            "sub-01/BRIK/scan.nii.gz",
            "sub-01/NIfTI/scan.nii.gz",
        ]
        matched = UniversalFileMatcher.match_files_batch(
            files, ["**/*.nii.gz"], exclude_patterns=["**/BRIK/**"]
        )
        assert len(matched) == 1
        assert "NIfTI" in matched[0]

    def test_empty_file_list(self):
        matched = UniversalFileMatcher.match_files_batch(
            [], ["**/*.nii.gz"]
        )
        assert matched == []

    def test_no_duplicates_in_result(self):
        files = ["sub-01/anat/scan.nii.gz"]
        matched = UniversalFileMatcher.match_files_batch(
            files, ["**/*.nii.gz", "**/*.nii.gz"]
        )
        assert matched.count("sub-01/anat/scan.nii.gz") == 1
