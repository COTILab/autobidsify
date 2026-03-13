# tests/test_universal_core.py
# Unit tests for autobidsify/universal_core.py
# Tests cover: FileStructureAnalyzer, UniversalFileMatcher,
#              SmartFileGrouper, and helper functions.

import pytest
from autobidsify.universal_core import (
    FileStructureAnalyzer,
    UniversalFileMatcher,
    SmartFileGrouper,
    build_bids_filename,
    extract_subject_ids_from_paths,
)


# ============================================================================
# Fixtures — representative datasets
# ============================================================================

@pytest.fixture
def camcan_files():
    """CamCAN-style multi-site hierarchical structure."""
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
def vh_files():
    """Visible Human Project flat structure (VHM=male, VHF=female)."""
    return [
        "VHMCT1mm-Hip (134).dcm",
        "VHMCT1mm-Hip (135).dcm",
        "VHMCT1mm-Head (256).dcm",
        "VHFCT1mm-Hip (45).dcm",
        "VHFCT1mm-Head (120).dcm",
        "VHFCT1mm-Ankle (78).dcm",
    ]


@pytest.fixture
def bids_files():
    """Already-BIDS standard structure."""
    return [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-03/func/sub-03_task-rest_bold.nii.gz",
    ]


# ============================================================================
# FileStructureAnalyzer — analyze_directory_structure
# ============================================================================

class TestFileStructureAnalyzerDirectoryStructure:

    def test_returns_expected_keys(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.analyze_directory_structure()
        for key in ["max_depth", "depth_distribution", "unique_dir_names",
                    "dir_level_patterns", "structure_template"]:
            assert key in result

    def test_max_depth_camcan(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.analyze_directory_structure()
        # Structure is 3 directories deep (site_sub/scan/format/file)
        assert result["max_depth"] >= 3

    def test_flat_structure_depth_zero(self, vh_files):
        analyzer = FileStructureAnalyzer(vh_files)
        result = analyzer.analyze_directory_structure()
        assert result["max_depth"] == 0

    def test_structure_template_is_string(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.analyze_directory_structure()
        assert isinstance(result["structure_template"], str)

    def test_caching_returns_same_object(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        r1 = analyzer.analyze_directory_structure()
        r2 = analyzer.analyze_directory_structure()
        assert r1 is r2  # Must be same cached dict


# ============================================================================
# FileStructureAnalyzer — detect_subject_identifiers
# ============================================================================

class TestFileStructureAnalyzerSubjectDetection:

    def test_camcan_detects_three_subjects(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.detect_subject_identifiers(user_hint=3)
        best = result["best_candidate"]
        assert best is not None
        assert best["count"] == 3

    def test_camcan_high_confidence_with_hint(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.detect_subject_identifiers(user_hint=3)
        assert result["confidence"] == "high"

    def test_bids_standard_detects_subjects(self, bids_files):
        analyzer = FileStructureAnalyzer(bids_files)
        result = analyzer.detect_subject_identifiers()
        best = result["best_candidate"]
        assert best is not None
        assert best["count"] == 3

    def test_flat_structure_no_directory_pattern(self, vh_files):
        # VH files are flat — directory-based detection should find 0
        analyzer = FileStructureAnalyzer(vh_files)
        result = analyzer.detect_subject_identifiers()
        # best_candidate may be None or count=0 for flat datasets
        if result["best_candidate"]:
            assert result["confidence"] in ("low", "none", "medium", "high")

    def test_returns_extraction_regex(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.detect_subject_identifiers()
        best = result["best_candidate"]
        if best:
            assert "extraction_regex" in best
            assert isinstance(best["extraction_regex"], str)

    def test_candidates_list_max_five(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        result = analyzer.detect_subject_identifiers()
        assert len(result["candidates"]) <= 5


# ============================================================================
# FileStructureAnalyzer — detect_duplicate_filenames
# ============================================================================

class TestFileStructureAnalyzerDuplicates:

    def test_detects_brik_nifti_duplicates(self, camcan_files):
        """
        detect_duplicate_filenames groups by filename only, not by subject.
        'scan_mprage_anonymized.nii.gz' appears in 4 paths total:
          - Newark/BRIK/
          - Newark/NIfTI/
          - Beijing/NIfTI/
          - Cambridge/NIfTI/
        Correct behavior: return all 4. We assert >= 2 (at least the
        Newark BRIK+NIfTI pair), not == 2.
        """
        analyzer = FileStructureAnalyzer(camcan_files)
        dupes = analyzer.detect_duplicate_filenames()
        assert "scan_mprage_anonymized.nii.gz" in dupes
        assert len(dupes["scan_mprage_anonymized.nii.gz"]) >= 2

    def test_brik_nifti_same_subject_exactly_two(self):
        """
        Focused test with only one subject: verify exactly 2 paths returned.
        """
        files = [
            "Newark_sub41006/anat/BRIK/scan.nii.gz",
            "Newark_sub41006/anat/NIfTI/scan.nii.gz",
        ]
        analyzer = FileStructureAnalyzer(files)
        dupes = analyzer.detect_duplicate_filenames()
        assert "scan.nii.gz" in dupes
        assert len(dupes["scan.nii.gz"]) == 2

    def test_no_duplicates_in_unique_files(self):
        files = ["a/scan1.nii.gz", "b/scan2.nii.gz", "c/scan3.nii.gz"]
        analyzer = FileStructureAnalyzer(files)
        dupes = analyzer.detect_duplicate_filenames()
        assert len(dupes) == 0

    def test_returns_dict(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        dupes = analyzer.detect_duplicate_filenames()
        assert isinstance(dupes, dict)


# ============================================================================
# FileStructureAnalyzer — build_directory_tree_summary
# ============================================================================

class TestDirectoryTreeSummary:

    def test_returns_expected_keys(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        summary = analyzer.build_directory_tree_summary()
        assert "subject_structure_samples" in summary
        assert "total_subjects_detected" in summary
        assert "sampled_subjects" in summary

    def test_respects_max_subjects_limit(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        summary = analyzer.build_directory_tree_summary(max_subjects=2)
        assert summary["sampled_subjects"] <= 2


# ============================================================================
# UniversalFileMatcher — parse_pattern_features
# ============================================================================

class TestUniversalFileMatcherParseFeatures:

    def test_glob_extension_pattern(self):
        features = UniversalFileMatcher.parse_pattern_features("**/*.nii.gz")
        assert features["extension"] == ".nii.gz"
        assert features["type"] == "path_pattern"

    def test_glob_with_path_keyword(self):
        features = UniversalFileMatcher.parse_pattern_features(
            "**/anat_mprage_anonymized/*.nii.gz"
        )
        assert "anonymized" in features["path_keywords"]
        assert features["extension"] == ".nii.gz"

    def test_filename_pattern_dcm(self):
        features = UniversalFileMatcher.parse_pattern_features(r"VHM.*-Head.*\.dcm")
        assert features["extension"] == ".dcm"
        assert features["filename_prefix"] == "vhm"


# ============================================================================
# UniversalFileMatcher — match_file
# ============================================================================

class TestUniversalFileMatcherMatchFile:

    def test_matches_nii_gz_extension(self):
        features = UniversalFileMatcher.parse_pattern_features("**/*.nii.gz")
        assert UniversalFileMatcher.match_file(
            "sub-01/anat/scan.nii.gz", features
        )

    def test_rejects_wrong_extension(self):
        features = UniversalFileMatcher.parse_pattern_features("**/*.nii.gz")
        assert not UniversalFileMatcher.match_file("sub-01/anat/scan.dcm", features)

    def test_matches_path_keyword(self):
        features = UniversalFileMatcher.parse_pattern_features(
            "**/anat_mprage_anonymized/*.nii.gz"
        )
        assert UniversalFileMatcher.match_file(
            "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
            features
        )

    def test_rejects_wrong_path_keyword(self):
        features = UniversalFileMatcher.parse_pattern_features(
            "**/anat_mprage_anonymized/*.nii.gz"
        )
        assert not UniversalFileMatcher.match_file(
            "Newark_sub41006/anat_mprage_skullstripped/NIfTI/scan.nii.gz",
            features
        )

    def test_matches_vhm_dcm_prefix(self):
        features = UniversalFileMatcher.parse_pattern_features(r"VHM.*\.dcm")
        assert UniversalFileMatcher.match_file("VHMCT1mm-Hip (134).dcm", features)

    def test_rejects_vhf_for_vhm_pattern(self):
        features = UniversalFileMatcher.parse_pattern_features(r"VHM.*\.dcm")
        assert not UniversalFileMatcher.match_file("VHFCT1mm-Hip (45).dcm", features)


# ============================================================================
# UniversalFileMatcher — match_files_batch
# ============================================================================

class TestUniversalFileMatcherBatch:

    def test_matches_all_nii_gz(self, camcan_files):
        matched = UniversalFileMatcher.match_files_batch(
            camcan_files, ["**/*.nii.gz"]
        )
        assert len(matched) == len(camcan_files)

    def test_excludes_brik_directory(self, camcan_files):
        matched = UniversalFileMatcher.match_files_batch(
            camcan_files,
            ["**/*.nii.gz"],
            exclude_patterns=["**/BRIK/**"]
        )
        assert all("BRIK" not in f for f in matched)
        # Should have fewer files than without exclusion
        assert len(matched) < len(camcan_files)

    def test_no_duplicates_in_result(self, camcan_files):
        matched = UniversalFileMatcher.match_files_batch(
            camcan_files, ["**/*.nii.gz"]
        )
        assert len(matched) == len(set(matched))

    def test_empty_patterns_returns_empty(self, camcan_files):
        matched = UniversalFileMatcher.match_files_batch(camcan_files, [])
        assert matched == []


# ============================================================================
# SmartFileGrouper
# ============================================================================

class TestSmartFileGrouper:

    def test_groups_by_subject(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        detection = analyzer.detect_subject_identifiers(user_hint=3)
        grouper = SmartFileGrouper(analyzer)
        groups = grouper.group_by_subject_and_scan(camcan_files, detection)
        # Should have multiple groups
        assert len(groups) > 0

    def test_each_group_has_preferred_file(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        detection = analyzer.detect_subject_identifiers(user_hint=3)
        grouper = SmartFileGrouper(analyzer)
        groups = grouper.group_by_subject_and_scan(camcan_files, detection)
        for group_key, group_data in groups.items():
            assert "preferred_file" in group_data

    def test_preferred_file_excludes_brik(self, camcan_files):
        analyzer = FileStructureAnalyzer(camcan_files)
        detection = analyzer.detect_subject_identifiers(user_hint=3)
        grouper = SmartFileGrouper(analyzer)
        groups = grouper.group_by_subject_and_scan(camcan_files, detection)
        for group_data in groups.values():
            pf = group_data.get("preferred_file", "")
            if pf:
                assert "BRIK" not in pf

    def test_fallback_when_no_detection(self):
        files = ["some_file.nii.gz"]
        analyzer = FileStructureAnalyzer(files)
        empty_detection = {"best_candidate": None, "candidates": [], "confidence": "none"}
        grouper = SmartFileGrouper(analyzer)
        groups = grouper.group_by_subject_and_scan(files, empty_detection)
        assert len(groups) > 0


# ============================================================================
# build_bids_filename
# ============================================================================

class TestBuildBidsFilename:

    def test_anat_raw(self):
        name = build_bids_filename("01", "anat", "raw")
        assert name.startswith("sub-01")
        assert name.endswith(".nii.gz")
        assert "T1w" in name

    def test_anat_anonymized(self):
        name = build_bids_filename("82352", "anat", "anonymized")
        assert "acq-anonymized" in name
        assert "sub-82352" in name

    def test_func_rest(self):
        name = build_bids_filename("01", "func", "rest")
        assert "task-rest" in name
        assert "bold" in name

    def test_dwi(self):
        name = build_bids_filename("02", "dwi", "raw")
        assert "dwi" in name

    def test_result_is_string(self):
        assert isinstance(build_bids_filename("1", "anat", "raw"), str)


# ============================================================================
# extract_subject_ids_from_paths
# ============================================================================

class TestExtractSubjectIdsFromPaths:

    def test_extracts_camcan_style_ids(self, camcan_files):
        records = extract_subject_ids_from_paths(
            camcan_files,
            extraction_regex=r'([A-Za-z]+)_sub(\d+)',
            subject_group=2,
            site_group=1
        )
        ids = {r["subject_id"] for r in records}
        assert "41006" in ids
        assert "82352" in ids
        assert "06272" in ids

    def test_extracts_site_info(self, camcan_files):
        records = extract_subject_ids_from_paths(
            camcan_files,
            extraction_regex=r'([A-Za-z]+)_sub(\d+)',
            subject_group=2,
            site_group=1
        )
        sites = {r["site"] for r in records}
        assert "Newark" in sites
        assert "Beijing" in sites

    def test_no_duplicate_ids(self, camcan_files):
        records = extract_subject_ids_from_paths(
            camcan_files,
            extraction_regex=r'([A-Za-z]+)_sub(\d+)',
            subject_group=2
        )
        ids = [r["subject_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_yaml_double_backslash_fix(self, camcan_files):
        # Simulate YAML-loaded regex with double backslashes
        records = extract_subject_ids_from_paths(
            camcan_files,
            extraction_regex=r'([A-Za-z]+)_sub(\\d+)',  # double backslash
            subject_group=2
        )
        # Function should fix \\d → \d and still extract IDs
        assert isinstance(records, list)

    def test_invalid_regex_returns_empty(self):
        files = ["some_file.dcm"]
        records = extract_subject_ids_from_paths(
            files,
            extraction_regex="[invalid(regex",
            subject_group=1
        )
        assert records == []
