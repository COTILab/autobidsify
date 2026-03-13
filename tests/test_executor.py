# tests/test_executor.py
# Unit tests for autobidsify/converters/executor.py
# Tests cover ONLY pure-Python logic — no file I/O, no LLM, no dcm2niix.
# Functions tested: _match_glob_pattern, analyze_filepath_universal,
#                   infer_scan_type_from_filepath, _normalize_filename,
#                   _select_preferred_file, infer_subdirectory_from_suffix,
#                   categorize_scan_type.

import pytest
from autobidsify.converters.executor import (
    _match_glob_pattern,
    analyze_filepath_universal,
    infer_scan_type_from_filepath,
    infer_subdirectory_from_suffix,
    categorize_scan_type,
    _normalize_filename,
    _select_preferred_file,
)


# ============================================================================
# _match_glob_pattern
# ============================================================================

class TestMatchGlobPattern:

    # ---- **/*.ext — match any file at any depth ----

    def test_double_star_extension_match(self):
        assert _match_glob_pattern("sub-01/anat/scan.nii.gz", "**/*.nii.gz")

    def test_double_star_extension_no_match_wrong_ext(self):
        assert not _match_glob_pattern("sub-01/anat/scan.dcm", "**/*.nii.gz")

    def test_double_star_extension_flat_file(self):
        assert _match_glob_pattern("scan.nii.gz", "**/*.nii.gz")

    # ---- **/TOKEN/** — match directory component ----

    def test_token_dir_match(self):
        assert _match_glob_pattern(
            "Newark_sub41006/anat/BRIK/scan.nii.gz", "**/BRIK/**"
        )

    def test_token_dir_no_match(self):
        assert not _match_glob_pattern(
            "Newark_sub41006/anat/NIfTI/scan.nii.gz", "**/BRIK/**"
        )

    def test_token_dir_case_insensitive(self):
        assert _match_glob_pattern(
            "sub-01/anat/brik/scan.nii.gz", "**/BRIK/**"
        )

    # ---- *token* — substring anywhere in path ----

    def test_star_token_star_match(self):
        assert _match_glob_pattern("VHMCT1mm-Hip (134).dcm", "*VHM*")

    def test_star_token_star_no_match(self):
        assert not _match_glob_pattern("VHFCT1mm-Hip (45).dcm", "*VHM*")

    # ---- *.ext — extension match on filename only ----

    def test_simple_extension_match(self):
        assert _match_glob_pattern("data/scan.dcm", "*.dcm")

    def test_simple_extension_no_match(self):
        assert not _match_glob_pattern("data/scan.nii.gz", "*.dcm")

    # ---- token* — filename prefix ----

    def test_prefix_match(self):
        assert _match_glob_pattern("VHM_scan.dcm", "VHM*")

    def test_prefix_no_match(self):
        assert not _match_glob_pattern("VHF_scan.dcm", "VHM*")

    # ---- fallback: substring ----

    def test_fallback_substring_match(self):
        assert _match_glob_pattern("some/path/sub-01/T1w.nii.gz", "sub-01")

    def test_fallback_substring_no_match(self):
        assert not _match_glob_pattern("some/path/sub-02/T1w.nii.gz", "sub-01")


# ============================================================================
# _normalize_filename
# ============================================================================

class TestNormalizeFilename:

    def test_strips_sequence_parens(self):
        assert _normalize_filename("VHFCT1mm-Hip (134).dcm") == "vhfct1mm-hip"

    def test_strips_trailing_numeric_suffix(self):
        result = _normalize_filename("scan_001.dcm")
        assert "001" not in result

    def test_preserves_base_name(self):
        result = _normalize_filename("scan_mprage_anonymized.nii.gz")
        assert result == "scan_mprage_anonymized"

    def test_returns_lowercase(self):
        result = _normalize_filename("ScanMprage.nii.gz")
        assert result == result.lower()

    def test_handles_nested_path(self):
        # Only the filename part should be normalized
        result = _normalize_filename("sub-01/anat/scan.nii.gz")
        assert "/" not in result


# ============================================================================
# _select_preferred_file
# ============================================================================

class TestSelectPreferredFile:

    def test_single_file_returned_as_is(self):
        assert _select_preferred_file(["sub/NIfTI/scan.nii.gz"]) == "sub/NIfTI/scan.nii.gz"

    def test_nifti_preferred_over_brik(self):
        files = [
            "sub/BRIK/scan.nii.gz",
            "sub/NIfTI/scan.nii.gz",
        ]
        result = _select_preferred_file(files)
        assert "NIfTI" in result
        assert "BRIK" not in result

    def test_non_brik_preferred_over_brik(self):
        files = [
            "sub/BRIK/scan.nii.gz",
            "sub/other/scan.nii.gz",
        ]
        result = _select_preferred_file(files)
        assert "BRIK" not in result

    def test_empty_list_returns_none(self):
        assert _select_preferred_file([]) is None

    def test_shorter_path_preferred(self):
        files = [
            "sub/anat/extra/deep/scan.nii.gz",
            "sub/anat/scan.nii.gz",
        ]
        result = _select_preferred_file(files)
        assert result == "sub/anat/scan.nii.gz"


# ============================================================================
# infer_scan_type_from_filepath
# ============================================================================

class TestInferScanType:

    def test_t1w_from_path_keyword(self):
        result = infer_scan_type_from_filepath(
            "sub-01/anat/scan.nii.gz", []
        )
        assert result["suffix"] in ("T1w", "anat") or "T1w" in result["suffix"]
        assert result["subdirectory"] == "anat"

    def test_bold_from_func_keyword(self):
        result = infer_scan_type_from_filepath(
            "sub-01/func/scan_rest.nii.gz", []
        )
        assert result["subdirectory"] == "func"

    def test_nirs_from_snirf_extension(self):
        result = infer_scan_type_from_filepath("sub-01/nirs/scan.snirf", [])
        assert result["subdirectory"] == "nirs"
        assert result["suffix"] == "nirs"

    def test_extracts_task_from_filename(self):
        result = infer_scan_type_from_filepath(
            "sub-01_task-FRESHMOTOR_nirs.snirf", []
        )
        assert "FRESHMOTOR" in result["suffix"] or "freshmotor" in result["suffix"].lower()

    def test_extracts_session_from_filename(self):
        result = infer_scan_type_from_filepath(
            "sub-01_ses-left2s_task-MOTOR_nirs.snirf", []
        )
        assert "ses-left2s" in result["suffix"]

    def test_no_spurious_ses_injection_without_dir(self):
        # If filepath has no ses-XX/ directory, ses- should NOT appear
        result = infer_scan_type_from_filepath(
            "sub-01/anat/scan_T1w.nii.gz", []
        )
        assert "ses-X" not in result["suffix"]
        assert "ses-Head" not in result["suffix"]

    def test_llm_rule_takes_priority(self):
        rules = [
            {
                "match_pattern": ".*anonymized.*",
                "bids_template": "sub-X_T1w.nii.gz"
            }
        ]
        result = infer_scan_type_from_filepath(
            "scan_mprage_anonymized.nii.gz", rules
        )
        assert "T1w" in result["suffix"]

    def test_placeholder_x_stripped_from_template(self):
        rules = [
            {
                "match_pattern": ".*",
                "bids_template": "sub-X_ses-X_T1w.nii.gz"
            }
        ]
        result = infer_scan_type_from_filepath("scan.nii.gz", rules)
        # ses-X placeholder must be removed from the suffix
        assert "ses-X" not in result["suffix"]


# ============================================================================
# infer_subdirectory_from_suffix
# ============================================================================

class TestInferSubdirectory:

    @pytest.mark.parametrize("suffix, expected", [
        ("T1w",             "anat"),
        ("T2w",             "anat"),
        ("task-rest_bold",  "func"),
        ("bold",            "func"),
        ("nirs",            "nirs"),
        ("task-motor_nirs", "nirs"),
        ("dwi",             "dwi"),
        ("unknown_scan",    "anat"),  # fallback
    ])
    def test_subdirectory_inference(self, suffix, expected):
        assert infer_subdirectory_from_suffix(suffix) == expected


# ============================================================================
# categorize_scan_type
# ============================================================================

class TestCategorizeScanType:

    @pytest.mark.parametrize("suffix, expected", [
        ("T1w",            "anatomical"),
        ("T2w",            "anatomical"),
        ("task-rest_bold", "functional"),
        ("nirs",           "functional"),
        ("dwi",            "diffusion"),
        ("unknown",        "unknown"),
    ])
    def test_categorization(self, suffix, expected):
        assert categorize_scan_type(suffix) == expected


# ============================================================================
# analyze_filepath_universal
# ============================================================================

class TestAnalyzeFilepathUniversal:

    @pytest.fixture
    def assignment_rules(self):
        return [
            {"subject": "1", "original": "VHM", "match": ["*VHM*"]},
            {"subject": "2", "original": "VHF", "match": ["*VHF*"]},
        ]

    def test_assigns_subject_by_match_pattern(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip (134).dcm", assignment_rules, [], modality="mri"
        )
        assert result["subject_id"] == "1"

    def test_assigns_vhf_correctly(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHFCT1mm-Head (120).dcm", assignment_rules, [], modality="mri"
        )
        assert result["subject_id"] == "2"

    def test_no_sub_prefix_in_subject_id(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip.dcm", assignment_rules, [], modality="mri"
        )
        # subject_id must be bare — not "sub-1"
        assert not result["subject_id"].startswith("sub-")

    def test_mri_bids_filename_has_nii_gz_extension(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip.dcm", assignment_rules, [], modality="mri"
        )
        assert result["bids_filename"].endswith(".nii.gz")

    def test_nirs_bids_filename_has_snirf_extension(self, assignment_rules):
        nirs_rules = [
            {"subject": "1", "original": "sub-01", "match": ["*sub-01*"]}
        ]
        result = analyze_filepath_universal(
            "sub-01_task-rest_nirs.snirf", nirs_rules, [], modality="nirs"
        )
        assert result["bids_filename"].endswith(".snirf")

    def test_fallback_to_original_field(self):
        rules = [{"subject": "3", "original": "BZZ021", "match": []}]
        result = analyze_filepath_universal(
            "BZZ021_scan.nii.gz", rules, [], modality="mri"
        )
        assert result["subject_id"] == "3"

    def test_unknown_subject_when_no_match(self):
        result = analyze_filepath_universal(
            "completely_unmatched_file.nii.gz", [], [], modality="mri"
        )
        assert result["subject_id"] == "unknown"

    def test_bids_standard_fallback(self):
        # Standard BIDS path: sub-07 extracted from directory
        result = analyze_filepath_universal(
            "sub-07/anat/sub-07_T1w.nii.gz", [], [], modality="mri"
        )
        assert result["subject_id"] == "07"
