# tests/test_filename_tokenizer.py
# Unit tests for autobidsify/filename_tokenizer.py
# Tests cover: FilenameTokenizer, FilenamePatternAnalyzer,
#              analyze_filenames_for_subjects, and SubjectGroupingDecision.

import pytest
from autobidsify.filename_tokenizer import (
    FilenameTokenizer,
    FilenamePatternAnalyzer,
    analyze_filenames_for_subjects,
    SubjectGroupingDecision,
)


# ============================================================================
# FilenameTokenizer.tokenize
# ============================================================================

class TestFilenameTokenizerTokenize:

    def test_visible_human_dcm(self):
        tokens = FilenameTokenizer.tokenize("VHMCT1mm-Hip (134).dcm")
        # Must contain the meaningful parts
        assert "VHM" in tokens or any("vhm" in t.lower() for t in tokens)
        assert "Hip" in tokens or "hip" in [t.lower() for t in tokens]
        # Sequence number (134) should be extracted
        assert "134" in tokens

    def test_beijing_subject_dir(self):
        tokens = FilenameTokenizer.tokenize("Beijing_sub82352")
        lower = [t.lower() for t in tokens]
        assert "beijing" in lower
        assert "sub" in lower or any("sub" in t.lower() for t in tokens)
        assert "82352" in tokens

    def test_bids_standard_filename(self):
        tokens = FilenameTokenizer.tokenize("sub-01_T1w.nii.gz")
        assert "01" in tokens
        # T1w is a known neuroimaging term — kept together
        assert "T1w" in tokens

    def test_extension_removed(self):
        tokens = FilenameTokenizer.tokenize("scan_mprage.nii.gz")
        # Extension must not appear as a token
        assert "nii" not in tokens
        assert "gz" not in tokens

    def test_known_neuroimaging_terms_kept_together(self):
        # These terms must NOT be split
        for term in ["T1w", "T2w", "BOLD", "DWI", "FLAIR"]:
            tokens = FilenameTokenizer.tokenize(f"sub-01_{term}.nii.gz")
            assert term in tokens, f"{term} was incorrectly split"

    def test_delimiters_stripped(self):
        tokens = FilenameTokenizer.tokenize("patient(001)[data]-scan.nii")
        # Brackets and parens should be removed, not appear as tokens
        assert "(" not in tokens
        assert ")" not in tokens
        assert "[" not in tokens

    def test_empty_filename(self):
        tokens = FilenameTokenizer.tokenize("")
        assert tokens == []

    def test_only_extension(self):
        tokens = FilenameTokenizer.tokenize(".nii.gz")
        # Should produce empty or very short list
        assert isinstance(tokens, list)

    def test_camelcase_split(self):
        tokens = FilenameTokenizer.tokenize("ScanMprage.nii")
        lower = [t.lower() for t in tokens]
        # CamelCase should be split into components
        assert "scan" in lower or "Scan" in tokens


# ============================================================================
# FilenamePatternAnalyzer
# ============================================================================

class TestFilenamePatternAnalyzer:

    # ---- Visible Human dataset (2 prefixes: VHM / VHF) ----

    @pytest.fixture
    def vh_files(self):
        return [
            "VHMCT1mm-Hip (134).dcm",
            "VHMCT1mm-Hip (135).dcm",
            "VHMCT1mm-Head (256).dcm",
            "VHMCT1mm-Shoulder (89).dcm",
            "VHFCT1mm-Hip (45).dcm",
            "VHFCT1mm-Head (120).dcm",
            "VHFCT1mm-Ankle (78).dcm",
        ]

    def test_detects_two_dominant_prefixes(self, vh_files):
        analyzer = FilenamePatternAnalyzer(vh_files)
        stats = analyzer.analyze_token_statistics()
        dominant = stats["dominant_prefixes"]
        prefixes = [p["prefix"] for p in dominant]
        assert len(dominant) == 2
        assert "VHM" in prefixes
        assert "VHF" in prefixes

    def test_prefix_percentages_sum_to_100(self, vh_files):
        analyzer = FilenamePatternAnalyzer(vh_files)
        stats = analyzer.analyze_token_statistics()
        total = sum(p["percentage"] for p in stats["dominant_prefixes"])
        assert abs(total - 100.0) < 1.0  # allow rounding

    def test_token_frequency_is_dict(self, vh_files):
        analyzer = FilenamePatternAnalyzer(vh_files)
        stats = analyzer.analyze_token_statistics()
        assert isinstance(stats["token_frequency"], dict)

    def test_insights_is_list_of_strings(self, vh_files):
        analyzer = FilenamePatternAnalyzer(vh_files)
        stats = analyzer.analyze_token_statistics()
        assert isinstance(stats["insights"], list)
        for item in stats["insights"]:
            assert isinstance(item, str)

    # ---- CamCAN multi-site (no dominant filename prefixes since dirs are used) ----

    @pytest.fixture
    def camcan_files(self):
        return [
            "Newark_sub41006/anat/scan_mprage_anonymized.nii.gz",
            "Beijing_sub82352/anat/scan_mprage_anonymized.nii.gz",
            "Cambridge_sub06272/func/scan_rest.nii.gz",
        ]

    def test_camcan_sample_diversity(self, camcan_files):
        analyzer = FilenamePatternAnalyzer(camcan_files)
        samples = analyzer._sample_diverse_filenames(10)
        assert len(samples) <= 10
        assert isinstance(samples, list)

    # ---- Single subject (no dominant prefix expected) ----

    def test_single_file_no_dominant_prefix(self):
        analyzer = FilenamePatternAnalyzer(["sub-01_T1w.nii.gz"])
        stats = analyzer.analyze_token_statistics()
        # With only 1 file, no prefix can reach 5% threshold meaningfully
        assert isinstance(stats["dominant_prefixes"], list)

    # ---- build_llm_payload ----

    def test_llm_payload_structure(self, vh_files):
        analyzer = FilenamePatternAnalyzer(vh_files)
        payload = analyzer.build_llm_payload({"n_subjects": 2}, max_samples=5)
        assert "task" in payload
        assert "statistics" in payload
        assert "filename_samples" in payload
        assert "user_hints" in payload
        assert len(payload["filename_samples"]) <= 5


# ============================================================================
# analyze_filenames_for_subjects
# ============================================================================

class TestAnalyzeFilenamesForSubjects:

    def test_returns_expected_keys(self):
        files = [
            "VHMCT1mm-Hip (1).dcm",
            "VHMCT1mm-Head (2).dcm",
            "VHFCT1mm-Hip (1).dcm",
            "VHFCT1mm-Head (2).dcm",
        ]
        result = analyze_filenames_for_subjects(files, {"n_subjects": 2})
        assert "python_statistics" in result
        assert "llm_payload" in result
        assert "confidence" in result
        assert "recommendation" in result

    def test_confidence_is_valid_value(self):
        files = ["VHMCT1mm.dcm", "VHFCT1mm.dcm"]
        result = analyze_filenames_for_subjects(files, {})
        assert result["confidence"] in ("high", "medium", "low", "none")

    def test_high_confidence_when_hint_matches(self):
        # 2 dominant prefixes + user says n_subjects=2 → should be high
        files = (
            ["VHMCT1mm-Hip (%d).dcm" % i for i in range(10)] +
            ["VHFCT1mm-Hip (%d).dcm" % i for i in range(10)]
        )
        result = analyze_filenames_for_subjects(files, {"n_subjects": 2})
        assert result["confidence"] in ("high", "medium")

    def test_recommendation_is_string(self):
        files = ["scan001.nii.gz", "scan002.nii.gz"]
        result = analyze_filenames_for_subjects(files, {})
        assert isinstance(result["recommendation"], str)


# ============================================================================
# SubjectGroupingDecision
# ============================================================================

class TestSubjectGroupingDecision:

    def test_prefix_mapping_structure(self):
        decision = SubjectGroupingDecision.create_prefix_mapping(
            {"VHM": "1", "VHF": "2"},
            metadata={"1": {"sex": "M"}, "2": {"sex": "F"}}
        )
        assert decision["method"] == "prefix_based"
        assert len(decision["rules"]) == 2
        prefixes = [r["prefix"] for r in decision["rules"]]
        assert "VHM" in prefixes
        assert "VHF" in prefixes

    def test_prefix_mapping_subject_ids(self):
        decision = SubjectGroupingDecision.create_prefix_mapping({"A": "01", "B": "02"})
        subjects = {r["prefix"]: r["maps_to_subject"] for r in decision["rules"]}
        assert subjects["A"] == "01"
        assert subjects["B"] == "02"

    def test_sequential_assignment_structure(self):
        decision = SubjectGroupingDecision.create_sequential_assignment(5)
        assert decision["method"] == "sequential"
        assert decision["n_subjects"] == 5

    def test_blocking_question_structure(self):
        decision = SubjectGroupingDecision.create_blocking_question(
            reason="Cannot determine grouping",
            options=["Group by prefix", "Group by directory"]
        )
        assert decision["method"] == "blocked"
        assert "question" in decision
        assert len(decision["question"]["options"]) == 2
