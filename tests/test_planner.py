# tests/test_planner.py
# Unit tests for autobidsify/converters/planner.py
# Tests cover ONLY the pure-Python functions — no LLM calls.
# Functions tested: _extract_subjects_from_directory_structure,
#                   _extract_subjects_from_flat_filenames,
#                   _generate_subject_id_mapping,
#                   _generate_participants_tsv_from_python,
#                   _merge_participants_with_metadata,
#                   _match_glob_pattern (via executor),
#                   _extract_numeric_id_from_identifier.

import pytest
from pathlib import Path

from autobidsify.converters.planner import (
    _extract_subjects_from_directory_structure,
    _extract_subjects_from_flat_filenames,
    _generate_subject_id_mapping,
    _generate_participants_tsv_from_python,
    _merge_participants_with_metadata,
    _extract_numeric_id_from_identifier,
)


# ============================================================================
# _extract_numeric_id_from_identifier
# ============================================================================

class TestExtractNumericId:

    def test_trailing_number(self):
        assert _extract_numeric_id_from_identifier("BZZ003") == "003"

    def test_preserves_leading_zeros(self):
        assert _extract_numeric_id_from_identifier("patient021") == "021"

    def test_sub_prefix(self):
        # sub-01 → '01'
        assert _extract_numeric_id_from_identifier("sub-01") == "01"

    def test_no_number_returns_none(self):
        assert _extract_numeric_id_from_identifier("healthy") is None

    def test_multiple_numbers_returns_last(self):
        # e.g. "Beijing_sub82352" → last numeric sequence
        result = _extract_numeric_id_from_identifier("Beijing_sub82352")
        assert result == "82352"


# ============================================================================
# _extract_subjects_from_directory_structure
# ============================================================================

class TestExtractSubjectsFromDirectory:

    @pytest.fixture
    def camcan_files(self):
        return [
            "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan.nii.gz",
            "Newark_sub41006/func_rest/NIfTI/scan_rest.nii.gz",
            "Beijing_sub82352/anat_mprage_anonymized/NIfTI/scan.nii.gz",
            "Cambridge_sub06272/anat_mprage_anonymized/NIfTI/scan.nii.gz",
        ]

    def test_detects_three_camcan_subjects(self, camcan_files):
        result = _extract_subjects_from_directory_structure(camcan_files)
        assert result["success"] is True
        assert result["subject_count"] == 3

    def test_detects_site_info(self, camcan_files):
        result = _extract_subjects_from_directory_structure(camcan_files)
        assert result["has_site_info"] is True
        sites = {r["site"] for r in result["subject_records"]}
        assert "Newark" in sites
        assert "Beijing" in sites

    def test_standard_bids_dirs(self):
        files = [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-02_T1w.nii.gz",
            "sub-10/anat/sub-10_T1w.nii.gz",
        ]
        result = _extract_subjects_from_directory_structure(files)
        assert result["success"] is True
        assert result["subject_count"] == 3

    def test_flat_structure_returns_failure(self):
        files = ["VHMCT1mm-Hip (1).dcm", "VHFCT1mm-Hip (1).dcm"]
        result = _extract_subjects_from_directory_structure(files)
        # Flat files have no subdirectory → should fail or return 0
        assert result["success"] is False or result["subject_count"] == 0

    def test_no_duplicate_subject_records(self, camcan_files):
        result = _extract_subjects_from_directory_structure(camcan_files)
        ids = [r["original_id"] for r in result["subject_records"]]
        assert len(ids) == len(set(ids))


# ============================================================================
# _extract_subjects_from_flat_filenames
# ============================================================================

class TestExtractSubjectsFromFlatFilenames:

    @pytest.fixture
    def vh_files(self):
        return [
            "VHMCT1mm-Hip (134).dcm",
            "VHMCT1mm-Hip (135).dcm",
            "VHMCT1mm-Head (256).dcm",
            "VHFCT1mm-Hip (45).dcm",
            "VHFCT1mm-Head (120).dcm",
        ]

    def test_detects_two_vh_subjects(self, vh_files):
        result = _extract_subjects_from_flat_filenames(vh_files, {})
        assert result["success"] is True
        # VHM and VHF are the two base identifiers
        ids = {r["original_id"] for r in result["subject_records"]}
        assert "VHM" in ids or any("VHM" in i for i in ids)
        assert "VHF" in ids or any("VHF" in i for i in ids)

    def test_file_counts_per_subject(self, vh_files):
        result = _extract_subjects_from_flat_filenames(vh_files, {})
        assert result["success"] is True
        total = sum(r["file_count"] for r in result["subject_records"])
        assert total == len(vh_files)

    def test_returns_success_false_for_empty(self):
        result = _extract_subjects_from_flat_filenames([], {})
        assert result["success"] is False


# ============================================================================
# _generate_subject_id_mapping
# ============================================================================

class TestGenerateSubjectIdMapping:

    @pytest.fixture
    def camcan_subject_info(self):
        return {
            "success": True,
            "subject_records": [
                {"original_id": "Newark_sub41006", "numeric_id": "41006", "site": "Newark"},
                {"original_id": "Beijing_sub82352", "numeric_id": "82352", "site": "Beijing"},
                {"original_id": "Cambridge_sub06272", "numeric_id": "06272", "site": "Cambridge"},
            ],
            "has_site_info": True,
        }

    @pytest.fixture
    def bids_subject_info(self):
        return {
            "success": True,
            "subject_records": [
                {"original_id": "sub-01", "numeric_id": "01", "site": None},
                {"original_id": "sub-02", "numeric_id": "02", "site": None},
                {"original_id": "sub-10", "numeric_id": "10", "site": None},
            ],
            "has_site_info": False,
        }

    # ---- already_bids strategy ----

    def test_already_bids_strips_sub_prefix(self, bids_subject_info):
        info = _generate_subject_id_mapping(bids_subject_info, {}, "auto")
        assert info["strategy_used"] == "already_bids"
        mapping = info["id_mapping"]
        assert mapping["sub-01"] == "01"
        assert mapping["sub-10"] == "10"

    def test_already_bids_preserves_leading_zeros(self, bids_subject_info):
        info = _generate_subject_id_mapping(bids_subject_info, {}, "auto")
        assert info["id_mapping"]["sub-01"] == "01"  # NOT "1"

    # ---- numeric strategy ----

    def test_numeric_strategy_extracts_numbers(self, camcan_subject_info):
        info = _generate_subject_id_mapping(
            camcan_subject_info, {}, "numeric"
        )
        assert info["strategy_used"] == "numeric"
        mapping = info["id_mapping"]
        # Each original_id maps to a numeric string
        for orig, bids in mapping.items():
            assert bids.isdigit() or bids.replace("0", "").isdigit()

    def test_numeric_strategy_includes_original_id_column(self, camcan_subject_info):
        info = _generate_subject_id_mapping(
            camcan_subject_info, {}, "numeric"
        )
        assert "original_id" in info["metadata_columns"]

    # ---- semantic strategy ----

    def test_semantic_strategy_removes_special_chars(self, camcan_subject_info):
        info = _generate_subject_id_mapping(
            camcan_subject_info, {}, "semantic"
        )
        assert info["strategy_used"] == "semantic"
        for bids_id in info["id_mapping"].values():
            assert "_" not in bids_id
            assert "-" not in bids_id

    # ---- empty input ----

    def test_empty_records_returns_empty_mapping(self):
        empty_info = {"subject_records": [], "has_site_info": False}
        info = _generate_subject_id_mapping(empty_info, {})
        assert info["id_mapping"] == {}

    # ---- reverse mapping ----

    def test_reverse_mapping_is_inverse(self, bids_subject_info):
        info = _generate_subject_id_mapping(bids_subject_info, {}, "auto")
        for orig, bids in info["id_mapping"].items():
            assert info["reverse_mapping"][bids] == orig


# ============================================================================
# _generate_participants_tsv_from_python
# ============================================================================

class TestGenerateParticipantsTsv:

    @pytest.fixture
    def subject_info_two(self):
        return {
            "subject_records": [
                {"original_id": "VHM", "numeric_id": "1", "site": None},
                {"original_id": "VHF", "numeric_id": "2", "site": None},
            ]
        }

    def test_creates_participants_tsv(self, tmp_path, subject_info_two):
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=[]
        )
        path = tmp_path / "participants.tsv"
        assert path.exists()

    def test_header_contains_participant_id(self, tmp_path, subject_info_two):
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=[]
        )
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert lines[0] == "participant_id"

    def test_correct_number_of_rows(self, tmp_path, subject_info_two):
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=[]
        )
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        # 1 header + 2 subjects
        assert len(lines) == 3

    def test_subject_ids_have_sub_prefix(self, tmp_path, subject_info_two):
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=[]
        )
        content = (tmp_path / "participants.tsv").read_text()
        assert "sub-1" in content
        assert "sub-2" in content

    def test_includes_original_id_column(self, tmp_path, subject_info_two):
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=["original_id"]
        )
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert "original_id" in lines[0]

    def test_overwrites_existing_file(self, tmp_path, subject_info_two):
        # Write stale file first
        (tmp_path / "participants.tsv").write_text("stale content\n")
        _generate_participants_tsv_from_python(
            subject_info_two, tmp_path,
            id_mapping={"VHM": "1", "VHF": "2"},
            metadata_columns=[]
        )
        content = (tmp_path / "participants.tsv").read_text()
        assert "stale content" not in content
        assert "participant_id" in content


# ============================================================================
# _merge_participants_with_metadata
# ============================================================================

class TestMergeParticipantsWithMetadata:

    @pytest.fixture
    def subject_info(self):
        return {
            "subject_records": [
                {"original_id": "VHM", "numeric_id": "1", "site": None},
                {"original_id": "VHF", "numeric_id": "2", "site": None},
            ]
        }

    @pytest.fixture
    def id_mapping_info(self):
        return {
            "id_mapping": {"VHM": "1", "VHF": "2"},
            "metadata_columns": [],
            "strategy_used": "numeric"
        }

    def test_adds_extra_columns_from_llm(self, tmp_path, subject_info, id_mapping_info):
        # Write base participants.tsv first
        (tmp_path / "participants.tsv").write_text(
            "participant_id\nsub-1\nsub-2\n"
        )
        plan = {
            "participant_metadata": {
                "1": {"sex": "M", "age": "38"},
                "2": {"sex": "F", "age": "42"},
            }
        }
        _merge_participants_with_metadata(
            plan, tmp_path, subject_info, id_mapping_info
        )
        content = (tmp_path / "participants.tsv").read_text()
        assert "sex" in content
        assert "age" in content
        assert "M" in content
        assert "F" in content

    def test_missing_metadata_fills_na(self, tmp_path, subject_info, id_mapping_info):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\nsub-1\nsub-2\n"
        )
        # Only subject "1" has metadata; subject "2" is missing
        plan = {
            "participant_metadata": {
                "1": {"sex": "M"},
            }
        }
        _merge_participants_with_metadata(
            plan, tmp_path, subject_info, id_mapping_info
        )
        content = (tmp_path / "participants.tsv").read_text()
        assert "n/a" in content

    def test_no_metadata_no_change(self, tmp_path, subject_info, id_mapping_info):
        original = "participant_id\nsub-1\nsub-2\n"
        (tmp_path / "participants.tsv").write_text(original)
        plan = {}  # No participant_metadata key
        _merge_participants_with_metadata(
            plan, tmp_path, subject_info, id_mapping_info
        )
        # File should be unchanged
        assert (tmp_path / "participants.tsv").read_text() == original
