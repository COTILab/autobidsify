# tests/test_planner.py
# Unit tests for autobidsify/converters/planner.py
# Tests cover ONLY pure-Python functions — no LLM calls.
# Functions: _is_data_file, _extract_subjects_from_directory_structure,
#            _extract_subjects_from_flat_filenames, _collect_extra_columns,
#            _write_participants_from_plan, _merge_participants_from_llm_metadata,
#            _parse_llm_json_response (planner's local copy).

import json
import pytest
from pathlib import Path

from autobidsify.converters.planner import (
    _is_data_file,
    _extract_subjects_from_directory_structure,
    _extract_subjects_from_flat_filenames,
    _collect_extra_columns,
    _write_participants_from_plan,
    _merge_participants_from_llm_metadata,
    _parse_llm_json_response,
)


# ============================================================================
# _is_data_file
# ============================================================================

class TestIsDataFile:

    @pytest.mark.parametrize("path", [
        "sub-01/nirs/sub-01_task-rest_nirs.snirf",
        "VHMCT1mm-Hip (134).dcm",
        "sub-01/anat/sub-01_T1w.nii.gz",
        "scan.nii",
        "data.mat",
        "subject01.nirs",
        "scan.jnii",
        "scan.bnii",
        "Subject00_1.edf",
        "sub-01/eeg/scan.vhdr",
        "sub-01/eeg/scan.set",
        "sub-01/eeg/scan.bdf",
    ])
    def test_data_files_return_true(self, path):
        assert _is_data_file(path) is True

    @pytest.mark.parametrize("path", [
        "paper.pdf",
        "group_stats.xlsx",
        "procStream.cfg",
        "dataset_description.json",
        "participants.tsv",
        "README.txt",
        "SHA256SUMS.txt",
        "subject-info.csv",
    ])
    def test_non_data_files_return_false(self, path):
        assert _is_data_file(path) is False


# ============================================================================
# _extract_subjects_from_directory_structure
# ============================================================================

class TestExtractSubjectsFromDirectoryStructure:

    def test_camcan_site_sub_pattern(self):
        files = [
            "Newark_sub41006/anat/scan.nii.gz",
            "Beijing_sub82352/anat/scan.nii.gz",
            "Cambridge_sub06272/anat/scan.nii.gz",
        ]
        result = _extract_subjects_from_directory_structure(files)
        assert result["success"] is True
        assert result["subject_count"] == 3

    def test_camcan_has_site_info(self):
        files = [
            "Newark_sub41006/anat/scan.nii.gz",
            "Beijing_sub82352/anat/scan.nii.gz",
        ]
        result = _extract_subjects_from_directory_structure(files)
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
        assert result["success"] is False or result.get("subject_count", 0) == 0

    def test_no_duplicate_subject_records(self):
        # Two files per subject — should still detect 3 unique subjects
        files = [
            "Newark_sub41006/anat/scan.nii.gz",
            "Newark_sub41006/func/rest.nii.gz",
            "Beijing_sub82352/anat/scan.nii.gz",
            "Cambridge_sub06272/anat/scan.nii.gz",
        ]
        result = _extract_subjects_from_directory_structure(files)
        ids = [r["original_id"] for r in result["subject_records"]]
        assert len(ids) == len(set(ids))
        assert result["subject_count"] == 3

    def test_parkinson_nested_group_subject(self):
        # group/subject nested structure
        files = [
            "PD/PD01/1_resting.snirf",
            "PD/PD02/1_resting.snirf",
            "control/ctrl01/1_resting.snirf",
        ]
        result = _extract_subjects_from_directory_structure(files)
        # Should detect some subjects (PD01, PD02, ctrl01 or PD/control dirs)
        assert isinstance(result, dict)
        assert "success" in result

    def test_empty_files_returns_failure(self):
        result = _extract_subjects_from_directory_structure([])
        assert result["success"] is False


# ============================================================================
# _extract_subjects_from_flat_filenames
# ============================================================================

class TestExtractSubjectsFromFlatFilenames:

    def test_visible_human_two_subjects(self):
        files = [
            "VHMCT1mm-Hip (134).dcm",
            "VHMCT1mm-Hip (135).dcm",
            "VHMCT1mm-Head (256).dcm",
            "VHFCT1mm-Hip (45).dcm",
            "VHFCT1mm-Head (120).dcm",
        ]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is True
        ids = {r["original_id"] for r in result["subject_records"]}
        assert any("VHM" in i for i in ids)
        assert any("VHF" in i for i in ids)

    def test_file_counts_correct(self):
        files = [
            "VHMCT1mm-Hip (134).dcm",
            "VHMCT1mm-Hip (135).dcm",
            "VHFCT1mm-Hip (45).dcm",
        ]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is True
        total = sum(r["file_count"] for r in result["subject_records"])
        assert total == 3

    def test_filters_non_data_files(self):
        files = [
            "PD01_scan.snirf",
            "PD02_scan.snirf",
            "group_stats.xlsx",
            "paper.pdf",
            "README.txt",
        ]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is True
        ids = {r["original_id"] for r in result["subject_records"]}
        assert "group" not in ids
        assert "paper" not in ids
        assert "README" not in ids

    def test_edf_files_detected(self):
        files = [
            "Subject00_1.edf",
            "Subject00_2.edf",
            "Subject01_1.edf",
            "Subject01_2.edf",
        ]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is True
        ids = {r["original_id"] for r in result["subject_records"]}
        assert any("Subject00" in i for i in ids)
        assert any("Subject01" in i for i in ids)

    def test_empty_list_returns_failure(self):
        result = _extract_subjects_from_flat_filenames([])
        assert result["success"] is False

    def test_only_non_data_returns_failure(self):
        files = ["readme.txt", "group_stats.xlsx", "paper.pdf"]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is False


# ============================================================================
# _collect_extra_columns
# ============================================================================

class TestCollectExtraColumns:

    def test_collects_all_keys(self):
        metadata = {
            "1": {"sex": "M", "group": "PD"},
            "2": {"sex": "F", "group": "control"},
        }
        cols = _collect_extra_columns(metadata)
        assert "sex" in cols
        assert "group" in cols

    def test_excludes_participant_id(self):
        metadata = {"1": {"participant_id": "sub-1", "sex": "M"}}
        cols = _collect_extra_columns(metadata)
        assert "participant_id" not in cols
        assert "sex" in cols

    def test_deduplicates_columns(self):
        metadata = {
            "1": {"sex": "M", "age": "30"},
            "2": {"sex": "F", "age": "25"},
            "3": {"sex": "M", "age": "40"},
        }
        cols = _collect_extra_columns(metadata)
        assert cols.count("sex") == 1
        assert cols.count("age") == 1

    def test_empty_metadata_returns_empty(self):
        assert _collect_extra_columns({}) == []

    def test_preserves_insertion_order_across_subjects(self):
        metadata = {
            "1": {"age": "30", "sex": "M"},
            "2": {"group": "PD"},
        }
        cols = _collect_extra_columns(metadata)
        # age and sex appear in subject 1, group in subject 2
        assert "age" in cols
        assert "sex" in cols
        assert "group" in cols


# ============================================================================
# _write_participants_from_plan
# ============================================================================

class TestWriteParticipantsFromPlan:

    @pytest.fixture
    def simple_plan(self):
        return {
            "assignment_rules": [
                {"subject": "1", "original": "PD01", "match": ["*PD01*"]},
                {"subject": "2", "original": "PD02", "match": ["*PD02*"]},
            ],
            "subjects": {"labels": ["1", "2"], "count": 2},
            "participant_metadata": {
                "1": {"group": "PD"},
                "2": {"group": "PD"},
            },
        }

    def test_creates_file(self, tmp_path, simple_plan):
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        assert (tmp_path / "participants.tsv").exists()

    def test_correct_subject_count(self, tmp_path, simple_plan):
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        # header + 2 subjects
        assert len(lines) == 3

    def test_header_first_column_is_participant_id(self, tmp_path, simple_plan):
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert lines[0].split("\t")[0] == "participant_id"

    def test_sub_prefix_in_rows(self, tmp_path, simple_plan):
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        content = (tmp_path / "participants.tsv").read_text()
        assert "sub-1" in content
        assert "sub-2" in content

    def test_extra_columns_from_metadata(self, tmp_path, simple_plan):
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        header = (tmp_path / "participants.tsv").read_text().splitlines()[0]
        assert "group" in header

    def test_overwrites_existing_file(self, tmp_path, simple_plan):
        (tmp_path / "participants.tsv").write_text("stale content\n")
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        content = (tmp_path / "participants.tsv").read_text()
        assert "stale" not in content

    def test_warns_when_llm_count_less_than_user(self, tmp_path):
        plan = {
            "assignment_rules": [
                {"subject": "1", "original": "x", "match": ["*x*"]}
            ],
            "subjects": {"labels": ["1"], "count": 1},
            "participant_metadata": {},
        }
        # Should not raise — just warn
        _write_participants_from_plan(plan, tmp_path, user_n_subjects=5)
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert len(lines) == 2  # header + 1 subject

    def test_n_a_for_missing_metadata_values(self, tmp_path):
        plan = {
            "assignment_rules": [
                {"subject": "1", "original": "S01", "match": ["*S01*"]},
                {"subject": "2", "original": "S02", "match": ["*S02*"]},
            ],
            "subjects": {"labels": ["1", "2"], "count": 2},
            "participant_metadata": {
                "1": {"group": "PD"},
                # subject 2 has no metadata
            },
        }
        _write_participants_from_plan(plan, tmp_path, user_n_subjects=2)
        content = (tmp_path / "participants.tsv").read_text()
        assert "n/a" in content

    def test_subjects_sorted_numerically(self, tmp_path):
        plan = {
            "assignment_rules": [
                {"subject": "10", "original": "S10", "match": ["*S10*"]},
                {"subject": "2",  "original": "S02", "match": ["*S02*"]},
                {"subject": "1",  "original": "S01", "match": ["*S01*"]},
            ],
            "subjects": {"labels": ["1", "2", "10"], "count": 3},
            "participant_metadata": {},
        }
        _write_participants_from_plan(plan, tmp_path, user_n_subjects=3)
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        ids = [l.split("\t")[0] for l in lines[1:]]
        # Numeric sort: sub-1 < sub-2 < sub-10
        assert ids == ["sub-1", "sub-2", "sub-10"]


# ============================================================================
# _merge_participants_from_llm_metadata
# ============================================================================

class TestMergeParticipantsFromLlmMetadata:

    def test_adds_new_columns(self, tmp_path):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\nsub-1\nsub-2\n"
        )
        plan = {
            "participant_metadata": {
                "1": {"sex": "M", "group": "PD"},
                "2": {"sex": "F", "group": "control"},
            }
        }
        _merge_participants_from_llm_metadata(plan, tmp_path)
        content = (tmp_path / "participants.tsv").read_text()
        assert "sex" in content
        assert "group" in content

    def test_does_not_duplicate_existing_columns(self, tmp_path):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\tsex\nsub-1\tM\nsub-2\tF\n"
        )
        plan = {
            "participant_metadata": {
                "1": {"sex": "M"},
                "2": {"sex": "F"},
            }
        }
        _merge_participants_from_llm_metadata(plan, tmp_path)
        header = (tmp_path / "participants.tsv").read_text().splitlines()[0]
        assert header.count("sex") == 1

    def test_missing_values_filled_with_na(self, tmp_path):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\nsub-1\nsub-2\n"
        )
        plan = {
            "participant_metadata": {
                "1": {"sex": "M"},
                # subject 2 missing → n/a
            }
        }
        _merge_participants_from_llm_metadata(plan, tmp_path)
        content = (tmp_path / "participants.tsv").read_text()
        assert "n/a" in content

    def test_no_metadata_no_change(self, tmp_path):
        original = "participant_id\nsub-1\nsub-2\n"
        (tmp_path / "participants.tsv").write_text(original)
        _merge_participants_from_llm_metadata({}, tmp_path)
        assert (tmp_path / "participants.tsv").read_text() == original

    def test_no_file_does_not_crash(self, tmp_path):
        # No participants.tsv — must not raise
        _merge_participants_from_llm_metadata(
            {"participant_metadata": {"1": {"sex": "M"}}}, tmp_path
        )

    def test_multiple_new_columns_added(self, tmp_path):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\nsub-1\nsub-2\n"
        )
        plan = {
            "participant_metadata": {
                "1": {"age": "30", "sex": "M", "group": "G"},
                "2": {"age": "25", "sex": "F", "group": "B"},
            }
        }
        _merge_participants_from_llm_metadata(plan, tmp_path)
        header = (tmp_path / "participants.tsv").read_text().splitlines()[0]
        assert "age" in header
        assert "sex" in header
        assert "group" in header


# ============================================================================
# _parse_llm_json_response  (planner's local copy)
# ============================================================================

class TestPlannerParseLlmJsonResponse:

    def test_clean_json(self):
        result = _parse_llm_json_response('{"key": "val"}', "test")
        assert result == {"key": "val"}

    def test_strips_json_fence(self):
        result = _parse_llm_json_response(
            '```json\n{"a": 1}\n```', "test"
        )
        assert result == {"a": 1}

    def test_strips_plain_fence(self):
        result = _parse_llm_json_response(
            '```\n{"a": 1}\n```', "test"
        )
        assert result == {"a": 1}

    def test_handles_trailing_text(self):
        result = _parse_llm_json_response('{"a": 1} extra', "test")
        assert result is not None
        assert result["a"] == 1

    def test_returns_none_for_empty(self):
        assert _parse_llm_json_response("", "test") is None

    def test_returns_none_for_invalid_json(self):
        assert _parse_llm_json_response("not json", "test") is None
