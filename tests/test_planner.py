# tests/test_planner.py
# Unit tests for autobidsify/converters/planner.py
# Tests cover ONLY the pure-Python functions — no LLM calls.

import pytest
from pathlib import Path

from autobidsify.converters.planner import (
    _extract_subjects_from_directory_structure,
    _extract_subjects_from_flat_filenames,
    _is_data_file,
    _collect_extra_columns,
    _write_participants_from_plan,
    _merge_participants_from_llm_metadata,
)


# ============================================================================
# _is_data_file
# ============================================================================

class TestIsDataFile:

    def test_snirf_is_data(self):
        assert _is_data_file("sub-01/nirs/sub-01_task-rest_nirs.snirf") is True

    def test_dcm_is_data(self):
        assert _is_data_file("VHMCT1mm-Hip (134).dcm") is True

    def test_nii_gz_is_data(self):
        assert _is_data_file("sub-01/anat/sub-01_T1w.nii.gz") is True

    def test_nii_is_data(self):
        assert _is_data_file("scan.nii") is True

    def test_mat_is_data(self):
        assert _is_data_file("data.mat") is True

    def test_nirs_is_data(self):
        assert _is_data_file("subject01.nirs") is True

    def test_pdf_is_not_data(self):
        assert _is_data_file("paper.pdf") is False

    def test_xlsx_is_not_data(self):
        assert _is_data_file("group_stats.xlsx") is False

    def test_cfg_is_not_data(self):
        assert _is_data_file("procStream.cfg") is False

    def test_json_is_not_data(self):
        assert _is_data_file("dataset_description.json") is False


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
        result = _extract_subjects_from_flat_filenames(vh_files)
        assert result["success"] is True
        ids = {r["original_id"] for r in result["subject_records"]}
        assert any("VHM" in i for i in ids)
        assert any("VHF" in i for i in ids)

    def test_file_counts_per_subject(self, vh_files):
        result = _extract_subjects_from_flat_filenames(vh_files)
        assert result["success"] is True
        total = sum(r["file_count"] for r in result["subject_records"])
        assert total == len(vh_files)

    def test_filters_non_data_files(self):
        files = [
            "PD01/1_resting.snirf",
            "PD01/2_walking.snirf",
            "group_stats.xlsx",    # should be filtered
            "paper.pdf",           # should be filtered
            "procStream.cfg",      # should be filtered
        ]
        result = _extract_subjects_from_flat_filenames(files)
        assert result["success"] is True
        ids = {r["original_id"] for r in result["subject_records"]}
        assert "group" not in ids
        assert "paper" not in ids
        assert "procStream" not in ids

    def test_returns_success_false_for_empty(self):
        result = _extract_subjects_from_flat_filenames([])
        assert result["success"] is False

    def test_returns_success_false_for_no_data_files(self):
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
        assert len(lines) == 3  # header + 2 subjects

    def test_participant_id_column(self, tmp_path, simple_plan):
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
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert "group" in lines[0]

    def test_overwrites_existing_file(self, tmp_path, simple_plan):
        (tmp_path / "participants.tsv").write_text("stale\n")
        _write_participants_from_plan(simple_plan, tmp_path, user_n_subjects=2)
        content = (tmp_path / "participants.tsv").read_text()
        assert "stale" not in content

    def test_warns_when_llm_count_less_than_user(self, tmp_path):
        plan = {
            "assignment_rules": [{"subject": "1", "original": "x", "match": ["*x*"]}],
            "subjects": {"labels": ["1"], "count": 1},
            "participant_metadata": {},
        }
        # Should not raise, just warn
        _write_participants_from_plan(plan, tmp_path, user_n_subjects=5)
        lines = (tmp_path / "participants.tsv").read_text().splitlines()
        assert len(lines) == 2  # header + 1 (no padding)


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
                # subject 2 missing
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
        # No participants.tsv exists
        _merge_participants_from_llm_metadata(
            {"participant_metadata": {"1": {"sex": "M"}}}, tmp_path
        )
        # Should not raise