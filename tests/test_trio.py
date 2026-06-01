# tests/test_trio.py
# Unit tests for autobidsify/stages/trio.py
# Tests cover ONLY pure-Python functions — no LLM calls.
# Functions: normalize_license_locally, _parse_llm_json_response,
#            _validate_dataset_description, _fix_field_types, check_trio_status.

import json
import pytest
from pathlib import Path

from autobidsify.stages.trio import (
    normalize_license_locally,
    check_trio_status,
    _parse_llm_json_response,
    _validate_dataset_description,
    _fix_field_types,
)
from autobidsify.constants import LICENSE_WHITELIST


# ============================================================================
# normalize_license_locally
# ============================================================================

class TestNormalizeLicenseLocally:

    # ── None / empty input ───────────────────────────────────────────────

    def test_none_returns_none(self):
        assert normalize_license_locally(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_license_locally("") is None

    # ── CC0 variants ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("raw", [
        "CC0", "cc0", "CC0 1.0", "Creative Commons Zero",
        "creativecommonszero", "CC0-1.0", "CC0 Universal",
    ])
    def test_cc0_variants(self, raw):
        assert normalize_license_locally(raw) == "CC0"

    # ── Public Domain ────────────────────────────────────────────────────

    @pytest.mark.parametrize("raw, expected", [
        ("PD",            "PD"),
        ("Public Domain", "PD"),
        ("public domain", "PD"),
        ("PDDL",          "PDDL"),
        ("PDDL-1.0",      "PDDL"),
    ])
    def test_public_domain_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ── Creative Commons ─────────────────────────────────────────────────

    @pytest.mark.parametrize("raw, expected", [
        ("CC-BY-4.0",                        "CC-BY-4.0"),
        ("CC BY 4.0",                        "CC-BY-4.0"),
        ("CCBY4",                            "CC-BY-4.0"),
        ("Creative Commons Attribution 4.0", "CC-BY-4.0"),
        ("CC-BY-SA-4.0",                     "CC-BY-SA-4.0"),
        ("CC BY-SA 4.0",                     "CC-BY-SA-4.0"),
        ("CC-BY-NC-4.0",                     "CC-BY-NC-4.0"),
        ("CC BY NC 4.0",                     "CC-BY-NC-4.0"),
        ("CC-BY-NC-SA-4.0",                  "CC-BY-NC-SA-4.0"),
        ("CC-BY-NC-ND-4.0",                  "CC-BY-NC-ND-4.0"),
    ])
    def test_cc_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ── OSI licenses ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("raw, expected", [
        ("MIT",                    "MIT"),
        ("MIT License",            "MIT"),
        ("mit license",            "MIT"),
        ("BSD-3-Clause",           "BSD-3-Clause"),
        ("BSD 3-Clause",           "BSD-3-Clause"),
        ("BSD3",                   "BSD-3-Clause"),
        ("BSD-2-Clause",           "BSD-2-Clause"),
        ("GPL-2.0",                "GPL-2.0"),
        ("GPL 2.0",                "GPL-2.0"),
        ("GPL-3.0",                "GPL-3.0"),
        ("GPL 3.0",                "GPL-3.0"),
        ("GPL-3.0+",               "GPL-3.0+"),
        ("MPL",                    "MPL"),
        ("Mozilla Public License 2.0", "MPL"),
    ])
    def test_osi_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ── Non-standard fallback ────────────────────────────────────────────

    @pytest.mark.parametrize("raw", [
        "Some custom license",
        "Proprietary",
        "All rights reserved",
        "University of Somewhere license",
        "Open Data Commons Attribution License v1.0",
    ])
    def test_unknown_returns_non_standard(self, raw):
        assert normalize_license_locally(raw) == "Non-Standard"

    # ── Result always in whitelist ────────────────────────────────────────

    def test_result_always_in_whitelist(self):
        samples = [
            "CC0", "MIT", "GPL 3.0", "BSD3", "CC BY 4.0",
            "totally unknown license string", "Proprietary"
        ]
        for raw in samples:
            result = normalize_license_locally(raw)
            if result is not None:
                assert result in LICENSE_WHITELIST, \
                    f"normalize_license_locally('{raw}') = '{result}' not in whitelist"


# ============================================================================
# _parse_llm_json_response
# ============================================================================

class TestParseLlmJsonResponse:

    def test_parses_clean_json(self):
        text = '{"name": "test", "value": 42}'
        result = _parse_llm_json_response(text, "test")
        assert result == {"name": "test", "value": 42}

    def test_strips_json_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_llm_json_response(text, "test")
        assert result == {"key": "value"}

    def test_strips_plain_markdown_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = _parse_llm_json_response(text, "test")
        assert result == {"key": "value"}

    def test_handles_trailing_text_via_raw_decode(self):
        text = '{"a": 1} some trailing text'
        result = _parse_llm_json_response(text, "test")
        assert result is not None
        assert result["a"] == 1

    def test_returns_none_for_empty_string(self):
        assert _parse_llm_json_response("", "test") is None

    def test_returns_none_for_whitespace_only(self):
        assert _parse_llm_json_response("   \n  ", "test") is None

    def test_returns_none_for_invalid_json(self):
        result = _parse_llm_json_response("not json at all", "test",
                                           show_preview=False)
        assert result is None

    def test_extracts_json_from_prose(self):
        text = 'Here is the result:\n{"license": "CC0"}\nEnd.'
        result = _parse_llm_json_response(text, "test")
        assert result is not None
        assert result["license"] == "CC0"

    def test_nested_dict_roundtrip(self):
        data = {"subjects": [{"id": "01", "sex": "M"}], "count": 1}
        text = json.dumps(data)
        result = _parse_llm_json_response(text, "test")
        assert result == data


# ============================================================================
# _validate_dataset_description
# ============================================================================

class TestValidateDatasetDescription:

    def test_valid_description_passes(self):
        dd = {"Name": "My Dataset", "BIDSVersion": "1.10.0", "License": "CC0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert is_valid
        assert len(issues) == 0

    def test_missing_name_invalid(self):
        dd = {"BIDSVersion": "1.10.0", "License": "CC0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid
        assert any("Name" in i for i in issues)

    def test_missing_bids_version_invalid(self):
        dd = {"Name": "Test", "License": "CC0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid
        assert any("BIDSVersion" in i for i in issues)

    def test_missing_license_invalid(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid
        assert any("License" in i for i in issues)

    def test_authors_must_be_list(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "CC0", "Authors": "John Doe"
        }
        _, issues = _validate_dataset_description(dd)
        assert any("Authors" in i for i in issues)

    def test_non_standard_without_data_license_warns(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "Non-Standard"
        }
        _, issues = _validate_dataset_description(dd)
        assert any("Non-Standard" in i or "DataLicense" in i for i in issues)

    def test_empty_fields_reported(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "CC0", "Acknowledgements": ""
        }
        _, issues = _validate_dataset_description(dd)
        assert any("Acknowledgements" in i or "Empty" in i for i in issues)

    def test_valid_with_authors_list(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "CC0", "Authors": ["Alice", "Bob"]
        }
        is_valid, issues = _validate_dataset_description(dd)
        assert is_valid


# ============================================================================
# _fix_field_types
# ============================================================================

class TestFixFieldTypes:

    def test_converts_string_authors_to_list(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Authors": "John Doe"}
        fixed, fixes = _fix_field_types(dd)
        assert isinstance(fixed["Authors"], list)
        assert fixed["Authors"] == ["John Doe"]
        assert len(fixes) > 0

    def test_removes_empty_authors_list(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Authors": []}
        fixed, _ = _fix_field_types(dd)
        assert "Authors" not in fixed

    def test_removes_empty_string_optional_field(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Acknowledgements": ""}
        fixed, _ = _fix_field_types(dd)
        assert "Acknowledgements" not in fixed

    def test_preserves_valid_authors_list(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Authors": ["Alice", "Bob"]}
        fixed, _ = _fix_field_types(dd)
        assert fixed["Authors"] == ["Alice", "Bob"]

    def test_required_fields_not_removed(self):
        # Name, BIDSVersion, License must survive even if empty string
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0"}
        fixed, _ = _fix_field_types(dd)
        assert "Name" in fixed
        assert "BIDSVersion" in fixed
        assert "License" in fixed

    def test_converts_funding_string_to_list(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Funding": "NIH R01"}
        fixed, _ = _fix_field_types(dd)
        assert isinstance(fixed["Funding"], list)
        assert fixed["Funding"] == ["NIH R01"]

    def test_no_changes_on_clean_input(self):
        dd = {"Name": "T", "BIDSVersion": "1.10.0", "License": "CC0",
              "Authors": ["Alice"]}
        fixed, fixes = _fix_field_types(dd)
        assert fixed == dd
        assert fixes == []


# ============================================================================
# check_trio_status
# ============================================================================

class TestCheckTrioStatus:

    def test_all_missing_returns_false(self, tmp_path):
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["exists"] is False
        assert status["readme"]["exists"] is False
        assert status["participants"]["exists"] is False

    def test_detects_dataset_description(self, tmp_path):
        (tmp_path / "dataset_description.json").write_text(
            '{"Name": "Test", "BIDSVersion": "1.10.0"}'
        )
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["exists"] is True

    def test_reads_dataset_description_contents(self, tmp_path):
        data = {"Name": "MyStudy", "BIDSVersion": "1.10.0", "License": "CC0"}
        (tmp_path / "dataset_description.json").write_text(json.dumps(data))
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["data"]["Name"] == "MyStudy"

    @pytest.mark.parametrize("variant", [
        "README.md", "readme.txt", "README", "README.rst"
    ])
    def test_detects_readme_variants(self, tmp_path, variant):
        (tmp_path / variant).write_text("# Dataset")
        status = check_trio_status(tmp_path)
        assert status["readme"]["exists"] is True
        assert status["readme"]["variant"] == variant

    def test_detects_participants_tsv(self, tmp_path):
        (tmp_path / "participants.tsv").write_text("participant_id\nsub-01\n")
        status = check_trio_status(tmp_path)
        assert status["participants"]["exists"] is True

    def test_all_three_present(self, tmp_path):
        (tmp_path / "dataset_description.json").write_text(
            '{"Name": "T", "BIDSVersion": "1.10.0"}'
        )
        (tmp_path / "README.md").write_text("# T")
        (tmp_path / "participants.tsv").write_text("participant_id\nsub-01\n")
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["exists"] is True
        assert status["readme"]["exists"] is True
        assert status["participants"]["exists"] is True

    def test_corrupt_json_does_not_crash(self, tmp_path):
        (tmp_path / "dataset_description.json").write_text("{ bad json }")
        # Should not raise, data just stays None
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["exists"] is True
        assert status["dataset_description"]["data"] is None
