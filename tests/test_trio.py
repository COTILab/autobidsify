# tests/test_trio.py
# Unit tests for autobidsify/stages/trio.py
# Tests cover: normalize_license_locally (all alias variants),
#              _parse_llm_json_response, _validate_dataset_description,
#              _fix_field_types, and check_trio_status.
# NOTE: LLM-calling functions (generate_dataset_description, etc.)
#       are NOT tested here — they require live API keys.

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

    # ---- CC0 variants ----
    @pytest.mark.parametrize("raw, expected", [
        ("CC0",                          "CC0"),
        ("cc0",                          "CC0"),
        ("CC0 1.0",                      "CC0"),
        ("Creative Commons Zero",        "CC0"),
        ("creativecommonszero",          "CC0"),
        ("CC0-1.0",                      "CC0"),
        ("CC0 Universal",                "CC0"),
    ])
    def test_cc0_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ---- Public Domain variants ----
    @pytest.mark.parametrize("raw, expected", [
        ("PD",            "PD"),
        ("Public Domain", "PD"),
        ("public domain", "PD"),
        ("PDDL",          "PDDL"),
        ("PDDL-1.0",      "PDDL"),
    ])
    def test_public_domain_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ---- Creative Commons variants ----
    @pytest.mark.parametrize("raw, expected", [
        ("CC-BY-4.0",                              "CC-BY-4.0"),
        ("CC BY 4.0",                              "CC-BY-4.0"),
        ("CCBY4",                                  "CC-BY-4.0"),
        ("Creative Commons Attribution 4.0",       "CC-BY-4.0"),
        ("CC-BY-SA-4.0",                           "CC-BY-SA-4.0"),
        ("CC BY-SA 4.0",                           "CC-BY-SA-4.0"),
        ("CC-BY-NC-4.0",                           "CC-BY-NC-4.0"),
        ("CC BY NC 4.0",                           "CC-BY-NC-4.0"),
        ("CC-BY-NC-SA-4.0",                        "CC-BY-NC-SA-4.0"),
        ("CC-BY-NC-ND-4.0",                        "CC-BY-NC-ND-4.0"),
    ])
    def test_creative_commons_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ---- OSI licenses ----
    @pytest.mark.parametrize("raw, expected", [
        ("MIT",              "MIT"),
        ("MIT License",      "MIT"),
        ("mit license",      "MIT"),
        ("BSD-3-Clause",     "BSD-3-Clause"),
        ("BSD 3-Clause",     "BSD-3-Clause"),
        ("BSD3",             "BSD-3-Clause"),
        ("BSD-2-Clause",     "BSD-2-Clause"),
        ("GPL-2.0",          "GPL-2.0"),
        ("GPL 2.0",          "GPL-2.0"),
        ("GPL-3.0",          "GPL-3.0"),
        ("GPL 3.0",          "GPL-3.0"),
        ("GPL-3.0+",         "GPL-3.0+"),
        ("MPL",              "MPL"),
        ("Mozilla Public License 2.0", "MPL"),
    ])
    def test_osi_license_variants(self, raw, expected):
        assert normalize_license_locally(raw) == expected

    # ---- Non-Standard fallback ----
    @pytest.mark.parametrize("raw", [
        "Some custom license",
        "Proprietary",
        "All rights reserved",
        "University of Somewhere license",
    ])
    def test_unknown_license_returns_non_standard(self, raw):
        result = normalize_license_locally(raw)
        assert result == "Non-Standard"

    # ---- Edge cases ----
    def test_none_returns_none(self):
        assert normalize_license_locally(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_license_locally("") is None

    def test_result_always_in_whitelist(self):
        samples = ["CC0", "MIT", "GPL 3.0", "BSD3", "CC BY 4.0",
                   "totally unknown license string"]
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

    def test_handles_extra_trailing_text(self):
        # raw_decode should handle JSON followed by extra content
        text = '{"a": 1} some trailing text'
        result = _parse_llm_json_response(text, "test")
        assert result is not None
        assert result["a"] == 1

    def test_returns_none_for_empty_string(self):
        assert _parse_llm_json_response("", "test") is None

    def test_returns_none_for_invalid_json(self):
        result = _parse_llm_json_response("not json at all", "test", show_preview=False)
        assert result is None

    def test_extracts_json_from_mixed_content(self):
        # Regex fallback: JSON embedded in prose
        text = 'Here is the result:\n{"license": "CC0"}\nEnd.'
        result = _parse_llm_json_response(text, "test")
        assert result is not None
        assert result["license"] == "CC0"


# ============================================================================
# _validate_dataset_description
# ============================================================================

class TestValidateDatasetDescription:

    def test_valid_description_passes(self):
        dd = {
            "Name": "My Dataset",
            "BIDSVersion": "1.10.0",
            "License": "CC0"
        }
        is_valid, issues = _validate_dataset_description(dd)
        assert is_valid
        assert len(issues) == 0

    def test_missing_name_is_invalid(self):
        dd = {"BIDSVersion": "1.10.0", "License": "CC0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid
        assert any("Name" in i for i in issues)

    def test_missing_bids_version_is_invalid(self):
        dd = {"Name": "Test", "License": "CC0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid

    def test_missing_license_is_invalid(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0"}
        is_valid, issues = _validate_dataset_description(dd)
        assert not is_valid
        assert any("License" in i for i in issues)

    def test_authors_must_be_list(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "CC0", "Authors": "John Doe"  # string instead of list
        }
        is_valid, issues = _validate_dataset_description(dd)
        assert any("Authors" in i for i in issues)

    def test_non_standard_without_data_license_warns(self):
        dd = {
            "Name": "Test", "BIDSVersion": "1.10.0",
            "License": "Non-Standard"
        }
        _, issues = _validate_dataset_description(dd)
        assert any("Non-Standard" in i or "DataLicense" in i for i in issues)


# ============================================================================
# _fix_field_types
# ============================================================================

class TestFixFieldTypes:

    def test_converts_string_authors_to_list(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0",
              "License": "CC0", "Authors": "John Doe"}
        fixed, fixes = _fix_field_types(dd)
        assert isinstance(fixed["Authors"], list)
        assert fixed["Authors"] == ["John Doe"]

    def test_removes_empty_authors_list(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0",
              "License": "CC0", "Authors": []}
        fixed, fixes = _fix_field_types(dd)
        assert "Authors" not in fixed

    def test_removes_empty_string_optional_fields(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0",
              "License": "CC0", "Acknowledgements": ""}
        fixed, _ = _fix_field_types(dd)
        assert "Acknowledgements" not in fixed

    def test_preserves_valid_fields(self):
        dd = {"Name": "Test", "BIDSVersion": "1.10.0",
              "License": "CC0", "Authors": ["Alice", "Bob"]}
        fixed, _ = _fix_field_types(dd)
        assert fixed["Authors"] == ["Alice", "Bob"]
        assert fixed["Name"] == "Test"


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
        dd_path = tmp_path / "dataset_description.json"
        dd_path.write_text('{"Name": "Test", "BIDSVersion": "1.10.0"}')
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["exists"] is True

    def test_detects_readme_variants(self, tmp_path):
        for variant in ["README.md", "readme.txt", "README"]:
            # Write each variant and check detection
            readme = tmp_path / variant
            readme.write_text("# Dataset")
            status = check_trio_status(tmp_path)
            assert status["readme"]["exists"] is True
            readme.unlink()

    def test_detects_participants_tsv(self, tmp_path):
        (tmp_path / "participants.tsv").write_text("participant_id\nsub-01\n")
        status = check_trio_status(tmp_path)
        assert status["participants"]["exists"] is True

    def test_reads_dataset_description_json(self, tmp_path):
        dd_path = tmp_path / "dataset_description.json"
        data = {"Name": "MyStudy", "BIDSVersion": "1.10.0", "License": "CC0"}
        dd_path.write_text(json.dumps(data))
        status = check_trio_status(tmp_path)
        assert status["dataset_description"]["data"]["Name"] == "MyStudy"
