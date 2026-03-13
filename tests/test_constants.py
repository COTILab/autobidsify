# tests/test_constants.py
# Unit tests for autobidsify/constants.py
# Validates that all constants have correct types, expected values, and
# are consistent with the BIDS 1.10.0 specification.

import pytest
from autobidsify import constants


# ============================================================================
# File extension sets
# ============================================================================

class TestFileExtensions:
    def test_mri_extensions_are_list(self):
        assert isinstance(constants.MRI_EXTENSIONS, list)

    def test_nirs_extensions_are_list(self):
        assert isinstance(constants.NIRS_EXTENSIONS, list)

    def test_mri_extensions_contain_expected_formats(self):
        for ext in [".nii", ".nii.gz", ".dcm", ".jnii", ".bnii"]:
            assert ext in constants.MRI_EXTENSIONS, f"{ext} missing from MRI_EXTENSIONS"

    def test_nirs_extensions_contain_expected_formats(self):
        for ext in [".snirf", ".nirs", ".mat"]:
            assert ext in constants.NIRS_EXTENSIONS, f"{ext} missing from NIRS_EXTENSIONS"

    def test_mat_is_only_in_nirs_not_mri(self):
        # .mat is Homer3/fNIRS only — must never be in MRI_EXTENSIONS
        assert ".mat" not in constants.MRI_EXTENSIONS
        assert ".mat" in constants.NIRS_EXTENSIONS

    def test_jnifti_extensions_subset_of_mri(self):
        for ext in constants.JNIFTI_EXTENSIONS:
            assert ext in constants.MRI_EXTENSIONS

    def test_no_overlap_between_mri_and_nirs(self):
        mri = set(constants.MRI_EXTENSIONS)
        nirs = set(constants.NIRS_EXTENSIONS)
        overlap = mri & nirs
        assert len(overlap) == 0, f"MRI and NIRS share extensions: {overlap}"

    def test_all_extensions_start_with_dot(self):
        for ext in constants.MRI_EXTENSIONS + constants.NIRS_EXTENSIONS:
            assert ext.startswith("."), f"Extension missing leading dot: {ext}"


# ============================================================================
# BIDS version
# ============================================================================

class TestBidsVersion:
    def test_bids_version_is_string(self):
        assert isinstance(constants.BIDS_VERSION, str)

    def test_bids_version_value(self):
        assert constants.BIDS_VERSION == "1.10.0"

    def test_bids_version_format(self):
        # Must follow semver pattern X.Y.Z
        parts = constants.BIDS_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ============================================================================
# License whitelist
# ============================================================================

class TestLicenseWhitelist:
    def test_is_list(self):
        assert isinstance(constants.LICENSE_WHITELIST, list)

    def test_contains_common_open_licenses(self):
        for lic in ["CC0", "MIT", "CC-BY-4.0", "PDDL"]:
            assert lic in constants.LICENSE_WHITELIST, f"{lic} missing"

    def test_contains_non_standard(self):
        assert "Non-Standard" in constants.LICENSE_WHITELIST

    def test_no_duplicates(self):
        assert len(constants.LICENSE_WHITELIST) == len(set(constants.LICENSE_WHITELIST))

    def test_license_descriptions_keys_match_whitelist(self):
        # Every key in LICENSE_DESCRIPTIONS must be in LICENSE_WHITELIST
        for key in constants.LICENSE_DESCRIPTIONS:
            assert key in constants.LICENSE_WHITELIST, \
                f"LICENSE_DESCRIPTIONS has key '{key}' not in whitelist"

    def test_license_descriptions_values_are_strings(self):
        for k, v in constants.LICENSE_DESCRIPTIONS.items():
            assert isinstance(v, str), f"Description for '{k}' is not a string"


# ============================================================================
# Modality constants
# ============================================================================

class TestModalityConstants:
    def test_modality_values(self):
        assert constants.MODALITY_MRI == "mri"
        assert constants.MODALITY_NIRS == "nirs"
        assert constants.MODALITY_MIXED == "mixed"

    def test_modality_are_lowercase_strings(self):
        for val in [constants.MODALITY_MRI, constants.MODALITY_NIRS, constants.MODALITY_MIXED]:
            assert val == val.lower()


# ============================================================================
# Severity levels
# ============================================================================

class TestSeverityLevels:
    def test_severity_values(self):
        assert constants.SEVERITY_INFO == "info"
        assert constants.SEVERITY_WARN == "warn"
        assert constants.SEVERITY_BLOCK == "block"


# ============================================================================
# Directory names and file names
# ============================================================================

class TestDirectoryNames:
    def test_staging_dir_name(self):
        assert constants.STAGING_DIR == "_staging"

    def test_trio_file_names(self):
        assert constants.TRIO_DATASET_DESC == "dataset_description.json"
        assert constants.TRIO_README == "README.md"
        assert constants.TRIO_PARTICIPANTS == "participants.tsv"

    def test_bids_plan_filename(self):
        assert constants.BIDS_PLAN == "BIDSPlan.yaml"


# ============================================================================
# Model configuration
# ============================================================================

class TestModelConfig:
    def test_default_model_is_string(self):
        assert isinstance(constants.DEFAULT_MODEL, str)
        assert len(constants.DEFAULT_MODEL) > 0

    def test_reasoning_model_prefixes_are_list_of_strings(self):
        assert isinstance(constants.REASONING_MODELS_PREFIXES, list)
        for prefix in constants.REASONING_MODELS_PREFIXES:
            assert isinstance(prefix, str)

    def test_qwen_prefix_in_list(self):
        assert "qwen" in constants.QWEN_MODEL_PREFIXES

    def test_task_temperatures_are_floats_in_range(self):
        for task, temp in constants.TASK_TEMPERATURES.items():
            assert isinstance(temp, float), f"Temperature for '{task}' is not float"
            assert 0.0 <= temp <= 1.0, f"Temperature for '{task}' out of [0,1]: {temp}"

    def test_task_temperatures_contains_expected_tasks(self):
        expected = ["classification", "dataset_description", "readme",
                    "participants", "bids_plan"]
        for task in expected:
            assert task in constants.TASK_TEMPERATURES, f"Missing task: {task}"


# ============================================================================
# Size limits
# ============================================================================

class TestSizeLimits:
    def test_limits_are_positive_integers(self):
        for name in ["MAX_DOCUMENT_SIZE_MB", "MAX_PDF_PAGES",
                     "MAX_TABLE_ROWS", "MAX_TEXT_SIZE",
                     "MAX_PDF_SIZE", "MAX_DOCX_SIZE"]:
            val = getattr(constants, name)
            assert isinstance(val, int), f"{name} is not int"
            assert val > 0, f"{name} must be positive"

    def test_pdf_size_larger_than_text_size(self):
        # PDFs are allowed to be larger than plain text
        assert constants.MAX_PDF_SIZE >= constants.MAX_TEXT_SIZE
