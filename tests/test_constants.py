# tests/test_constants.py
# Unit tests for autobidsify/constants.py — v10, EEG support added.

import pytest
from autobidsify import constants


class TestFileExtensions:

    def test_mri_extensions_are_list(self):
        assert isinstance(constants.MRI_EXTENSIONS, list)

    def test_nirs_extensions_are_list(self):
        assert isinstance(constants.NIRS_EXTENSIONS, list)

    def test_eeg_extensions_are_list(self):
        assert isinstance(constants.EEG_EXTENSIONS, list)

    def test_eeg_aux_extensions_are_list(self):
        assert isinstance(constants.EEG_AUX_EXTENSIONS, list)

    def test_mri_contains_expected_formats(self):
        for ext in [".nii", ".nii.gz", ".dcm", ".jnii", ".bnii"]:
            assert ext in constants.MRI_EXTENSIONS

    def test_nirs_contains_expected_formats(self):
        for ext in [".snirf", ".nirs", ".mat"]:
            assert ext in constants.NIRS_EXTENSIONS

    def test_eeg_contains_expected_formats(self):
        for ext in [".edf", ".vhdr", ".set", ".bdf"]:
            assert ext in constants.EEG_EXTENSIONS

    def test_eeg_aux_contains_brainvision_eeglab(self):
        for ext in [".vmrk", ".eeg", ".fdt"]:
            assert ext in constants.EEG_AUX_EXTENSIONS

    def test_mat_only_in_nirs_not_mri(self):
        assert ".mat" not in constants.MRI_EXTENSIONS
        assert ".mat" in constants.NIRS_EXTENSIONS

    def test_jnifti_subset_of_mri(self):
        for ext in constants.JNIFTI_EXTENSIONS:
            assert ext in constants.MRI_EXTENSIONS

    def test_no_overlap_mri_nirs(self):
        overlap = set(constants.MRI_EXTENSIONS) & set(constants.NIRS_EXTENSIONS)
        assert len(overlap) == 0, f"MRI/NIRS share: {overlap}"

    def test_no_overlap_mri_eeg(self):
        overlap = set(constants.MRI_EXTENSIONS) & set(constants.EEG_EXTENSIONS)
        assert len(overlap) == 0, f"MRI/EEG share: {overlap}"

    def test_no_overlap_nirs_eeg(self):
        overlap = set(constants.NIRS_EXTENSIONS) & set(constants.EEG_EXTENSIONS)
        assert len(overlap) == 0, f"NIRS/EEG share: {overlap}"

    def test_all_extensions_start_with_dot(self):
        all_exts = (constants.MRI_EXTENSIONS + constants.NIRS_EXTENSIONS +
                    constants.EEG_EXTENSIONS + constants.EEG_AUX_EXTENSIONS)
        for ext in all_exts:
            assert ext.startswith(".")


class TestBidsVersion:

    def test_is_string(self):
        assert isinstance(constants.BIDS_VERSION, str)

    def test_value(self):
        assert constants.BIDS_VERSION == "1.10.0"

    def test_semver_format(self):
        parts = constants.BIDS_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestLicenseWhitelist:

    def test_is_list(self):
        assert isinstance(constants.LICENSE_WHITELIST, list)

    def test_contains_common_licenses(self):
        for lic in ["CC0", "MIT", "CC-BY-4.0", "PDDL", "Non-Standard"]:
            assert lic in constants.LICENSE_WHITELIST

    def test_no_duplicates(self):
        assert len(constants.LICENSE_WHITELIST) == len(set(constants.LICENSE_WHITELIST))

    def test_description_keys_in_whitelist(self):
        for key in constants.LICENSE_DESCRIPTIONS:
            assert key in constants.LICENSE_WHITELIST

    def test_description_values_are_strings(self):
        for k, v in constants.LICENSE_DESCRIPTIONS.items():
            assert isinstance(v, str)


class TestModalityConstants:

    def test_all_four_modalities_defined(self):
        assert constants.MODALITY_MRI   == "mri"
        assert constants.MODALITY_NIRS  == "nirs"
        assert constants.MODALITY_MIXED == "mixed"
        assert constants.MODALITY_EEG   == "eeg"

    def test_all_lowercase(self):
        for val in [constants.MODALITY_MRI, constants.MODALITY_NIRS,
                    constants.MODALITY_MIXED, constants.MODALITY_EEG]:
            assert val == val.lower()


class TestSeverityLevels:

    def test_severity_values(self):
        assert constants.SEVERITY_INFO  == "info"
        assert constants.SEVERITY_WARN  == "warn"
        assert constants.SEVERITY_BLOCK == "block"


class TestDirectoryNames:

    def test_staging_dir(self):
        assert constants.STAGING_DIR == "_staging"

    def test_pool_dirs_include_eeg(self):
        assert "nirs_pool" in constants.NIRS_POOL
        assert "mri_pool"  in constants.MRI_POOL
        assert "eeg_pool"  in constants.EEG_POOL
        assert "unknown"   in constants.UNKNOWN_POOL

    def test_trio_filenames(self):
        assert constants.TRIO_DATASET_DESC == "dataset_description.json"
        assert constants.TRIO_README       == "README.md"
        assert constants.TRIO_PARTICIPANTS == "participants.tsv"

    def test_bids_plan_filename(self):
        assert constants.BIDS_PLAN == "BIDSPlan.yaml"


class TestModelConfig:

    def test_default_model_is_nonempty_string(self):
        assert isinstance(constants.DEFAULT_MODEL, str)
        assert len(constants.DEFAULT_MODEL) > 0

    def test_reasoning_prefixes_list_of_strings(self):
        assert isinstance(constants.REASONING_MODELS_PREFIXES, list)
        for p in constants.REASONING_MODELS_PREFIXES:
            assert isinstance(p, str)

    def test_qwen_prefix_present(self):
        assert "qwen" in constants.QWEN_MODEL_PREFIXES

    def test_task_temperatures_floats_in_range(self):
        for task, temp in constants.TASK_TEMPERATURES.items():
            assert isinstance(temp, float), f"{task}: not float"
            assert 0.0 <= temp <= 1.0,      f"{task}: {temp} out of range"

    def test_task_temperatures_has_expected_tasks(self):
        for task in ["classification", "dataset_description", "readme",
                     "participants", "bids_plan"]:
            assert task in constants.TASK_TEMPERATURES


class TestSizeLimits:

    def test_all_positive_integers(self):
        for name in ["MAX_DOCUMENT_SIZE_MB", "MAX_PDF_PAGES", "MAX_TABLE_ROWS",
                     "MAX_TEXT_SIZE", "MAX_PDF_SIZE", "MAX_DOCX_SIZE"]:
            val = getattr(constants, name)
            assert isinstance(val, int), f"{name} not int"
            assert val > 0,              f"{name} not positive"

    def test_pdf_size_at_least_text_size(self):
        assert constants.MAX_PDF_SIZE >= constants.MAX_TEXT_SIZE
