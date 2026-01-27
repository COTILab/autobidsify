# constants.py
# Global constants and configurations

# ==================== TRIO FILE NAMES ====================
TRIO_DATASET_DESC = "dataset_description.json"
TRIO_README = "README.md"
TRIO_PARTICIPANTS = "participants.tsv"

# ==================== LICENSE WHITELIST (BIDS 1.10.0) ====================
LICENSE_WHITELIST = [
    "PDDL",
    "CC0",
    "PD",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "CDDL-1.0",
    "MPL",
    "MIT",
    "GPL-2.0",
    "GPL-2.0+",
    "GPL-3.0",
    "GPL-3.0+",
    "LGPL-3.0+",
    "GFDL-1.3",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-NC-ND-4.0",
    "Non-Standard"
]

LICENSE_DESCRIPTIONS = {
    "PDDL": "PDDL-1.0 - Open Data Commons Public Domain Dedication & License 1.0",
    "CC0": "CC0-1.0 - Creative Commons Zero v1.0 Universal",
    "PD": "PD - Public domain",
    "CC-BY-4.0": "CC-BY-4.0 - Creative Commons Attribution 4.0 International",
    "CC-BY-SA-4.0": "CC-BY-SA-4.0 - Creative Commons Attribution Share Alike 4.0 International",
    "BSD-3-Clause": "BSD-3-Clause - BSD 3-Clause License",
    "BSD-2-Clause": "BSD-2-Clause - BSD 2-Clause License",
    "CDDL-1.0": "CDDL-1.0 - Common Development and Distribution License 1.0",
    "MPL": "MPL - Mozilla Public License 2.0",
    "MIT": "MIT - MIT License",
    "GPL-2.0": "GPL-2.0 - GNU General Public License v2.0 only",
    "GPL-2.0+": "GPL-2.0+ - GNU General Public License v2.0 and later",
    "GPL-3.0": "GPL-3.0 - GNU General Public License v3.0 only",
    "GPL-3.0+": "GPL-3.0+ - GNU General Public License v3.0 or later",
    "LGPL-3.0+": "LGPL-3.0+ - GNU Lesser General Public License v3.0 or later",
    "GFDL-1.3": "GFDL-1.3 - GNU Free Documentation License v1.3",
    "CC-BY-NC-4.0": "CC-BY-NC-4.0 - Creative Commons Attribution Non Commercial 4.0",
    "CC-BY-NC-SA-4.0": "CC-BY-NC-SA-4.0 - Creative Commons Attribution Non Commercial Share Alike 4.0",
    "CC-BY-NC-ND-4.0": "CC-BY-NC-ND-4.0 - Creative Commons Attribution Non Commercial No Derivatives 4.0",
    "Non-Standard": "Non-Standard - User-defined license"
}

# ==================== SEVERITY LEVELS ====================
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_BLOCK = "block"

# ==================== FILE EXTENSIONS ====================
# MODIFIED: Added JNIfTI extensions (.jnii, .bnii)
MRI_EXTENSIONS = ['.nii', '.nii.gz', '.dcm', '.mat', '.jnii', '.bnii']
NIRS_EXTENSIONS = ['.snirf', '.nirs', '.csv', '.mat']
DOCUMENT_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.rst']

# NEW: JNIfTI specific extensions
JNIFTI_EXTENSIONS = ['.jnii', '.bnii']

# ==================== MODALITY TYPES ====================
MODALITY_MRI = "mri"
MODALITY_NIRS = "nirs"
MODALITY_MIXED = "mixed"

# ==================== BIDS VERSIONS ====================
BIDS_VERSION = "1.10.0"

# ==================== DIRECTORY NAMES ====================
STAGING_DIR = "_staging"
EXTRACTED_DIR = "extracted"
EVIDENCE_BUNDLE = "evidence_bundle.json"
CLASSIFICATION_RESULT = "classification.json"
CLASSIFICATION_PLAN = "classification_plan.json"

NIRS_POOL = "_staging/nirs_pool"
MRI_POOL = "_staging/mri_pool"
UNKNOWN_POOL = "_staging/unknown"

HEADERS_DRAFT = "nirs_headers_draft.json"
HEADERS_NORMALIZED = "nirs_headers_normalized.json"
VOXEL_DRAFT = "mri_voxel_draft.json"
VOXEL_FINAL_PLAN = "mri_voxel_final_plan.json"
BIDS_PLAN = "BIDSPlan.yaml"

# ==================== DEFAULT VALUES ====================
DEFAULT_MODEL = "gpt-4o"
DEFAULT_NSUBJECTS = 1

REASONING_MODELS_PREFIXES = ["o1", "o3", "gpt-5"]

TASK_TEMPERATURES = {
    "classification": 0.15,
    "dataset_description": 0.1,
    "readme": 0.4,
    "participants": 0.2,
    "nirs_draft": 0.2,
    "nirs_normalize": 0.1,
    "mri_voxel_draft": 0.2,
    "mri_voxel_final": 0.1,
    "bids_plan": 0.15
}

# ==================== VALIDATION THRESHOLDS ====================
MAX_DOCUMENT_SIZE_MB = 10
MAX_PDF_PAGES = 50
MAX_TABLE_ROWS = 100
MAX_TEXT_SIZE = 1024 * 1024
MAX_PDF_SIZE = 10 * 1024 * 1024
MAX_DOCX_SIZE = 10 * 1024 * 1024
