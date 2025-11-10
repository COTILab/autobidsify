import os
import json
from typing import Any
from openai import OpenAI, OpenAIError
from utils import warn, fatal

class LLMHardFail(Exception):
    def __init__(self, step: str, error_type: str, message: str):
        self.step = step
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{step}] {error_type}: {message}")

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fatal("OPENAI_API_KEY not found in environment. Please set it.")
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise LLMHardFail("Initialization", "ClientError", str(e))

def _call_llm(model: str, system_prompt: str, user_payload: str, step: str, temperature: float = None) -> str:
    client = _get_client()
    
    try:
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ]
        }
        
        is_reasoning_model = (
            model.startswith("o1") or 
            model.startswith("o3") or
            model.startswith("gpt-5")
        )
        
        if is_reasoning_model:
            api_params["max_completion_tokens"] = 32000
        else:
            api_params["max_completion_tokens"] = 16000
            if temperature is not None:
                api_params["temperature"] = temperature
        
        response = client.chat.completions.create(**api_params)
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
                if content and content.strip():
                    return content.strip()
                else:
                    raise LLMHardFail(step, "EmptyResponse", f"LLM returned empty content")
            else:
                raise LLMHardFail(step, "InvalidResponse", "Response has no message.content")
        else:
            raise LLMHardFail(step, "InvalidResponse", "Response has no choices")
        
    except OpenAIError as e:
        raise LLMHardFail(step, e.__class__.__name__, str(e))
    except Exception as e:
        raise LLMHardFail(step, "UnexpectedError", str(e))

PROMPT_CLASSIFICATION = """You are a neuroimaging data triage expert.

Input: evidence bundle with documents[] containing full protocol text.

Output JSON (ONLY valid JSON, no extra text):
{
  "nirs_files": ["path/to/file.csv"],
  "mri_files": ["path/to/brain.nii.gz"],
  "unknown_files": ["path/to/ambiguous.mat"],
  "classification_rationale": {...},
  "questions": [...]
}"""

PROMPT_TRIO_DATASET_DESC = """You are a BIDS dataset_description.json generator.

CRITICAL: Use user_hints.user_text to extract dataset information!

CRITICAL RULES:
- Authors MUST be array: ["Name 1", "Name 2", "Name 3"]
- Funding MUST be array
- EthicsApprovals MUST be array
- DO NOT include empty strings "" or empty arrays []
- License normalization: "CC BY 4.0" -> "CC-BY-4.0"

Extract from user_hints.user_text:
- Dataset name
- Authors/institutions mentioned
- Funding sources
- License information (default to PD for public domain datasets)

Input: {existing, documents, user_hints, counts_by_ext}

Output JSON (ONLY valid JSON, no extra text):
{
  "action": "create|update",
  "dataset_description": {
    "Name": "...",
    "BIDSVersion": "1.10.0",
    "DatasetType": "raw",
    "License": "PD",
    "Authors": ["Institution or Author Name"]
  },
  "extraction_log": {...},
  "questions": []
}"""

PROMPT_TRIO_README = """Generate README.md for BIDS dataset.

CRITICAL: Use user_hints.user_text as primary source for README content.

Create comprehensive README with sections:
- Overview
- Dataset Description  
- Data Acquisition
- File Organization
- Usage Notes
- References

Output: Direct Markdown text (no JSON wrapper)"""

PROMPT_TRIO_PARTICIPANTS = """You are a BIDS participants.tsv generator.

CRITICAL: Extract participant metadata from user_hints.user_text!

Examples:
- "1 male, 1 female" → sex column: M, F
- "ages 25-65" → age column
- "patients and controls" → group column

Return column structure (Python will generate rows):

Output JSON:
{
  "columns": [
    {"name": "participant_id", "required": true},
    {"name": "sex", "levels": ["M", "F"]},
    {"name": "group", "levels": ["patient", "control"]}
  ]
}"""

PROMPT_NIRS_DRAFT = """fNIRS-to-SNIRF mapper (Draft).

Output JSON (ONLY valid JSON):
{
  "draft": {...},
  "confidence": 0.8,
  "questions": [...]
}"""

PROMPT_NIRS_NORMALIZE = """fNIRS-to-SNIRF mapper (Normalize).

Output JSON (ONLY valid JSON):
{
  "normalized": {...},
  "questions": [...]
}"""

PROMPT_MRI_VOXEL_DRAFT = """MRI voxelization planner (Draft).

Output JSON (ONLY valid JSON):
{
  "volume_candidates": [...],
  "meta_candidates": {...},
  "confidence": 0.8
}"""

PROMPT_MRI_VOXEL_FINAL = """MRI voxelization planner (Final).

Output JSON (ONLY valid JSON):
{
  "conversions": [...],
  "questions": []
}"""

PROMPT_BIDS_PLAN = """You are a BIDS dataset architect with complete decision-making authority.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION: Design a complete BIDS conversion plan for ANY neuroimaging dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL YAML ESCAPING:
- Use DOUBLE backslashes in regex: \\\\d \\\\w \\\\s (NOT \\d \\w \\s)

INPUT STRUCTURE:
{
  "file_structure": {
    "all_files": [complete list of ALL files with full paths/names],
    "total_files": number,
    "counts_by_ext": {".dcm": 2979, ...}
  },
  "user_context": {
    "description": "User's explanation of dataset (READ THIS FIRST!)",
    "n_subjects_hint": number or null,
    "modality_hint": "mri" | "nirs" | "mixed" | null
  },
  "documents": [
    {"filename": "protocol.pdf", "content": "full text..."}
  ],
  "python_observations": {
    "unique_prefixes": ["VHM", "VHF", ...],
    "unique_keywords": ["Head", "Hip", ...],
    "directory_structure": "flat" | "hierarchical",
    "note": "Any patterns Python noticed without interpretation"
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE DATASET (Your primary responsibility!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read user_context.description to extract:
1. Subject count and identification method
   - "2 subjects: male and female" → 2 subjects
   - "VHM = male, VHF = female" → prefix-based mapping
   - "650 participants" → 650 subjects

2. Subject-to-filename mapping rules
   - How are subjects identified in filenames?
   - Prefixes? Directory names? Embedded IDs?

3. Participant metadata
   - Sex, age, group, site, etc.
   - Extract from description

4. File organization logic
   - Body parts → acquisition variants
   - Tasks → task entities
   - Runs → run entities

Analyze all_files to validate and discover:
- Count unique subject identifiers
- Find all variants (body parts, tasks, acquisitions)
- Detect file format (DICOM, NIfTI, CSV, etc.)
- Identify modality types

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: BIDS-COMPLIANT MODALITY SUFFIXES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BIDS anatomical modality suffixes (MUST use these EXACT names):
- T1w: T1-weighted MRI
- T2w: T2-weighted MRI
- T1rho: T1-rho MRI
- T1map: Quantitative T1 map
- T2map: Quantitative T2 map
- T2star: T2* weighted MRI
- FLAIR: Fluid attenuated inversion recovery
- FLASH: Fast low angle shot MRI
- PD: Proton density weighted MRI
- PDmap: Quantitative PD map
- PDT2: Combined PD/T2 weighted MRI
- inplaneT1: T1 weighted in plane
- inplaneT2: T2 weighted in plane
- angio: Angiography (MR or CT)
- defacemask: Defacing mask
- SWImagandphase: Susceptibility weighted imaging

CRITICAL CT SCAN HANDLING:
❌ NEVER use these as modality suffix: CT, ct, CTscan, CT-scan, ctscan
✓ For CT scans, ALWAYS use "T1w" as the modality suffix
✓ Use acquisition label to specify it's CT data: acq-ct, acq-cthip, acq-ctchest

Why T1w for CT? CT provides structural anatomy similar to T1-weighted MRI. BIDS does
not have a dedicated CT suffix, so T1w is the appropriate choice for structural CT imaging.

Examples of CORRECT CT naming:
✓ sub-01_acq-ct_T1w.nii.gz              (general CT scan)
✓ sub-01_acq-ctchest_T1w.nii.gz         (CT of chest)
✓ sub-01_acq-cthead_T1w.nii.gz          (CT of head)
✓ sub-02_acq-cthip_T1w.nii.gz           (CT of hip)
✓ sub-03_acq-ctabdomen_T1w.nii.gz       (CT of abdomen)

Examples of INCORRECT CT naming (DO NOT USE):
❌ sub-01_CT.nii.gz                      (not BIDS compliant)
❌ sub-01_acq-chest_CT.nii.gz            (CT suffix not allowed)
❌ sub-01_ct.nii.gz                      (not valid)
❌ sub-01_acq-hip_CTscan.nii.gz          (not BIDS compliant)

How to detect CT scans:
1. Look for "CT" in filename patterns (e.g., "VHMCT1mm-Hip")
2. Check DICOM headers if available (Modality tag)
3. Read user description for mentions of "CT scan", "computed tomography"
4. File naming patterns like "*CT*", "*ct*" in original data

When you encounter CT data:
1. Detect it's CT from filename, description, or metadata
2. Use acq-ct{bodypart} or acq-ct to preserve CT information
3. ALWAYS use T1w as the modality suffix (never CT)
4. Document in README that these are CT scans converted to T1w format

Example transformation for CT data:
Input:  "VHMCT1mm-Hip (134).dcm"  (CT scan of hip, detected from "CT" in name)
Output: "sub-1_acq-cthip_T1w.nii.gz"  ✓ CORRECT (T1w suffix, ct in acquisition)
NOT:    "sub-1_acq-hip_CT.nii.gz"     ❌ WRONG (CT suffix not allowed)

Input:  "patient_001_chest_ct.dcm"
Output: "sub-001_acq-ctchest_T1w.nii.gz"  ✓ CORRECT
NOT:    "sub-001_acq-chest_CT.nii.gz"     ❌ WRONG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: DESIGN SUBJECT GROUPING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose the appropriate grouping method:

METHOD 1: prefix_based
When: Subjects identified by filename prefixes
Example: VHM, VHF, PatientA, SiteB_Patient01

Output structure:
```yaml
subject_grouping:
  method: prefix_based
  description: "Clear explanation of mapping logic"
  rules:
    - prefix: "VHM"
      maps_to_subject: "1"
      match_pattern: "VHM.*"
      metadata:
        sex: "M"
        description: "male cadaver"
    - prefix: "VHF"
      maps_to_subject: "2"
      match_pattern: "VHF.*"
      metadata:
        sex: "F"
        description: "female cadaver"
```

METHOD 2: directory_based  
When: Each subject has their own directory
Example: sub-01/, sub-02/, Cambridge_sub06272/

Output structure:
```yaml
subject_grouping:
  method: directory_based
  extraction_pattern: "([A-Za-z]+)_sub(\\\\d+)"
  subject_from_group: 2
  site_from_group: 1
```

METHOD 3: filename_pattern
When: Subject ID embedded in filename
Example: patient_001_scan.nii.gz, subject_025_T1.nii

Output structure:
```yaml
subject_grouping:
  method: filename_pattern
  extraction_regex: "patient_(\\\\d+)_.*"
  subject_from_group: 1
```

METHOD 4: single_subject
When: All files belong to one subject

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DESIGN FILENAME TRANSFORMATION RULES  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EACH unique filename pattern, create transformation rules.

Analyze filenames to extract BIDS entities:
- Modality type: For CT use T1w, for MRI use T1w/T2w/BOLD/DWI/etc
- Acquisition variants: body parts, protocols, contrasts
- Run numbers: if multiple scans of same type
- Session: if longitudinal data

BIDS entity patterns:
- Anatomical: sub-{subject}_[ses-{session}_][acq-{acquisition}_]<modality>.nii.gz
- Functional: sub-{subject}_[ses-{session}_]task-{task}_[acq-{acq}_][run-{run}_]bold.nii.gz
- Diffusion: sub-{subject}_[ses-{session}_][acq-{acq}_]dwi.nii.gz

Example transformation design for CT data:

Input filename: VHMCT1mm-Hip (134).dcm
Analysis:
  - VHM → subject 1 (from grouping rules)
  - CT → indicates CT scan → use T1w suffix (NOT CT suffix!)
  - Hip → body part → acq-cthip (combine ct + bodypart in acquisition label)
  - (134) → slice number (ignore, dcm2niix will combine)

Output design:
```yaml
filename_rules:
  - match_pattern: "VHM.*-Hip.*\\\\.dcm"
    bids_template: "sub-1_acq-cthip_T1w.nii.gz"
    extract_entities:
      subject: "1"
      acquisition: "cthip"
      modality_suffix: "T1w"
```

CRITICAL: Design rules for ALL observed patterns in all_files!
Don't just handle examples - handle EVERY file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: DETERMINE FILE FORMAT CONVERSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detect file format and set conversion needs:

.dcm files (DICOM):
```yaml
format_ready: false
convert_to: "dicom_to_nifti"
```

.nii.gz files (already NIfTI):
```yaml
format_ready: true
convert_to: none
```

.mat files (MATLAB arrays):
```yaml
format_ready: false
convert_to: "mat_to_nifti"
```

.csv files (fNIRS tables):
```yaml
format_ready: false
convert_to: "csv_to_snirf"
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE OUTPUT 1: Visible Human (Prefix-based grouping with DICOM→NIfTI conversion)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

subject_grouping:
  method: prefix_based
  description: "VHM prefix = male subject (sub-1), VHF prefix = female subject (sub-2)"
  rules:
    - prefix: "VHM"
      maps_to_subject: "1"
      match_pattern: "VHM.*"
      metadata:
        sex: "M"
        description: "Visible Human Male cadaver"
    - prefix: "VHF"
      maps_to_subject: "2"
      match_pattern: "VHF.*"
      metadata:
        sex: "F"
        description: "Visible Human Female cadaver"

participant_metadata:
  "1":
    sex: "M"
    group: "cadaver"
    specimen_id: "Visible Human Male"
  "2":
    sex: "F"
    group: "cadaver"
    specimen_id: "Visible Human Female"

standardization:
  apply: true
  strategy: "prefix_to_numeric_ids"
  reason: "Converting VHM/VHF prefixes to standard sub-1/sub-2 format"

subjects:
  labels: ["1", "2"]

assignment_rules:
  - subject: "1"
    prefix: "VHM"
    match: ["**/VHM*", "**/VHM*.dcm"]
  - subject: "2"
    prefix: "VHF"
    match: ["**/VHF*", "**/VHF*.dcm"]

mappings:
  - modality: mri
    match: ["**/*.dcm"]
    format_ready: false
    convert_to: "dicom_to_nifti"
    bids_out: "sub-{subject}/anat/sub-{subject}_acq-{bodypart}_T1w.nii.gz"
    filename_rules:
      - match_pattern: "VHM.*-Head.*\\\\.dcm"
        bids_template: "sub-1_acq-cthead_T1w.nii.gz"
        extract_entities:
          subject: "1"
          acquisition: "cthead"
      - match_pattern: "VHM.*-Hip.*\\\\.dcm"
        bids_template: "sub-1_acq-cthip_T1w.nii.gz"
        extract_entities:
          subject: "1"
          acquisition: "cthip"
      - match_pattern: "VHM.*-Shoulder.*\\\\.dcm"
        bids_template: "sub-1_acq-ctshoulder_T1w.nii.gz"
        extract_entities:
          subject: "1"
          acquisition: "ctshoulder"
      - match_pattern: "VHM.*-Pelvis.*\\\\.dcm"
        bids_template: "sub-1_acq-ctpelvis_T1w.nii.gz"
        extract_entities:
          subject: "1"
          acquisition: "ctpelvis"
      - match_pattern: "VHF.*-Head.*\\\\.dcm"
        bids_template: "sub-2_acq-cthead_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "cthead"
      - match_pattern: "VHF.*-Hip.*\\\\.dcm"
        bids_template: "sub-2_acq-cthip_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "cthip"
      - match_pattern: "VHF.*-Shoulder.*\\\\.dcm"
        bids_template: "sub-2_acq-ctshoulder_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "ctshoulder"
      - match_pattern: "VHF.*-Knee.*\\\\.dcm"
        bids_template: "sub-2_acq-ctknee_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "ctknee"
      - match_pattern: "VHF.*-Ankle.*\\\\.dcm"
        bids_template: "sub-2_acq-ctankle_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "ctankle"
      - match_pattern: "VHF.*-Pelvis.*\\\\.dcm"
        bids_template: "sub-2_acq-ctpelvis_T1w.nii.gz"
        extract_entities:
          subject: "2"
          acquisition: "ctpelvis"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE OUTPUT 2: Multi-site Study (Directory-based grouping with NIfTI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

subject_grouping:
  method: directory_based
  extraction_pattern: "([A-Za-z]+)_sub(\\\\d+)"
  subject_from_group: 2
  site_from_group: 1
  
participant_metadata:
  "06272":
    site: "Cambridge"
  "82980":
    site: "Beijing"

standardization:
  apply: true
  strategy: "extract_site_to_column"
  reason: "Multi-site dataset with site prefixes in directory names"

subjects:
  labels: ["06272", "82980", "12345"]

assignment_rules:
  - subject: "06272"
    original: "Cambridge_sub06272"
    site: "Cambridge"
    match: ["**/Cambridge_sub06272/**"]

mappings:
  - modality: mri
    match: ["**/*mprage*.nii.gz", "**/*t1*.nii.gz"]
    format_ready: true
    convert_to: none
    filename_rules:
      - match_pattern: ".*anonymi[sz]ed.*"
        bids_template: "sub-{subject}_acq-anonymized_T1w.nii.gz"
      - match_pattern: ".*skull.*"
        bids_template: "sub-{subject}_acq-skullstripped_T1w.nii.gz"
      - match_pattern: ".*mprage.*"
        bids_template: "sub-{subject}_T1w.nii.gz"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE OUTPUT 3: Standard BIDS (Already compliant, no changes needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

subject_grouping:
  method: directory_based
  extraction_pattern: "sub-(\\\\d+)"
  subject_from_group: 1

standardization:
  apply: false
  reason: "Already in BIDS format"

subjects:
  labels: ["01", "02", "03"]

mappings:
  - modality: mri
    match: ["**/sub-*/anat/*.nii.gz"]
    format_ready: true
    convert_to: none
    filename_rules: []  # Keep original names

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. READ user_context.description FIRST - it's your primary guide
2. ANALYZE all_files to validate and discover patterns
3. DO NOT assume standard BIDS patterns - every dataset is unique
4. EXTRACT participant metadata from description (sex, age, group, site, etc.)
5. DESIGN filename_rules that handle ALL files (not just examples)
6. For CT scans: ALWAYS use T1w suffix, NEVER use CT suffix
7. For unusual datasets, BE CREATIVE in grouping strategy
8. If you cannot determine something, add a BLOCKING question

ENTITY EXTRACTION KEYWORDS (use these to analyze filenames):

Anatomical modalities (use EXACT BIDS suffix):
- T1w, T2w, FLAIR, PD, angio (NEVER: CT, ct, CTscan)
- For CT scans: ALWAYS use T1w suffix with acq-ct* label

Functional:
- BOLD, rest, task, fMRI

Diffusion:
- DWI, DTI, diffusion

Acquisition variants (map to acq-{variant}):
- Body parts: head, hip, shoulder, knee, ankle, pelvis, chest, abdomen
- For CT body parts: cthead, cthip, ctshoulder, ctknee, etc.
- Protocols: anonymized, skullstripped, normalized
- Contrasts: gad, contrast
- Sequences: mprage, space, spgr, epi

Run/Session indicators:
- run, r, repeat, session, ses, visit, timepoint

REMEMBER: 
- You have complete authority to design the conversion plan
- Python will execute YOUR decisions exactly as specified
- The goal is a valid BIDS dataset that preserves all information
- Every dataset is unique - design rules that fit THIS dataset
- CT scans MUST use T1w suffix, not CT

OUTPUT: Raw YAML only (no markdown fences, no extra text before or after)
"""

def llm_classify(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_CLASSIFICATION, payload, "Classification", temperature=0.15)

def llm_trio_dataset_description(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_DATASET_DESC, payload, "Trio_DatasetDesc", temperature=0.1)

def llm_trio_readme(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_README, payload, "Trio_README", temperature=0.4)

def llm_trio_participants(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_PARTICIPANTS, payload, "Trio_Participants", temperature=0.2)

def llm_nirs_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_DRAFT, payload, "NIRS_Draft", temperature=0.2)

def llm_nirs_normalize(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_NORMALIZE, payload, "NIRS_Normalize", temperature=0.1)

def llm_mri_voxel_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_DRAFT, payload, "MRI_Voxel_Draft", temperature=0.2)

def llm_mri_voxel_final(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_FINAL, payload, "MRI_Voxel_Final", temperature=0.1)

def llm_bids_plan(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_BIDS_PLAN, payload, "BIDSPlan", temperature=0.15)
