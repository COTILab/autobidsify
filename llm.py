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

# Keep existing prompts unchanged
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

# UPDATED PROMPT with filename token analysis support
PROMPT_BIDS_PLAN = """You are a BIDS dataset architect with complete decision-making authority.

═══════════════════════════════════════════════════════════════════════════
MISSION: Design a complete BIDS conversion plan for ANY neuroimaging dataset
═══════════════════════════════════════════════════════════════════════════

CRITICAL YAML ESCAPING:
- Use DOUBLE backslashes in regex: \\\\d \\\\w \\\\s (NOT \\d \\w \\s)

INPUT STRUCTURE:
{
  "file_count": total_files,
  "counts_by_ext": {".dcm": 2979, ...},
  "sample_files": [representative file paths],
  
  "user_hints": {
    "n_subjects": number or null,
    "user_text": "User's description",
    "modality_hint": "mri|nirs|mixed"
  },
  
  "subject_summary": {
    "total_subjects": number (from directory analysis),
    "pattern_types": ["site_prefixed", ...],
    "pattern_examples": {...}
  },
  
  "filename_token_analysis": {
    "total_files": 2979,
    "dominant_prefixes": [
      {"prefix": "VHM", "count": 1500, "percentage": 50.4},
      {"prefix": "VHF", "count": 1479, "percentage": 49.6}
    ],
    "token_frequency": {"VHM": 1500, "CT": 2979, ...},
    "insights": [
      "Two major prefixes detected: 'VHM' (50.4%) and 'VHF' (49.6%)"
    ]
  },
  
  "analysis_decision": {
    "use_filename_analysis": true/false,
    "path_subjects": number,
    "filename_subjects": number
  }
}

═══════════════════════════════════════════════════════════════════════════
CRITICAL: HANDLING FLAT VS HIERARCHICAL STRUCTURES
═══════════════════════════════════════════════════════════════════════════

SCENARIO 1: Flat structure (all files in one directory)
  - subject_summary.total_subjects = 0
  - filename_token_analysis.dominant_prefixes = ["VHM", "VHF"]
  - analysis_decision.use_filename_analysis = true
  
  ACTION: Use filename_token_analysis to determine subjects!
  
  Example:
    Filename samples: ["VHMCT1mm-Hip (134).dcm", "VHFCT1mm-Head (89).dcm"]
    Dominant prefixes: VHM (50%), VHF (50%)
    User hint: n_subjects = 2
    
    CONCLUSION: VHM = subject "1", VHF = subject "2"
    
    Output:
      subject_grouping:
        method: filename_prefix
        rules:
          - prefix: "VHM"
            maps_to_subject: "1"
            match_pattern: "VHM.*"
            metadata:
              sex: "M"
          - prefix: "VHF"
            maps_to_subject: "2"
            match_pattern: "VHF.*"
            metadata:
              sex: "F"

SCENARIO 2: Hierarchical structure (subjects have directories)
  - subject_summary.total_subjects > 0
  - subject_summary.pattern_types = ["site_prefixed"]
  - analysis_decision.use_filename_analysis = false
  
  ACTION: Use subject_summary (Python already detected subjects)
  
  Example:
    Pattern examples: ["Cambridge_sub06272", "Beijing_sub82980"]
    
    Python already created assignment_rules, just validate them!

═══════════════════════════════════════════════════════════════════════════
STEP 1: DETERMINE SUBJECT GROUPING METHOD
═══════════════════════════════════════════════════════════════════════════

Check analysis_decision.use_filename_analysis:

IF TRUE (flat structure):
  - Analyze filename_token_analysis.dominant_prefixes
  - Match prefix count with user_hints.n_subjects
  - Design filename_prefix grouping rules
  - Extract participant metadata from user_hints.user_text

IF FALSE (hierarchical):
  - Validate subject_summary.pattern_examples
  - Python already provided grouping via assignment_rules
  - Just add participant_metadata if available

═══════════════════════════════════════════════════════════════════════════
CT SCAN HANDLING (unchanged from before)
═══════════════════════════════════════════════════════════════════════════

✓ For CT scans, ALWAYS use "T1w" suffix
✓ Use acquisition label: acq-ct, acq-cthip, acq-cthead
✗ NEVER use CT, ct, CTscan as modality suffix

Example:
  Input:  "VHMCT1mm-Hip (134).dcm"
  Output: "sub-1_acq-cthip_T1w.nii.gz"

═══════════════════════════════════════════════════════════════════════════
OUTPUT STRUCTURE
═══════════════════════════════════════════════════════════════════════════

subject_grouping:
  method: filename_prefix | directory_based | filename_pattern
  description: "Explanation"
  rules: [...]

participant_metadata:
  "1":
    sex: "M"
    group: "cadaver"
  "2":
    sex: "F"
    group: "cadaver"

standardization:
  apply: true/false
  strategy: "..."
  reason: "..."

subjects:
  labels: ["1", "2", ...]

assignment_rules:
  - subject: "1"
    original: "VHM" | "Cambridge_sub06272"
    match: ["**/VHM*"] | ["**/Cambridge_sub06272/**"]

mappings:
  - modality: mri
    match: ["**/*.dcm"]
    format_ready: false
    convert_to: "dicom_to_nifti"
    filename_rules:
      - match_pattern: "VHM.*-Hip.*\\\\.dcm"
        bids_template: "sub-1_acq-cthip_T1w.nii.gz"

OUTPUT: Raw YAML only (no markdown fences, no extra text)
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
