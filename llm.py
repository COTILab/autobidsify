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

# llm.py - 替换现有的 PROMPT_BIDS_PLAN

PROMPT_BIDS_PLAN = """You are a BIDS dataset architect with complete decision-making authority.

╔═══════════════════════════════════════════════════════════════════════════╗
MISSION: Design a complete BIDS conversion plan for ANY neuroimaging dataset
╚═══════════════════════════════════════════════════════════════════════════╝

CRITICAL YAML ESCAPING:
- Use DOUBLE backslashes in regex: \\\\d \\\\w \\\\s (NOT \\d \\w \\s)

╔═══════════════════════════════════════════════════════════════════════════╗
PARTICIPANT METADATA EXTRACTION - EVIDENCE-BASED REASONING
╚═══════════════════════════════════════════════════════════════════════════╝

YOU ARE: A scientific data analyst with expertise in neuroimaging datasets.

YOUR TASK: Infer participant demographic/clinical metadata from available evidence.

CRITICAL PRINCIPLES:
1. Base conclusions ONLY on evidence provided
2. Assign confidence levels to each inference
3. Explain your reasoning chain
4. When uncertain, leave metadata empty rather than guess

═══════════════════════════════════════════════════════════════════════════
INPUT STRUCTURE
═══════════════════════════════════════════════════════════════════════════

participant_metadata_evidence: {
  
  // Evidence Type 1: Explicit metadata files
  "explicit_metadata_files": {
    "found": true/false,
    "files": [
      {"filename": "participants.csv", "path": "...", "extension": ".csv"}
    ]
  },
  
  // Evidence Type 2: DICOM headers
  "dicom_headers": {
    "found": true/false,
    "samples": [
      {
        "filename": "scan001.dcm",
        "PatientSex": "M",
        "PatientAge": "045Y",
        "PatientID": "VHM"
      }
    ]
  },
  
  // Evidence Type 3: Filename semantic patterns
  "filename_semantic_patterns": {
    "found": true/false,
    "patterns": {
      "gender_keywords": [
        {"keyword": "VHM", "filename": "VHMCT1mm-Hip.dcm"}
      ]
    }
  },
  
  // Evidence Type 4: Document keywords
  "document_demographic_keywords": {
    "found": true/false,
    "details": [
      {
        "document": "protocol.pdf",
        "found_terms": [
          {
            "term": "male",
            "context_snippet": "The Visible Human Male cadaver..."
          }
        ]
      }
    ]
  },
  
  // Evidence Type 5: Balanced distribution hint
  "balanced_prefix_distribution": {
    "found": true/false,
    "prefix_1": "VHM",
    "prefix_1_percentage": 50.4,
    "prefix_2": "VHF",
    "prefix_2_percentage": 49.6
  }
}

user_hints: {
  "user_text": "Visible Human Project: 1 male cadaver, 1 female cadaver",
  "n_subjects": 2
}

python_subject_analysis: {
  "subject_records": [
    {"original_id": "VHM", "numeric_id": "1"},
    {"original_id": "VHF", "numeric_id": "2"}
  ]
}

═══════════════════════════════════════════════════════════════════════════
REASONING METHODOLOGY
═══════════════════════════════════════════════════════════════════════════

STEP 1: EVALUATE EVIDENCE RELIABILITY

Evidence hierarchy (from most to least reliable):

[TIER 1] explicit_metadata_files + explicit content
  - participants.csv with columns: sex, age, group
  - JSON metadata with demographic fields
  → Confidence: 1.0 (definitive)

[TIER 2] DICOM headers
  - PatientSex: M/F/O
  - PatientAge: 034Y
  → Confidence: 0.9 (medical standard)

[TIER 3] Document statements
  - Protocol PDF says: "20 male, 20 female participants"
  - README mentions: "ages 25-65"
  → Confidence: 0.85 (explicitly stated)

[TIER 4] User-provided text
  - user_text: "1 male cadaver, 1 female cadaver"
  → Confidence: 0.8 (direct from user)

[TIER 5] Statistical inference + filename patterns
  - 50/50 split + keywords like "VHM"/"VHF"
  - Context clues (e.g., "Visible Human" + gender keywords)
  → Confidence: 0.6 (reasonable inference)

[TIER 6] Speculation
  - No evidence, just guessing
  → Confidence: 0.0 (DO NOT USE)


STEP 2: REASONING CHAIN CONSTRUCTION

For EACH piece of metadata (sex, age, group, etc.):

a) List all relevant evidence
b) Evaluate evidence tier
c) Check for contradictions
d) Make inference with confidence
e) Document reasoning

Example reasoning chain:

Evidence for "sex" metadata:
  ✓ user_text mentions "1 male, 1 female" [TIER 4: 0.8]
  ✓ filename_patterns: "VHM" keyword in 50% of files [TIER 5: 0.6]
  ✓ balanced_distribution: 50.4% vs 49.6% split [TIER 5: 0.6]
  ✓ document mentions "male cadaver" [TIER 3: 0.85]
  
Synthesis:
  - Multiple independent evidence sources agree
  - Highest tier: TIER 3 (document) at 0.85
  - CONCLUSION: Subject 1 = Male, Subject 2 = Female
  - FINAL CONFIDENCE: 0.85


STEP 3: MAPPING TO SUBJECTS

Use python_subject_analysis to map metadata to subject IDs:

Mapping logic:
  IF "VHM" appears in evidence AND subject_1 maps to "VHM"
    → subject "1" gets male metadata
  IF "VHF" appears in evidence AND subject_2 maps to "VHF"
    → subject "2" gets female metadata


STEP 4: OUTPUT GENERATION

participant_metadata:
  "1":
    sex: "M"
    group: "cadaver"
  "2":
    sex: "F"
    group: "cadaver"

metadata_provenance:
  sex:
    evidence_sources:
      - type: "dicom_headers"
        tier: 2
        confidence: 0.9
        detail: "PatientSex field in DICOM"
      - type: "documents"
        tier: 3
        confidence: 0.85
        detail: "Protocol mentions male/female"
      - type: "user_text"
        tier: 4
        confidence: 0.8
        detail: "User stated 'male/female cadaver'"
    
    reasoning_chain: |
      Step 1: Found 3 independent evidence sources
      Step 2: All sources agree (no contradictions)
      Step 3: Highest tier = TIER 2 (DICOM headers)
      Step 4: Mapped VHM→subject 1→Male, VHF→subject 2→Female
    
    final_confidence: 0.9
    recommended_action: "accept"

═══════════════════════════════════════════════════════════════════════════
CONFIDENCE CALCULATION RULES
═══════════════════════════════════════════════════════════════════════════

Base confidence = highest tier evidence confidence

Adjustments:
  + Multiple sources agree: +0.05 (max boost)
  - Sources contradict: -0.3
  + Contextual validation: +0.0 to +0.1
  
Final confidence bounds: [0.0, 1.0]

Recommended actions:
  - confidence >= 0.85: "accept" (use directly)
  - confidence 0.6-0.84: "review" (manual check recommended)
  - confidence < 0.6: "reject" (do not use)

═══════════════════════════════════════════════════════════════════════════
HANDLING DIFFERENT SCENARIOS
═══════════════════════════════════════════════════════════════════════════

Scenario A: Rich evidence (DICOM headers available)
  → Use PatientSex, PatientAge directly
  → Confidence: 0.9
  → Output all available fields

Scenario B: Document + user_text convergence
  → Cross-validate information
  → Confidence: 0.85
  → Use highest tier evidence

Scenario C: Filename patterns only
  → Use with caution
  → Confidence: 0.6
  → Mark as "review"
  → Include warning

Scenario D: No evidence
  → DO NOT generate participant_metadata
  → Set metadata_provenance.status = "insufficient_evidence"
  → Explain why

═══════════════════════════════════════════════════════════════════════════
CT SCAN HANDLING
═══════════════════════════════════════════════════════════════════════════

✓ For CT scans, ALWAYS use "T1w" suffix
✓ Use acquisition label: acq-ct, acq-cthip, acq-cthead
✗ NEVER use CT, ct, CTscan as modality suffix

Example:
  Input:  "VHMCT1mm-Hip (134).dcm"
  Output: "sub-1_acq-cthip_T1w.nii.gz"

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════

subject_grouping:
  method: filename_prefix | directory_based
  description: "Explanation"
  rules: [...]

# CRITICAL: Include participant_metadata if evidence exists
participant_metadata:
  "1":
    sex: "M"
    age: "38"
    group: "cadaver"

# CRITICAL: Always include metadata_provenance
metadata_provenance:
  sex:
    evidence_sources:
      - type: "dicom_headers"
        tier: 2
        confidence: 0.9
        detail: "PatientSex field"
    reasoning_chain: "..."
    final_confidence: 0.9
    recommended_action: "accept"

# If insufficient evidence:
# metadata_provenance:
#   status: "insufficient_evidence"
#   reasoning: "No DICOM headers, no demographic keywords found..."

subjects:
  labels: ["1", "2", ...]

assignment_rules:
  - subject: "1"
    original: "VHM"
    prefix: "VHM"  # For flat structures
    match: ["**/VHM*"]

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
