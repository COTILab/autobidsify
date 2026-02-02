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

# ============================================================================
# PROMPTS
# ============================================================================

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

╔═══════════════════════════════════════════════════════════════════════════╗
║ MISSION: Design a BIDS conversion plan by analyzing filename patterns    ║
║          and user descriptions to identify subject groupings              ║
╚═══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: READ USER DESCRIPTION
Look for explicit grouping information in user_hints.user_text:
- Number of groups mentioned
- Type of grouping (age, condition, timepoint, etc.)
- Key distinguishing terms

Step 2: ANALYZE FILENAME PATTERNS
Examine sample_files to find discriminative tokens:
- Look at token_positions to see which position varies
- Identify tokens that DISTINGUISH different groups
- Common patterns: age codes (neo, 1yr, 2yr), IDs (001, 002), conditions (pre, post)

Step 3: CROSS-VALIDATE
Compare user description with filename patterns:
- Do the patterns match the user's description?
- How many unique discriminative tokens exist?

Step 4: GENERATE ASSIGNMENT RULES
Create rules using EXACT tokens from filenames (NOT semantic interpretations)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: EXACT TOKEN MATCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When generating assignment_rules, use the EXACT tokens from sample_files:

✓ CORRECT (uses actual filename tokens):
  sample_files: ["infant-neo-aal.nii", "infant-1yr-aal.nii"]
  
  assignment_rules:
    - subject: 'neo'
      original: 'neo'           # ← EXACT token from filename
      match: ['*neo*']
    - subject: '1yr'
      original: '1yr'           # ← EXACT token from filename
      match: ['*1yr*']

✗ WRONG (uses semantic interpretation):
  assignment_rules:
    - subject: 'neo'
      original: 'neonate'       # ← NOT in filenames!
      match: ['*neo*']          # ← Pattern is right but 'original' is wrong

WHY THIS MATTERS:
The executor uses BOTH 'match' and 'original' for matching:
1. First tries 'match' patterns (glob-style)
2. Falls back to 'original' (substring search)

Both must contain actual filename tokens, not semantic names!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE ANALYSIS WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given input:
  user_text: "Atlas with 3 age groups: neonates, 1-year-olds, 2-year-olds"
  sample_files: [
    "infant-neo-aal.nii",
    "infant-neo-seg-gm.nii",
    "infant-1yr-aal.nii",
    "infant-1yr-seg-gm.nii",
    "infant-2yr-aal.nii",
    "infant-2yr-seg-gm.nii"
  ]
  token_positions: {
    "0": {"infant": 27},
    "1": {"neo": 9, "1": 9, "2": 9}
  }

REASONING:
1. User says "3 age groups" → expect 3 subjects
2. Position 0: all files have "infant" (common prefix, NOT discriminative)
3. Position 1: three tokens appear equally (neo=9, 1=9, 2=9)
4. Position 1 tokens are DISCRIMINATIVE → these define subjects
5. Actual tokens in files: "neo", "1yr", "2yr" (from sample_files)

OUTPUT:
subjects:
  labels: ['neo', '1yr', '2yr']  # Use actual tokens
  count: 3

assignment_rules:
  - subject: 'neo'
    original: 'neo'        # ← EXACT token
    match: ['*neo*']
  - subject: '1yr'
    original: '1yr'        # ← EXACT token (includes 'yr')
    match: ['*1yr*']
  - subject: '2yr'
    original: '2yr'        # ← EXACT token (includes 'yr')
    match: ['*2yr*']

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON PATTERNS AND HOW TO HANDLE THEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pattern 1: Age/Condition Groups (Flat structure)
─────────────────────────────────────────────────
Files: template-young-*.nii, template-old-*.nii
→ 2 subjects: 'young' and 'old'
→ Use exact tokens: original: 'young', match: ['*young*']

Pattern 2: Numeric Patient IDs
───────────────────────────────
Files: patient001-T1.nii, patient002-T1.nii
→ Many subjects (001, 002, ...)
→ Use exact IDs: original: '001', match: ['*001*']

Pattern 3: Site-Subject Structure
──────────────────────────────────
Files: Beijing_sub001/anat/scan.nii
→ Directory-based
→ Use directory: original: 'Beijing_sub001', match: ['**/Beijing_sub001/**']

Pattern 4: Single Template/Atlas
─────────────────────────────────
Files: MNI152_T1_1mm.nii, MNI152_T1_2mm.nii
→ 1 subject (different resolutions of same template)
→ Use template name: original: 'MNI152', match: ['*MNI152*']

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

subjects:
  labels: [list of subject IDs using EXACT filename tokens]
  count: total number
  source: llm_analysis

assignment_rules:
  - subject: 'exact_token'     # Subject ID (from filename)
    original: 'exact_token'    # MUST be exact token from filename
    match: ['*exact_token*']   # Glob pattern using same token

mappings:
  - modality: mri
    match: ['*.nii', '*.nii.gz', '**/*.nii', '**/*.nii.gz']
    format_ready: true
    convert_to: none
    filename_rules:
      - match_pattern: '.*pattern.*'
        bids_template: 'sub-SUBJECTID_suffix.nii.gz'

CRITICAL RULES:
1. 'original' field = EXACT token from actual filenames
2. 'match' patterns = glob patterns using SAME exact token
3. Use 'sub-X' in bids_template where X will be replaced with subject ID
4. mappings.match must include BOTH root and nested patterns:
   ['*.nii', '**/*.nii'] not just ['**/*.nii']

OUTPUT: Raw YAML only (no markdown fences, no extra text)

Now analyze the evidence and generate the plan.
"""


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

# ============================================================================
# LLM CALL FUNCTIONS
# ============================================================================

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
