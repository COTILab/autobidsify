# llm.py
# Unified LLM caller supporting OpenAI and Qwen (Ollama / REST API / DashScope).

import os
import json
from typing import Any, Optional
from autobidsify.utils import warn, fatal, info


# ============================================================================
# Exception
# ============================================================================

class LLMHardFail(Exception):
    def __init__(self, step: str, error_type: str, message: str):
        self.step       = step
        self.error_type = error_type
        self.message    = message
        super().__init__(f"[{step}] {error_type}: {message}")


# ============================================================================
# Provider detection
# ============================================================================

def is_qwen_model(model: str) -> bool:
    """Return True for any Qwen model (qwen*)."""
    return model.lower().startswith('qwen')


def is_openai_model(model: str) -> bool:
    """Return True for OpenAI models (gpt*, o1*, o3*)."""
    return model.lower().startswith(('gpt', 'o1', 'o3'))


def is_reasoning_model(model: str) -> bool:
    """Return True for reasoning-style models that skip temperature."""
    m = model.lower()
    return m.startswith("o1") or m.startswith("o3") or m.startswith("gpt-5")


# ============================================================================
# OpenAI
# ============================================================================

def _get_openai_client():
    """Build and return an OpenAI client.  Calls fatal() on missing key."""
    try:
        from openai import OpenAI
    except ImportError:
        fatal("openai library not installed.  Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fatal(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "If you want to use Qwen instead, pass --model qwen3-coder-next:latest "
            "(or any qwen* model name)."
        )

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise LLMHardFail("Initialization", "ClientError", str(e))


def _call_openai(model: str, system_prompt: str, user_payload: str,
                 step: str, temperature: Optional[float] = None) -> str:
    """Call the OpenAI chat-completions endpoint."""
    client = _get_openai_client()

    try:
        from openai import OpenAIError
    except ImportError:
        fatal("openai library not installed")

    try:
        params: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload},
            ],
        }
        if is_reasoning_model(model):
            params["max_completion_tokens"] = 32000
        else:
            params["max_completion_tokens"] = 16000
            if temperature is not None:
                params["temperature"] = temperature

        response = client.chat.completions.create(**params)

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if content and content.strip():
                    return content.strip()
                raise LLMHardFail(step, "EmptyResponse", "OpenAI returned empty content")
            raise LLMHardFail(step, "InvalidResponse", "Response has no message.content")
        raise LLMHardFail(step, "InvalidResponse", "Response has no choices")

    except OpenAIError as e:
        raise LLMHardFail(step, e.__class__.__name__, str(e))
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "UnexpectedError", str(e))


# ============================================================================
# Qwen — local Ollama package
# ============================================================================

def _call_qwen_ollama(model: str, system_prompt: str, user_payload: str,
                      step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via the local Ollama Python package.

    Requires:
      pip install ollama
      ollama serve
      ollama pull <model>
    """
    try:
        import ollama
    except ImportError:
        raise LLMHardFail(step, "OllamaNotInstalled",
                          "ollama library not installed.  Run: pip install ollama")

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ]
        options: dict = {"top_p": 0.8}
        if temperature is not None:
            options["temperature"] = temperature

        response = ollama.chat(model=model, messages=messages, options=options)

        # ollama >= 0.6: object-style response
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse", "Qwen (Ollama) returned empty content")

        # ollama < 0.6: dict-style response
        if isinstance(response, dict) and "message" in response:
            content = response["message"].get("content", "")
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse", "Qwen (Ollama) returned empty content")

        raise LLMHardFail(
            step, "InvalidResponse",
            f"Unexpected Ollama response type: {type(response)}"
        )

    except ImportError:
        raise
    except LLMHardFail:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg:
            raise LLMHardFail(step, "OllamaNotRunning",
                              "Cannot connect to Ollama.  Run: ollama serve")
        if "not found" in msg or "pull" in msg:
            raise LLMHardFail(step, "ModelNotFound",
                              f"Model '{model}' not found.  Run: ollama pull {model}")
        raise LLMHardFail(step, "QwenError", str(e))


# ============================================================================
# Qwen — remote Ollama REST API
# ============================================================================

def _call_qwen_rest_api(model: str, system_prompt: str, user_payload: str,
                        step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via a remote Ollama REST API endpoint.

    No local Ollama installation required.  Only needs the requests library.

    Setup:
      export OLLAMA_BASE_URL=http://your-server.com:11434
    """
    try:
        import requests
    except ImportError:
        raise LLMHardFail(step, "RequestsNotInstalled",
                          "requests library not installed.  Run: pip install requests")

    base_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    if not base_url:
        raise LLMHardFail(step, "MissingOllamaBaseURL",
                          "OLLAMA_BASE_URL is not set.")

    payload: dict = {
        "model":   model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ],
        "stream":  False,
        "options": {"top_p": 0.8},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        if content and content.strip():
            return content.strip()
        raise LLMHardFail(step, "EmptyResponse",
                          "Ollama REST API returned empty content")
    except requests.exceptions.ConnectionError:
        raise LLMHardFail(step, "OllamaRESTUnreachable",
                          f"Cannot reach Ollama REST API at {base_url}.")
    except requests.exceptions.Timeout:
        raise LLMHardFail(step, "OllamaRESTTimeout",
                          f"Ollama REST API timed out at {base_url}.")
    except requests.exceptions.HTTPError as e:
        raise LLMHardFail(step, "OllamaRESTHTTPError", str(e))
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "OllamaRESTError", str(e))


# ============================================================================
# Qwen — DashScope cloud API
# ============================================================================

def _call_qwen_api(model: str, system_prompt: str, user_payload: str,
                   step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via the Alibaba DashScope cloud API.

    Setup:
      pip install dashscope
      export DASHSCOPE_API_KEY='sk-...'
    """
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        raise LLMHardFail(step, "DashScopeNotInstalled",
                          "dashscope not installed.  Run: pip install dashscope")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise LLMHardFail(step, "MissingAPIKey",
                          "DASHSCOPE_API_KEY is not set.  "
                          "Get one at: https://dashscope.console.aliyun.com/")

    dashscope.api_key = api_key

    model_mapping = {
        "qwen-max":     "qwen-max",
        "qwen-plus":    "qwen-plus",
        "qwen-turbo":   "qwen-turbo",
        "qwen2.5-coder": "qwen-coder-plus",
    }
    ds_model = model_mapping.get(model, model)

    try:
        response = Generation.call(
            model=ds_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload},
            ],
            result_format="message",
            temperature=temperature if temperature is not None else 0.85,
            top_p=0.8,
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse",
                              "DashScope returned empty content")
        raise LLMHardFail(step, "APIError",
                          f"DashScope error {response.code}: {response.message}")
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "QwenAPIError", str(e))


# ============================================================================
# Qwen dispatcher
# ============================================================================

def _call_qwen(model: str, system_prompt: str, user_payload: str,
               step: str, temperature: Optional[float] = None) -> str:
    """
    Route a Qwen call to the appropriate backend.

    Priority:
      1. OLLAMA_BASE_URL set          → remote Ollama REST API
      2. ollama Python package usable → local Ollama process
      3. DASHSCOPE_API_KEY set        → DashScope cloud API
      4. Nothing available            → fatal with clear instructions

    IMPORTANT: if none of the above conditions are met, the function calls
    fatal() with a Qwen-specific message.  This function is ONLY reached
    when is_qwen_model() returns True — it is never called for OpenAI models.
    """
    # Priority 1 ── Remote Ollama REST API
    if os.getenv("OLLAMA_BASE_URL"):
        info(f"Using remote Ollama REST API: {os.getenv('OLLAMA_BASE_URL')}")
        return _call_qwen_rest_api(model, system_prompt, user_payload, step, temperature)

    # Priority 2 ── Local Ollama Python package
    try:
        return _call_qwen_ollama(model, system_prompt, user_payload, step, temperature)
    except LLMHardFail as e:
        if e.error_type not in ("OllamaNotInstalled", "OllamaNotRunning", "ModelNotFound"):
            raise   # unexpected error — re-raise as-is

        # Priority 3 ── DashScope cloud API
        if os.getenv("DASHSCOPE_API_KEY"):
            warn(f"Ollama not available ({e.error_type}), falling back to DashScope API...")
            return _call_qwen_api(model, system_prompt, user_payload, step, temperature)

        # Priority 4 ── Nothing available
        fatal(
            f"Cannot call Qwen model '{model}' — no backend is available.\n"
            "\n"
            "Choose one of the following options:\n"
            "\n"
            "  Option 1 — Remote Ollama REST API (no local install required)\n"
            "    export OLLAMA_BASE_URL=http://your-server.com:11434\n"
            "\n"
            "  Option 2 — Local Ollama\n"
            "    ollama serve\n"
            f"    ollama pull {model}\n"
            "\n"
            "  Option 3 — DashScope cloud API\n"
            "    export DASHSCOPE_API_KEY='sk-...'\n"
            "    (get a key at https://dashscope.console.aliyun.com/)\n"
            "\n"
            "If you intended to use OpenAI instead, pass --model gpt-4o\n"
            "and make sure OPENAI_API_KEY is set."
        )


# ============================================================================
# Temperature inference for Qwen
# ============================================================================

def _infer_qwen_temperature(model: str,
                            base_temperature: Optional[float]) -> Optional[float]:
    """
    Adjust temperature for Qwen based on model-name keywords.

    Rules:
      think / careful / compare / reason  → cap at 0.15  (reasoning model)
      next  / fast    / turbo   / lite    → floor at 0.4  (speed model)
      everything else                     → floor at 0.3  (avoid too-low temps)
    """
    if base_temperature is None:
        return None

    m = model.lower()

    if any(kw in m for kw in ("think", "careful", "compare", "reason")):
        return min(base_temperature, 0.15)

    if any(kw in m for kw in ("next", "fast", "turbo", "lite")):
        return max(base_temperature, 0.4)

    return max(base_temperature, 0.3)


# ============================================================================
# Unified entry point
# ============================================================================

def _call_llm(model: str, system_prompt: str, user_payload: str,
              step: str, temperature: Optional[float] = None) -> str:
    """
    Route an LLM call to the correct provider based on model name.

    Routing:
      qwen*        → _call_qwen()    (Ollama / REST API / DashScope)
      gpt* o1* o3* → _call_openai() (OpenAI API)
      anything else → LLMHardFail(UnknownProvider)

    The two providers are completely independent — using a Qwen model
    never touches OPENAI_API_KEY, and vice versa.
    """
    if is_qwen_model(model):
        temp = _infer_qwen_temperature(model, temperature)
        return _call_qwen(model, system_prompt, user_payload, step, temp)

    if is_openai_model(model):
        return _call_openai(model, system_prompt, user_payload, step, temperature)

    raise LLMHardFail(
        step, "UnknownProvider",
        f"Unrecognised model name: '{model}'.\n"
        "  OpenAI models start with: gpt, o1, o3\n"
        "  Qwen models start with:   qwen\n"
        "Examples: --model gpt-4o   or   --model qwen3-coder-next:latest"
    )


# ============================================================================
# Prompts
# ============================================================================

PROMPT_TRIO_DATASET_DESC = """You are a BIDS dataset_description.json metadata extractor.

═══════════════════════════════════════════════════════
YOUR JOB
═══════════════════════════════════════════════════════

Extract dataset metadata from the input. Return ONLY valid JSON, no markdown.

═══════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════

1. LICENSE — output as "raw_license" (plain string, NOT normalized):
   - Copy exactly what the user wrote, e.g. "CC0", "CC BY 4.0",
     "Creative Commons Zero", "public domain", "MIT license"
   - Do NOT try to normalize or format it — Python will do that
   - If the user wrote "License: CC0" → raw_license: "CC0"
   - If the document says "released under Creative Commons" → raw_license: "Creative Commons"
   - If no license mentioned anywhere → omit raw_license

2. AUTHORS — extract from ALL available sources:
   - Search in order: user_hints.user_text → documents[]
   - Look for: explicit author lists, citation patterns, "Created by",
     "Principal Investigator", "Contact", "Contributors" sections
   - If full names are available, use them: ["Last FM", "Last FM"]
   - If only "et al." citation exists, keep first author + et al.: ["Shafto MA et al."]
   - Do NOT infer, guess, or use outside knowledge to expand author lists
   - Do NOT fabricate names not present in any input source
   - If no author information found anywhere, omit Authors field entirely

   EXAMPLES (follow exactly):

   Input: "Smith et al. (2023). A neuroimaging study..."
   Output: "Authors": ["Smith et al."]

   Input: "Created by John Doe, Jane Smith and Bob Lee"
   Output: "Authors": ["John Doe", "Jane Smith", "Bob Lee"]

   Input: "Data collected by the CamCAN team. Contact: info@cam.ac.uk"
   Output: (omit Authors field)

   Input: "Shafto et al. (2014). The Cambridge Centre for Ageing..."
   Output: "Authors": ["Shafto et al."]

3. NAME — infer from context:
   - Look for explicit dataset name in user_hints.user_text
   - If not found, infer from the scientific context
   - Keep it short and descriptive

4. MISSING FIELDS — omit rather than guess:
   - If you cannot determine a field with reasonable confidence, omit it
   - Never invent information not present in the input

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════

{
  "dataset_description": {
    "Name": "...",
    "BIDSVersion": "1.10.0",
    "DatasetType": "raw",
    "Authors": ["First Last", "First Last"]
  },
  "raw_license": "CC0",
  "extraction_log": {
    "Name": "inferred from user_text: '...'",
    "raw_license": "found in user_text: 'License: CC0'",
    "Authors": "extracted from citation in user_text"
  },
  "questions": []
}

Notes:
- raw_license goes at the TOP LEVEL (not inside dataset_description)
- dataset_description should NOT contain a "License" field — Python adds it after normalization
- BIDSVersion must always be "1.10.0"
- DatasetType must always be "raw"
- Output ONLY valid JSON, no extra text, no markdown fences

FIELD SOURCE RULES (STRICT - violations cause data integrity failure):
┌─────────────────┬────────────────────────────────────────────────────┐
│ Field           │ Allowed sources                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│ Authors         │ user_hints.user_text or documents[] ONLY           │
│                 │ NEVER use training knowledge to expand et al.      │
│ raw_license     │ user_hints.user_text or documents[] ONLY           │
│ Name            │ may infer from context if not explicit             │
│ BIDSVersion     │ always "1.10.0" (fixed)                            │
│ DatasetType     │ always "raw" (fixed)                               │
└─────────────────┴────────────────────────────────────────────────────┘
"""

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

═══════════════════════════════════════════════════════════════════════
SUPPORTED FORMATS AND CONVERSION RULES (v10 - CRITICAL)
═══════════════════════════════════════════════════════════════════════

MRI FORMATS (modality: mri):
  Input formats:
    • DICOM (.dcm)           → Convert to NIfTI using dcm2niix
    • NIfTI (.nii, .nii.gz)  → Already BIDS-ready, copy directly
    • JNIfTI (.jnii, .bnii)  → Convert to NIfTI using jnifti_converter

  BIDS output: .nii.gz files only

fNIRS FORMATS (modality: nirs):
  Input formats:
    • SNIRF (.snirf)         → Already BIDS-ready, copy directly
    • Homer3 (.nirs)         → Convert to SNIRF
    • MATLAB (.mat)          → Convert to SNIRF

  BIDS output: .snirf files only

═══════════════════════════════════════════════════════════════════════
CRITICAL RULES FOR .mat FILES
═══════════════════════════════════════════════════════════════════════

.mat files are AMBIGUOUS - they can contain EITHER MRI or fNIRS data!

Decision logic:
1. Check user_hints.modality_hint:
   - If "nirs" → treat as fNIRS, set convert_to: "snirf"
   - If "mri" → treat as MRI voxel data (rare, needs special handling)

2. Check user_hints.user_text for keywords:
   - Keywords indicating fNIRS: "fNIRS", "NIRS", "optodes", "channels", "oxygenation"
   - Keywords indicating MRI: "MRI", "anatomical", "voxels", "brain volume"

3. Default assumption:
   - If modality_hint = "nirs" → .mat is fNIRS data
   - Otherwise → ask user for clarification

═══════════════════════════════════════════════════════════════════════
FORMAT_READY AND CONVERT_TO RULES
═══════════════════════════════════════════════════════════════════════

format_ready: true  → .nii/.nii.gz (MRI) or .snirf (fNIRS) — copy directly
format_ready: false → needs conversion:
  .dcm / .jnii / .bnii → convert_to: nifti
  .mat / .nirs         → convert_to: snirf
convert_to: "none"   → only when format_ready: true

═══════════════════════════════════════════════════════════════════════
SUBJECT ID STRATEGY
═══════════════════════════════════════════════════════════════════════

CRITICAL: In assignment_rules, 'subject' must be the BARE ID without 'sub-' prefix.

✓ CORRECT:  subject: '1'      → executor creates sub-1
✗ WRONG:    subject: 'sub-1'  → executor creates sub-sub-1

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

subjects:
  labels: [list of BIDS IDs without 'sub-' prefix]
  count: N
  source: llm_analysis
  id_strategy: numeric / semantic

assignment_rules:
  - subject: 'bids_id'
    original: 'token'
    match: ['*token*']

participant_metadata:
  'bids_id':
    original_id: 'xxx'
    site: 'xxx'
    sex: 'M'

mappings:
  - modality: mri
    match: ['*.nii.gz', '**/*.dcm']
    exclude: []
    format_ready: true
    convert_to: none
    filename_rules:
      - match_pattern: '.*T1.*'
        bids_template: 'sub-X_T1w.nii.gz'

OUTPUT: Raw YAML only (no markdown, no explanation)
"""


# ============================================================================
# Public LLM call wrappers
# ============================================================================

def llm_trio_dataset_description(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_DATASET_DESC, payload,
                     "Trio_DatasetDesc", temperature=0.1)

def llm_trio_readme(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_README, payload,
                     "Trio_README", temperature=0.4)

def llm_trio_participants(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_TRIO_PARTICIPANTS, payload,
                     "Trio_Participants", temperature=0.2)

def llm_nirs_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_DRAFT, payload,
                     "NIRS_Draft", temperature=0.2)

def llm_nirs_normalize(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_NORMALIZE, payload,
                     "NIRS_Normalize", temperature=0.1)

def llm_mri_voxel_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_DRAFT, payload,
                     "MRI_Voxel_Draft", temperature=0.2)

def llm_mri_voxel_final(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_FINAL, payload,
                     "MRI_Voxel_Final", temperature=0.1)

def llm_bids_plan(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_BIDS_PLAN, payload,
                     "BIDSPlan", temperature=0.15)