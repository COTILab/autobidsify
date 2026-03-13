import os
import json
from typing import Any, Optional
from autobidsify.utils import warn, fatal, info

# ============================================================================
# Exception Classes
# ============================================================================

class LLMHardFail(Exception):
    def __init__(self, step: str, error_type: str, message: str):
        self.step = step
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{step}] {error_type}: {message}")


# ============================================================================
# Provider Detection
# ============================================================================

def is_qwen_model(model: str) -> bool:
    """Check if model is a Qwen model."""
    return model.startswith('qwen')


def is_openai_model(model: str) -> bool:
    """Check if model is an OpenAI model."""
    return model.startswith(('gpt', 'o1', 'o3'))


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o1/o3/gpt-5 series)."""
    return (
        model.startswith("o1") or 
        model.startswith("o3") or
        model.startswith("gpt-5")
    )


# ============================================================================
# OpenAI Client
# ============================================================================

def _get_openai_client():
    """Get OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        fatal("openai library not installed. Install with: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fatal("OPENAI_API_KEY not found in environment. Please set it.")
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise LLMHardFail("Initialization", "ClientError", str(e))


def _call_openai(model: str, system_prompt: str, user_payload: str, 
                step: str, temperature: Optional[float] = None) -> str:
    """Call OpenAI API."""
    client = _get_openai_client()
    
    try:
        from openai import OpenAIError
    except ImportError:
        fatal("openai library not installed")
    
    try:
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ]
        }
        
        # Reasoning models use different parameters
        if is_reasoning_model(model):
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
                    raise LLMHardFail(step, "EmptyResponse", "LLM returned empty content")
            else:
                raise LLMHardFail(step, "InvalidResponse", "Response has no message.content")
        else:
            raise LLMHardFail(step, "InvalidResponse", "Response has no choices")
        
    except OpenAIError as e:
        raise LLMHardFail(step, e.__class__.__name__, str(e))
    except Exception as e:
        raise LLMHardFail(step, "UnexpectedError", str(e))


# ============================================================================
# Qwen Client via Ollama
# ============================================================================

def _call_qwen_ollama(model: str, system_prompt: str, user_payload: str,
                     step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via Ollama (local deployment).
    
    Compatible with ollama >= 0.6 (object-style response).
    
    Requires:
    1. Ollama installed and running
    2. Model pulled: ollama pull qwen2.5-coder:7b
    3. Python package: pip install ollama
    """
    try:
        import ollama
    except ImportError:
        raise LLMHardFail(step, "OllamaNotInstalled",
                         "ollama library not installed. Install with: pip install ollama")
    
    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_payload}
        ]
        
        # options = {}
        options = {'top_p': 0.8}
        if temperature is not None:
            options['temperature'] = temperature
        
        # Call Ollama
        if options:
            response = ollama.chat(model=model, messages=messages, options=options)
        else:
            response = ollama.chat(model=model, messages=messages)
        
        # FIXED: Handle ollama 0.6+ object-style response
        # response is ChatResponse object with .message.content attributes
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
            if content and content.strip():
                return content.strip()
            else:
                raise LLMHardFail(step, "EmptyResponse", "Qwen returned empty content")
        
        # Fallback for older ollama versions (dict-style)
        elif isinstance(response, dict) and 'message' in response:
            message = response['message']
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if content and content.strip():
                    return content.strip()
                else:
                    raise LLMHardFail(step, "EmptyResponse", "Qwen returned empty content")
        
        # If neither format works
        else:
            raise LLMHardFail(step, "InvalidResponse", 
                            f"Unexpected response format. Type: {type(response)}, "
                            f"Has message: {hasattr(response, 'message')}")
    
    except ImportError:
        raise
    except LLMHardFail:
        raise
    except Exception as e:
        error_msg = str(e)
        if 'connection' in error_msg.lower() or 'refused' in error_msg.lower():
            raise LLMHardFail(step, "OllamaNotRunning", 
                            "Could not connect to Ollama. Start with: ollama serve")
        elif 'not found' in error_msg.lower() or 'pull' in error_msg.lower():
            raise LLMHardFail(step, "ModelNotFound",
                            f"Model '{model}' not found. Pull with: ollama pull {model}")
        else:
            raise LLMHardFail(step, "QwenError", str(e))


# ============================================================================
# Qwen Client via DashScope API (optional)
# ============================================================================

def _call_qwen_api(model: str, system_prompt: str, user_payload: str,
                  step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via DashScope API (Alibaba Cloud).
    
    ALTERNATIVE to Ollama - uses cloud API instead of local deployment.
    
    Setup:
    1. Get API key from: https://dashscope.console.aliyun.com/
    2. Set environment: export DASHSCOPE_API_KEY='sk-...'
    3. Install: pip install dashscope
    
    Args:
        model: Qwen model name (e.g., 'qwen-max', 'qwen-plus', 'qwen-turbo')
        system_prompt: System prompt
        user_payload: User message
        step: Step name for logging
        temperature: Temperature parameter
    
    Returns:
        Model response text
    """
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        raise LLMHardFail(step, "DashScopeNotInstalled",
                         "dashscope library not installed. Install with: pip install dashscope")
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise LLMHardFail(step, "MissingAPIKey",
                         "DASHSCOPE_API_KEY not found. Get it from: https://dashscope.console.aliyun.com/")
    
    dashscope.api_key = api_key
    
    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_payload}
        ]
        
        # Map simplified model names to DashScope model names
        model_mapping = {
            'qwen-max': 'qwen-max',
            'qwen-plus': 'qwen-plus', 
            'qwen-turbo': 'qwen-turbo',
            'qwen2.5-coder': 'qwen-coder-plus',
        }
        
        # Use mapped name or original
        dashscope_model = model_mapping.get(model, model)
        
        response = Generation.call(
            model=dashscope_model,
            messages=messages,
            result_format='message',
            temperature=temperature if temperature is not None else 0.85,
            top_p=0.8
        )
        
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if content and content.strip():
                return content.strip()
            else:
                raise LLMHardFail(step, "EmptyResponse", "Qwen API returned empty content")
        else:
            raise LLMHardFail(step, "APIError", 
                            f"DashScope API error: {response.code} - {response.message}")
    
    except ImportError:
        raise
    except Exception as e:
        raise LLMHardFail(step, "QwenAPIError", str(e))

# ============================================================================
# Qwen Client via Ollama REST API (remote, no local installation required)
# ============================================================================

def _call_qwen_rest_api(model: str, system_prompt: str, user_payload: str,
                        step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via a remote Ollama REST API endpoint.

    Does NOT require the ollama Python package or a local Ollama installation.
    Only requires the requests library (standard dependency).

    Setup:
        Set environment variable pointing to your remote Ollama server:
        export OLLAMA_BASE_URL=http://your-server.com:11434

    The Ollama REST API is OpenAI-compatible, using endpoint:
        POST {OLLAMA_BASE_URL}/api/chat
    """
    try:
        import requests
    except ImportError:
        raise LLMHardFail(step, "RequestsNotInstalled",
                          "requests library not installed. Install with: pip install requests")

    base_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    # This should never be called without OLLAMA_BASE_URL set,
    # but guard defensively anyway.
    if not base_url:
        raise LLMHardFail(step, "MissingOllamaBaseURL",
                          "OLLAMA_BASE_URL environment variable is not set.")

    endpoint = f"{base_url}/api/chat"

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ],
        "stream": False,   # get full response in one shot
        "options": {"top_p": 0.8},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()          # raises HTTPError for 4xx / 5xx
        data = response.json()

        # Ollama REST response shape:
        # {"message": {"role": "assistant", "content": "..."}, ...}
        content = data.get("message", {}).get("content", "")
        if content and content.strip():
            return content.strip()
        else:
            raise LLMHardFail(step, "EmptyResponse",
                              "Ollama REST API returned empty content")

    except requests.exceptions.ConnectionError:
        raise LLMHardFail(step, "OllamaRESTUnreachable",
                          f"Cannot reach Ollama REST API at {base_url}. "
                          "Check that the server is running and the URL is correct.")
    except requests.exceptions.Timeout:
        raise LLMHardFail(step, "OllamaRESTTimeout",
                          f"Ollama REST API timed out at {base_url}.")
    except requests.exceptions.HTTPError as e:
        raise LLMHardFail(step, "OllamaRESTHTTPError", str(e))
    except Exception as e:
        raise LLMHardFail(step, "OllamaRESTError", str(e))

def _call_qwen(model: str, system_prompt: str, user_payload: str,
               step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen model.

    Priority order:
    1. OLLAMA_BASE_URL set → remote Ollama REST API (no local install needed)
    2. ollama Python package available → local Ollama process
    3. DASHSCOPE_API_KEY set → DashScope cloud API
    4. None of the above → fatal with instructions
    """
    # ── Priority 1: Remote Ollama REST API ───────────────────────────────
    if os.getenv("OLLAMA_BASE_URL"):
        info(f"Using remote Ollama REST API: {os.getenv('OLLAMA_BASE_URL')}")
        return _call_qwen_rest_api(model, system_prompt, user_payload,
                                   step, temperature)

    # ── Priority 2: Local Ollama Python package ───────────────────────────
    try:
        return _call_qwen_ollama(model, system_prompt, user_payload,
                                 step, temperature)
    except LLMHardFail as e:
        if e.error_type in ["OllamaNotInstalled", "OllamaNotRunning", "ModelNotFound"]:

            # ── Priority 3: DashScope cloud API ──────────────────────────
            if os.getenv("DASHSCOPE_API_KEY"):
                warn("Ollama not available, trying DashScope API...")
                return _call_qwen_api(model, system_prompt, user_payload,
                                      step, temperature)

            # ── Priority 4: Nothing available ────────────────────────────
            else:
                fatal("Qwen model requires one of:")
                fatal("  Option 1: Remote Ollama REST API (no local install)")
                fatal("    - Set: export OLLAMA_BASE_URL=http://your-server:11434")
                fatal("  Option 2: Local Ollama")
                fatal("    - Start: ollama serve")
                fatal("    - Pull:  ollama pull qwen2.5-coder:7b")
                fatal("  Option 3: DashScope API (cloud)")
                fatal("    - Get key: https://dashscope.console.aliyun.com/")
                fatal("    - Set: export DASHSCOPE_API_KEY='sk-...'")
        else:
            raise


# ============================================================================
# Unified LLM Call
# ============================================================================

def _infer_qwen_temperature(model: str, base_temperature: Optional[float]) -> Optional[float]:
    """
    Infer appropriate temperature for Qwen model based on model name semantics.
    
    Rules (keyword-based, not hardcoded model names):
    - think/careful/compare → reasoning-style model, prone to hallucination → lower to 0.15
    - next/fast/turbo       → speed-optimized model, prone to skipping fields → raise to 0.4
    - coder                 → instruction-following model → keep at 0.3
    - default               → raise low temps to 0.3 minimum
    """
    if base_temperature is None:
        return None
    
    model_lower = model.lower()
    
    # Reasoning/careful models: hallucination risk → lower temperature
    if any(kw in model_lower for kw in ['think', 'careful', 'compare', 'reason']):
        return min(base_temperature, 0.15)
    
    # Speed/next models: field-skipping risk → raise temperature
    if any(kw in model_lower for kw in ['next', 'fast', 'turbo', 'lite']):
        return max(base_temperature, 0.4)
    
    # All other Qwen: raise low temps to 0.3 minimum
    if base_temperature < 0.2:
        return 0.3
    
    return base_temperature


def _call_llm(model: str, system_prompt: str, user_payload: str, 
             step: str, temperature: Optional[float] = None) -> str:
    """
    Universal LLM caller - supports OpenAI and Qwen.
    
    Auto-detects provider based on model name:
    - gpt*, o1*, o3* → OpenAI
    - qwen* → Qwen (Ollama or DashScope API)
    
    For Qwen: automatically adjusts temperature based on model name semantics.
    """
    if is_qwen_model(model):
        qwen_temp = _infer_qwen_temperature(model, temperature)
        return _call_qwen(model, system_prompt, user_payload, step, qwen_temp)
    elif is_openai_model(model):
        return _call_openai(model, system_prompt, user_payload, step, temperature)
    else:
        raise LLMHardFail(step, "UnknownProvider", 
                         f"Unknown model: {model}. Use 'gpt*', 'o1*', 'o3*', or 'qwen*'")

# ============================================================================
# PROMPTS
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

EXAMPLE for .mat in fNIRS dataset:
  user_hints:
    modality_hint: "nirs"
  
  → Decision: .mat files are fNIRS data

  mappings:
  - modality: mri         # OR nirs
    match: ['*.nii.gz', '**/*.dcm']  # Match patterns
    exclude: []           # Optional: glob patterns to exclude (e.g. ['**/BRIK/**'])
                          # Analyze sample_files to detect duplicate/redundant directories
                          # Common patterns: BRIK, backup, old, duplicate, raw_orig
    format_ready: true    # true ONLY for .nii/.nii.gz (MRI) or .snirf (fNIRS)
    convert_to: none      # OR 'nifti' (DICOM/JNIfTI) OR 'snirf' (.mat/.nirs)
    filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_task-rest_nirs.snirf'

═══════════════════════════════════════════════════════════════════════
FORMAT_READY AND CONVERT_TO RULES
═══════════════════════════════════════════════════════════════════════

format_ready: true (file is already BIDS-compliant, just copy):
  - MRI: .nii, .nii.gz files
  - fNIRS: .snirf files

format_ready: false (file needs conversion):
  - MRI: .dcm (convert_to: nifti), .jnii/.bnii (convert_to: nifti)
  - fNIRS: .mat (convert_to: snirf), .nirs (convert_to: snirf)

convert_to values:
  - "none": No conversion, direct copy (only when format_ready: true)
  - "nifti": Convert to NIfTI format (for DICOM, JNIfTI)
  - "snirf": Convert to SNIRF format (for .mat, .nirs)

CRITICAL EXAMPLES:

✓ CORRECT - fNIRS .mat files:
  mappings:
    - modality: nirs
      match: ['*.mat']
      format_ready: false
      convert_to: snirf
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_task-rest_nirs.snirf'

✓ CORRECT - SNIRF files (already ready):
  mappings:
    - modality: nirs
      match: ['*.snirf']
      format_ready: true
      convert_to: none
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_task-rest_nirs.snirf'

✗ WRONG - treating .mat as ready:
  mappings:
    - modality: nirs
      match: ['*.mat']
      format_ready: true    # ← WRONG! .mat needs conversion
      convert_to: none      # ← WRONG! Must specify snirf

✓ CORRECT - DICOM files:
  mappings:
    - modality: mri
      match: ['*.dcm']
      format_ready: false
      convert_to: nifti
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_T1w.nii.gz'

═══════════════════════════════════════════════════════════════════════
CORE PRINCIPLE: CONTEXT-DRIVEN REASONING
═══════════════════════════════════════════════════════════════════════

You receive THREE sources of information:
1. user_hints.user_text      - Human's description of the dataset
2. user_hints.modality_hint  - Explicit modality (mri/nirs/mixed)
3. sample_files               - Representative filenames
4. python_subject_analysis    - Statistical pattern detection

YOUR JOB: Synthesize ALL sources to understand the TRUE dataset structure.

CRITICAL DECISION TREE for .mat files:

1. Check modality_hint:
   - If "nirs" → modality: nirs, convert_to: snirf
   - If "mri" → modality: mri (may need voxel mapping plan)
   - If "mixed" or None → Check user_text for keywords

2. Check user_text for fNIRS keywords:
   - "fNIRS", "NIRS", "optodes", "channels" → treat as fNIRS
   - "MRI", "anatomical", "voxels" → treat as MRI

3. If still uncertain:
   - Generate blocking question asking user to clarify

═══════════════════════════════════════════════════════════════════════
SUBJECT ID STRATEGY
═══════════════════════════════════════════════════════════════════════

Check python_subject_analysis.id_mapping if available.

Strategy 1: NUMERIC - Use sub-1, sub-2, ... (add original_id to metadata)
Strategy 2: SEMANTIC - Preserve original IDs (sub-neo, sub-Beijing001)

Choose based on:
- Dataset size (>10 subjects → numeric)
- Multi-site data (→ semantic to preserve site info)
- Meaningful short IDs (→ semantic)

CRITICAL: SUBJECT ID FORMAT

In assignment_rules, the 'subject' field should be BARE ID without 'sub-' prefix:

✓ CORRECT:
  assignment_rules:
    - subject: '1'          # ← Just the number
      original: 'BZZ021'
      match: ['*BZZ021*']
  
  Result: executor creates "sub-1" (correct)

✗ WRONG:
  assignment_rules:
    - subject: 'sub-1'      # ← Don't add 'sub-' here!
      original: 'BZZ021'
      match: ['*BZZ021*']
  
  Result: executor creates "sub-sub-1" (double prefix!)

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

subjects:
  labels: [list of BIDS IDs without 'sub-' prefix]
  count: N
  source: llm_analysis
  id_strategy: numeric / semantic

assignment_rules:
  - subject: 'bids_id'    # Bare ID (no 'sub-' prefix)
    original: 'token'     # EXACT token from filenames
    match: ['*token*']    # Glob pattern

participant_metadata:
  'bids_id':              # Must match subjects.labels
    original_id: 'xxx'    # If using numeric strategy
    site: 'xxx'           # If multi-site
    age: 'xxx'            # If available
    sex: 'M'              # If available

mappings:
  - modality: mri         # OR nirs
    match: ['*.nii.gz', '**/*.dcm']  # Match patterns
    format_ready: true    # true ONLY for .nii/.nii.gz (MRI) or .snirf (fNIRS)
    convert_to: none      # OR 'nifti' (DICOM/JNIfTI) OR 'snirf' (.mat/.nirs)
    filename_rules:
      - match_pattern: '.*T1.*'
        bids_template: 'sub-X_T1w.nii.gz'  # MRI example
      # OR
      - match_pattern: '.*rest.*'
        bids_template: 'sub-X_task-rest_nirs.snirf'  # fNIRS example

COMPLETE EXAMPLES:

Example 1: MRI dataset with NIfTI files in BRIK/NIfTI duplicate directories
  mappings:
    - modality: mri
      match: ['**/*.nii.gz']
      exclude: ['**/BRIK/**']   ← exclude duplicate BRIK directory
      format_ready: true
      convert_to: none
      filename_rules:
        - match_pattern: '.*anonymized.*'
          bids_template: 'sub-X_T1w.nii.gz'
        - match_pattern: '.*skullstripped.*'
          bids_template: 'sub-X_desc-skullstripped_T1w.nii.gz'
        - match_pattern: '.*rest.*'
          bids_template: 'sub-X_task-rest_bold.nii.gz'

Example 2: MRI dataset with DICOM files
  mappings:
    - modality: mri
      match: ['*.dcm', '**/*.dcm']
      exclude: []
      format_ready: false
      convert_to: nifti
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_T1w.nii.gz'

Example 3: fNIRS dataset with .mat files
  mappings:
    - modality: nirs
      match: ['*.mat', '**/*.mat']
      format_ready: false
      convert_to: snirf
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_task-rest_nirs.snirf'

Example 4: Mixed dataset
  mappings:
    - modality: mri
      match: ['*.nii.gz']
      format_ready: true
      convert_to: none
      filename_rules:
        - match_pattern: '.*T1.*'
          bids_template: 'sub-X_T1w.nii.gz'
    
    - modality: nirs
      match: ['*.mat']
      format_ready: false
      convert_to: snirf
      filename_rules:
        - match_pattern: '.*'
          bids_template: 'sub-X_task-rest_nirs.snirf'

OUTPUT: Raw YAML only (no markdown, no explanation)
"""


# ============================================================================
# LLM CALL FUNCTIONS
# ============================================================================

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