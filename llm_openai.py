# -*- coding: utf-8 -*-
"""
OpenAI (ChatGPT) LLM driver using the Python SDK and the Responses API.

- Default model is "gpt-5-mini" for cost; pass "--model gpt-5" for best quality.
- If OPENAI_API_KEY is missing, or an API error occurs, we fall back to a deterministic stub.
"""
import os, json
from typing import Dict, Any
import yaml

try:
    # Official SDK; see docs:
    # - Models:     https://platform.openai.com/docs/models
    # - Responses:  https://platform.openai.com/docs/api-reference/responses
    # - Python SDK: https://github.com/openai/openai-python
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from rules import default_bidsmap_stub

SYSTEM_PROMPT = """You are a BIDS expert. You will receive a compact JSON "evidence bundle" that summarizes a dataset:
- directory structure stats (file extensions), representative samples with small headers,
- presence/absence of the user trio: README.md, participants.tsv, dataset_description.json.

TASK: Produce a single YAML "BIDSMap" describing how to convert/copy and name files into a minimal valid BIDS layout.
HARD RULES:
1) If the user trio is present (partially or fully), treat those files as hints and copy them to the BIDS root unchanged.
2) Keep ONLY BIDS-required content + the user trio at the BIDS root; route ALL other files to derivatives/ buckets.
3) For naming: prefer omitting 'ses' and 'run' when there's a single session or single repetition.
4) For 'task': infer from filenames/docs if possible; otherwise provide a safe per-modality default (e.g., 'rest' for fMRI).
5) If uncertain, add an item to 'questions:' explaining what needs confirmation (e.g., task label, license).
6) Output ONLY YAML. No prose, no markdown fences.
YAML MUST contain keys:
- bids_version
- policy.keep_user_trio, policy.route_residuals_to_derivatives, policy.derivatives_buckets (include '*.md' except README.md routed to docs)
- entities.subject/session/run
- mappings: list of {modality, match:[...], sidecar?: {...}}
- participants: respect_user_file: true, fallback_generate_if_missing: false
- questions: list
"""

def _build_user_message(evidence: Dict[str, Any], prefer_user_trio: bool) -> str:
    """Prepare a compact user message to control token usage."""
    msg = {
        "prefer_user_trio": bool(prefer_user_trio),
        "root": evidence.get("root"),
        "counts": evidence.get("counts"),
        "trio_found": evidence.get("trio_found"),
        # Keep samples bounded to avoid overlong prompts
        "samples": evidence.get("samples", [])[:30],
    }
    return json.dumps(msg, ensure_ascii=False)

def _call_openai_responses(model: str, system_prompt: str, user_json: str) -> str:
    """Call the OpenAI Responses API and return the text output."""
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_json}
        ],
        temperature=0.2,
    )
    # Extract text from the first output item
    if hasattr(resp, "output") and resp.output:
        chunks = []
        for item in resp.output:
            if getattr(item, "type", "") == "output_text":
                chunks.append(item.text)
        return "".join(chunks).strip()
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    raise RuntimeError("OpenAI response had no text output")

def generate_bidsmap_with_openai_or_stub(evidence: Dict[str, Any], llm_backend: str, model: str, prefer_user_trio: bool = True) -> str:
    """
    Generate a BIDSMap via ChatGPT (Responses API). Fall back to a default stub if:
    - SDK not installed, or
    - OPENAI_API_KEY missing, or
    - Any API error occurs.
    """
    if llm_backend != "openai":
        return default_bidsmap_stub()
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return default_bidsmap_stub()
    try:
        user_json = _build_user_message(evidence, prefer_user_trio=prefer_user_trio)
        yaml_text = _call_openai_responses(model=model, system_prompt=SYSTEM_PROMPT, user_json=user_json)
        parsed = yaml.safe_load(yaml_text.strip().strip("`"))
        if not isinstance(parsed, dict) or "mappings" not in parsed or "policy" not in parsed:
            # If the model added fence or stray text, try to parse again after trimming
            parsed = yaml.safe_load(yaml_text)
        return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    except Exception:
        return default_bidsmap_stub()

