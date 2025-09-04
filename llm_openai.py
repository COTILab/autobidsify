# -*- coding: utf-8 -*-
"""
OpenAI (ChatGPT) LLM driver using the Python SDK and the Responses API.

Update: The system prompt now requests 'assignment_rules' so the model
assigns each file to a subject without relying on user-supplied regex.
"""
import os, json, yaml
from typing import Dict, Any

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from rules import default_bidsmap_stub

SYSTEM_PROMPT = """You are a BIDS expert. You will receive a compact JSON evidence bundle with:
- directory stats, sample headers,
- presence/absence of user trio (README.md, participants.tsv, dataset_description.json),
- a FULL list of file paths (all_files),
- optional user_hints.n_subjects = total number of subjects.

TASK: Output a single YAML 'BIDSMap' for a minimal valid BIDS layout.
HARD RULES:
1) If the user trio exists, copy them unchanged to BIDS root.
2) Keep ONLY BIDS-required content + user trio at BIDS root; route ALL other files to derivatives/ buckets.
3) Prefer omitting 'ses' and 'run' when singletons.
4) For task: infer from names if possible; otherwise use a safe default (e.g., 'rest' for NIRS/fMRI).
5) If uncertain, add to 'questions:'.
6) Output ONLY YAML (no markdown fences).

NEW REQUIREMENTS:
- Infer subject membership WITHOUT using user-provided regex. Use the provided all_files and any patterns you see.
- Use user_hints.n_subjects if present, and produce a SUBJECT ASSIGNMENT plan:
  subjects:
    total: <N>                # if known
    labels: ["01","02",...]   # your chosen subject labels
  assignment_rules:
    - subject: "01"
      match: ["**/BZZ014*.nirs", "**/BZZ014/*", "..."]  # globs or explicit files
    - subject: "02"
      match: ["**/BZZ003*.nirs", "..."]
- In mappings, use {subject} token in bids_out instead of regex. Also provide {ext} if extension matters.

YAML MUST contain:
- bids_version
- policy.keep_user_trio, policy.route_residuals_to_derivatives, policy.derivatives_buckets (include '*.md' except README.md routed to docs)
- entities.subject/session/run
- subjects / assignment_rules (as defined above)
- mappings: list of {modality, match:[...], bids_out: "...{subject}...{ext}", sidecar?: {...}}
- participants: respect_user_file: true, fallback_generate_if_missing: false
- questions: list
"""

def _build_user_message(evidence: Dict[str, Any], prefer_user_trio: bool) -> str:
    msg = {
        "prefer_user_trio": bool(prefer_user_trio),
        "root": evidence.get("root"),
        "counts": evidence.get("counts"),
        "trio_found": evidence.get("trio_found"),
        "all_files": evidence.get("all_files", [])[:50000],  # large cap
        "user_hints": evidence.get("user_hints", {}),
        "samples": evidence.get("samples", [])[:50],
    }
    return json.dumps(msg, ensure_ascii=False)

def _call_openai_responses(model: str, system_prompt: str, user_json: str) -> str:
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[{"role":"system","content":system_prompt},
               {"role":"user","content":user_json}],
        temperature=0.2,
    )
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
    if llm_backend != "openai":
        return default_bidsmap_stub()
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return default_bidsmap_stub()
    try:
        user_json = _build_user_message(evidence, prefer_user_trio=prefer_user_trio)
        yaml_text = _call_openai_responses(model=model, system_prompt=SYSTEM_PROMPT, user_json=user_json)
        parsed = yaml.safe_load(yaml_text.strip().strip("`"))
        if not isinstance(parsed, dict) or "mappings" not in parsed:
            parsed = yaml.safe_load(yaml_text)
        return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    except Exception:
        return default_bidsmap_stub()

