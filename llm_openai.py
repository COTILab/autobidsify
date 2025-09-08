# -*- coding: utf-8 -*-
"""
OpenAI (ChatGPT) LLM driver using the Python SDK and the Responses API.

Update: The system prompt now requests 'assignment_rules' so the model
assigns each file to a subject without relying on user-supplied regex.
"""
import os, json, yaml
from typing import Dict, Any
from rules import default_bidsmap_stub

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ---- Smart stub that builds assignment_rules+mappings from evidence (no LLM) ----
from typing import Dict, Any

NIRS_SUFFIXES = (".nirs", ".snirf", ".mat", ".csv", ".tsv", ".txt")
NIRS_HINTS = ("nirs", "fnirs", "nirx", "homer", "snirf")

def _make_stub_from_evidence(evidence: Dict[str, Any]) -> str:
    """
    Build a usable BIDSMap without LLM:
    - Read evidence['all_files'] (relative paths) and user_hints.n_subjects.
    - Pick NIRS-like files by suffix/keyword.
    - Assign the first N candidates to sub-01..sub-N (1 file per subject).
    - Produce assignment_rules + mappings using {subject} and {ext}.
    """
    all_files = [str(x) for x in evidence.get("all_files", [])]
    n = int(evidence.get("user_hints", {}).get("n_subjects", 0) or 0)

    # candidates: NIRS-like files
    cand = []
    for rel in all_files:
        low = rel.lower()
        if low.endswith(NIRS_SUFFIXES) or any(h in low for h in NIRS_HINTS):
            cand.append(rel)
    cand = sorted(cand)

    # subject labels
    labels = [str(i+1).zfill(2) for i in range(n)] if n > 0 else []

    # assignment rules: one file per subject (first N)
    rules = []
    for i, lab in enumerate(labels):
        match_list = [cand[i]] if i < len(cand) else []
        rules.append({"subject": lab, "match": match_list})

    data = {
        "bids_version": "1.10.0",
        "policy": {
            "keep_user_trio": True,
            "route_residuals_to_derivatives": True,
            "derivatives_buckets": {
                "docs": ["**/*.pdf","**/*.html","**/*.htm","**/*.md"],
                "intermediate": ["**/*.log","**/*.tmp","**/*.bak"],
                "_misc": ["**/*"]
            }
        },
        "entities": {
            "subject": {"pad": 2, "sanitize": True},
            "session": {"omit_if_single": True},
            "run": {"omit_if_single": True}
        },
        "subjects": {"total": n, "labels": labels},
        "assignment_rules": rules,
        "mappings": [
            {
                "modality": "nirs",
                "match": [
                    {"pattern": "**/*.nirs",  "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                    {"pattern": "**/*.snirf", "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                    {"pattern": "**/*.mat",   "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                    {"pattern": "**/*.csv",   "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                    {"pattern": "**/*.tsv",   "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                    {"pattern": "**/*.txt",   "bids_out": "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"},
                ],
                "sidecar": { "TaskName": "rest", "SamplingFrequency": "unknown" }
            }
        ],
        "participants": { "respect_user_file": True, "fallback_generate_if_missing": False },
        "questions": [
            "Stub plan: one file per subject by order. Adjust assignment_rules if subjects have multiple files."
        ]
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

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

'''
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
'''

def generate_bidsmap_with_openai_or_stub(evidence: Dict[str, Any], llm_backend: str, model: str, prefer_user_trio: bool = True) -> str:
    # If user explicitly chooses 'stub', build a smart stub from evidence
    if llm_backend != "openai":
        return _make_stub_from_evidence(evidence)

    # If OpenAI SDK or API key is missing, fallback to smart stub
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return _make_stub_from_evidence(evidence)

    # Try calling OpenAI; on any failure fallback to smart stub
    try:
        user_json = _build_user_message(evidence, prefer_user_trio=prefer_user_trio)
        yaml_text = _call_openai_responses(model=model, system_prompt=SYSTEM_PROMPT, user_json=user_json)
        parsed = yaml.safe_load(yaml_text.strip().strip("`"))
        if not isinstance(parsed, dict) or "mappings" not in parsed:
            parsed = yaml.safe_load(yaml_text)
        return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    except Exception:
        return _make_stub_from_evidence(evidence)


