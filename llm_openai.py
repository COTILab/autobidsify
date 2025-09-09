# -*- coding: utf-8 -*-
"""
LLM driver to synthesize (if missing):
- dataset_description.json (Name/License required; warn if missing)
- README.md (Markdown)
- participants.tsv (if real participant info not found)

If OpenAI is unavailable, we fallback to a deterministic stub generator that:
- Infers Name from user_text (first line)
- Leaves License empty (and warns)
- Builds README from counts and detected modalities
- Creates participants.tsv with participant_id=sub-XX (N rows)
"""
import os, json, re
from typing import Dict, Any, Tuple, List
from utils import write_text, write_json

# Optional: OpenAI
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

SYSTEM_PROMPT = """You are a BIDS dataset steward and technical writer.
You will receive a compact 'evidence bundle' JSON that includes:
- directory listing stats and samples (text/table/json snippets),
- SNIRF/NIfTI/DICOM metadata (no raw pixels/voxels),
- whether trio files exist, and any trio texts if present,
- the user's free-form description (user_text),
- and user_hints.n_subjects.

TASK:
1) If dataset_description.json is missing, produce one JSON string. REQUIRED fields:
   - Name (non-empty), License (non-empty). If truly unknown, leave empty BUT
     add a clear WARNING list that explains what the user must provide.
   - Fill other fields when possible: BIDSVersion, DatasetType ("raw"), HEDVersion,
     Authors, Acknowledgements, HowToAcknowledge, Funding, EthicsApprovals,
     ReferencesAndLinks, DatasetDOI, GeneratedBy, SourceDatasets.
2) If README.md is missing, produce a Markdown string that:
   - summarizes the dataset (modalities, subjects count, file structure),
   - mentions License and how to acknowledge,
   - lists TODO/Questions for the user (e.g., confirm task labels, ethics).
3) If participants.tsv is missing:
   - Try to infer participant info from any tabular/text evidence (columns like participant_id/subject/id).
   - If still unknown, output only one column `participant_id` with values `sub-01..sub-N`
     where N = user_hints.n_subjects.

Return a single JSON object with keys:
- generated: { "dataset_description.json": <string or null>,
               "README.md": <string or null>,
               "participants.tsv": <string or null> }
- warnings: [ ... human-readable warnings ... ]
"""

def _first_line(txt: str) -> str:
    for line in (txt or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""

def _stub_generate(evidence: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Fallback generator when OpenAI SDK/key is not available."""
    warnings: List[str] = []
    n = int(evidence.get("user_hints", {}).get("n_subjects", 0) or 0)
    user_text = evidence.get("user_text") or ""
    name_guess = _first_line(user_text)
    if not name_guess:
        warnings.append("dataset_description.json: Name is missing; please provide a dataset title.")
    # License: force user to provide; leave empty with warning
    warnings.append("dataset_description.json: License is missing; please provide a standard license (e.g., CC-BY-4.0).")

    # dataset_description.json
    ds = {
        "Name": name_guess or "",
        "BIDSVersion": "1.10.0",
        "HEDVersion": "",
        "DatasetType": "raw",
        "License": "",
        "DataLicense": "",
        "Authors": [],
        "Acknowledgements": "",
        "HowToAcknowledge": "",
        "Funding": [],
        "EthicsApprovals": [],
        "ReferencesAndLinks": [],
        "DatasetDOI": "",
        "GeneratedBy": [],
        "SourceDatasets": []
    }
    ds_json_text = json.dumps(ds, indent=2, ensure_ascii=False)

    # README.md
    counts = evidence.get("counts", {})
    modal = []
    if any(k in counts for k in [".snirf"]): modal.append("SNIRF")
    if any(k in counts for k in [".nii", ".nii.gz"]): modal.append("NIfTI")
    if any(k in counts for k in [".dcm"]): modal.append("DICOM")
    mod_str = ", ".join(modal) if modal else "unknown"
    readme = f"""# Dataset Overview

- **Name**: {name_guess or "(please fill)"}  
- **Modalities detected**: {mod_str}  
- **Subjects**: {n if n>0 else "(please provide --nsubjects)"}  

## Organization
This repository follows a minimal BIDS-like structure. Raw DICOM are kept in `sourcedata/dicom/`. Non-BIDS files are preserved in `derivatives/` for provenance.

## License
(Please provide a standard license, e.g., CC-BY-4.0. This field is required in `dataset_description.json`.)

## Acknowledgements
(Add how to acknowledge this dataset if applicable.)

## TODO / Questions
- Confirm dataset **Name** and **License**.
- Confirm task labels (default `task-rest` used for fNIRS and fMRI).
- Provide any EthicsApprovals, Funding, and References if applicable.
"""

    # participants.tsv
    # Unknown participants → generate only participant_id column
    parts_lines = ["participant_id"]
    if n and n > 0:
        parts_lines += [f"sub-{str(i+1).zfill(2)}" for i in range(n)]
    else:
        warnings.append("participants.tsv: --nsubjects is missing or 0; generated header only.")
    participants_tsv = "\n".join(parts_lines)

    out = {
        "dataset_description.json": ds_json_text,
        "README.md": readme,
        "participants.tsv": participants_tsv
    }
    return out, warnings

def _call_openai(evidence: Dict[str, Any], model: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Call OpenAI Responses to generate trio files if missing.
    We always pass the evidence bundle (already compact) plus the policy in SYSTEM_PROMPT.
    """
    client = OpenAI()
    payload = json.dumps(evidence, ensure_ascii=False)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload}
        ],
        temperature=0.2
    )
    # Extract text
    text = ""
    if getattr(resp, "output", None):
        for item in resp.output:
            if getattr(item, "type", "") == "output_text":
                text += item.text
    elif getattr(resp, "output_text", None):
        text = resp.output_text

    # Expect a JSON object with 'generated' and 'warnings'
    try:
        data = json.loads(text)
        generated = data.get("generated", {})
        warnings = data.get("warnings", [])
        # Ensure keys exist
        for key in ["dataset_description.json", "README.md", "participants.tsv"]:
            generated.setdefault(key, None)
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        return generated, warnings
    except Exception:
        # If the model output isn't parseable, fallback to stub
        return _stub_generate(evidence)

def generate_trio_if_missing(evidence: Dict[str, Any], out_dir: str, model: str = "gpt-5-mini") -> List[str]:
    """
    Decide which trio files are missing, then generate only the missing ones via OpenAI or stub.
    Return a list of human-readable warnings.
    """
    out_dir = Path(out_dir)
    trio_found = evidence.get("trio_found", {})
    need_ds = not bool(trio_found.get("dataset_description.json"))
    need_md = not bool(trio_found.get("readme.md"))
    need_parts = not bool(trio_found.get("participants.tsv"))

    # If none missing, nothing to do
    if not (need_ds or need_md or need_parts):
        return []

    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        generated, warnings = _stub_generate(evidence)
    else:
        try:
            generated, warnings = _call_openai(evidence, model=model)
        except Exception:
            generated, warnings = _stub_generate(evidence)

    # Write only missing ones
    if need_ds and generated.get("dataset_description.json"):
        try:
            # Accept both JSON string or dict
            val = generated["dataset_description.json"]
            if isinstance(val, str):
                Path(out_dir, "dataset_description.json").write_text(val, encoding="utf-8")
            else:
                write_json(Path(out_dir, "dataset_description.json"), val)
        except Exception as e:
            warnings.append(f"Failed to write dataset_description.json: {e}")

    if need_md and generated.get("README.md"):
        try:
            write_text(Path(out_dir, "README.md"), generated["README.md"])
        except Exception as e:
            warnings.append(f"Failed to write README.md: {e}")

    if need_parts and generated.get("participants.tsv"):
        try:
            write_text(Path(out_dir, "participants.tsv"), generated["participants.tsv"])
        except Exception as e:
            warnings.append(f"Failed to write participants.tsv: {e}")

    return warnings

