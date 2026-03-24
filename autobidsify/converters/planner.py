# converters/planner.py

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import re
from datetime import datetime
from collections import defaultdict
from autobidsify.utils import write_json, read_json, write_yaml, info, warn, fatal
from autobidsify.constants import SEVERITY_BLOCK
from autobidsify.llm import llm_nirs_draft, llm_nirs_normalize, llm_bids_plan

HEADERS_DRAFT      = "nirs_headers_draft.json"
HEADERS_NORMALIZED = "nirs_headers_normalized.json"
BIDS_PLAN          = "BIDSPlan.yaml"

# Data file extensions — used for filtering in multiple places
_DATA_EXTS = {
    '.snirf', '.nirs', '.mat',
    '.dcm', '.nii', '.jnii', '.bnii', '.nii.gz',
}


# ============================================================================
# Helpers
# ============================================================================

def _parse_llm_json_response(response_text: str, step_name: str) -> Optional[Dict[str, Any]]:
    """Strip markdown fences and parse JSON from LLM response."""
    if not response_text or not response_text.strip():
        warn(f"{step_name}: LLM returned empty response")
        return None

    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
                return obj
            except Exception:
                pass
        warn(f"{step_name}: Failed to parse JSON: {e}")
        return None


def _is_data_file(path: str) -> bool:
    """Return True if path has a recognised neuroimaging extension."""
    low = path.lower()
    if low.endswith('.nii.gz'):
        return True
    ext = ('.' + low.rsplit('.', 1)[-1]) if '.' in low else ''
    return ext in _DATA_EXTS


def _extract_subjects_from_directory_structure(all_files: List[str]) -> Dict[str, Any]:
    """
    Detect subjects from top-level directory names.
    Supports: site_sub-NN, sub-NN, subject-NN, pure-numeric dirs.
    """
    patterns = [
        (r'([A-Za-z]+)_sub(\d+)', True,  2, 1, "site_prefixed"),
        (r'sub-(\w+)',             False, 1, None, "standard_bids"),
        (r'subject[_-]?(\d+)',    False, 1, None, "simple"),
        (r'^(\d{3,})$',           False, 1, None, "numeric_only"),
    ]

    subject_records: List[Dict] = []
    seen_ids: set = set()

    for filepath in all_files:
        parts = filepath.split('/')
        for part in parts[:2]:
            for regex, has_site, id_grp, site_grp, pname in patterns:
                m = re.match(regex, part, re.IGNORECASE)
                if m:
                    original_id = m.group(0)
                    if original_id in seen_ids:
                        break
                    seen_ids.add(original_id)
                    subject_records.append({
                        "original_id": original_id,
                        "numeric_id":  m.group(id_grp),
                        "site":        m.group(site_grp) if has_site and site_grp else None,
                        "pattern_name": pname,
                    })
                    break

    if not subject_records:
        return {"success": False, "method": "directory_structure"}

    subject_records.sort(
        key=lambda x: int(x["numeric_id"]) if x["numeric_id"].isdigit() else 0
    )
    return {
        "success":       True,
        "method":        "directory_structure",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "has_site_info": any(r["site"] for r in subject_records),
    }


def _extract_subjects_from_flat_filenames(all_files: List[str]) -> Dict[str, Any]:
    """
    Detect subjects from data-file filename prefixes.
    Only processes files with recognised neuroimaging extensions.
    """
    identifier_to_files: Dict[str, List[str]] = defaultdict(list)

    for filepath in all_files:
        filename = filepath.split('/')[-1]
        low = filename.lower()

        # Extension check
        if low.endswith('.nii.gz'):
            ext = '.nii.gz'
        else:
            ext = ('.' + low.rsplit('.', 1)[-1]) if '.' in low else ''
        if ext not in _DATA_EXTS:
            continue

        # Strip extension(s)
        name_no_ext = filename
        if name_no_ext.lower().endswith('.nii.gz'):
            name_no_ext = name_no_ext[:-7]
        elif '.' in name_no_ext:
            name_no_ext = name_no_ext.rsplit('.', 1)[0]

        m = re.match(r'^([A-Za-z0-9\-]+)', name_no_ext)
        if m:
            identifier_to_files[m.group(1)].append(filepath)

    if not identifier_to_files:
        return {"success": False, "method": "flat_filename"}

    def _sort_key(ident: str) -> int:
        nums = re.findall(r'\d+', ident)
        return int(nums[-1]) if nums else 999999

    sorted_ids = sorted(identifier_to_files.keys(), key=_sort_key)
    subject_records = [
        {
            "original_id":  ident,
            "numeric_id":   str(i),
            "site":         None,
            "pattern_name": "filename_identifier",
            "file_count":   len(identifier_to_files[ident]),
        }
        for i, ident in enumerate(sorted_ids, 1)
    ]

    info(f"  Detected {len(subject_records)} unique identifiers:")
    for rec in subject_records[:10]:
        info(f"    '{rec['original_id']}': {rec['file_count']} file(s)")
    if len(subject_records) > 10:
        info(f"    ... and {len(subject_records) - 10} more")

    return {
        "success":         True,
        "method":          "flat_filename_identifiers",
        "subject_records": subject_records,
        "subject_count":   len(subject_records),
        "has_site_info":   False,
    }


def _collect_extra_columns(metadata: Dict[str, Any]) -> List[str]:
    """Return deduplicated extra column names from participant_metadata."""
    seen: set = set()
    cols: List[str] = []
    for meta in metadata.values():
        for col in meta.keys():
            if col not in seen and col != "participant_id":
                seen.add(col)
                cols.append(col)
    return cols


def _write_participants_from_plan(
    plan_yaml: Dict[str, Any],
    out_dir: Path,
    user_n_subjects: Optional[int],
) -> None:
    """
    Write participants.tsv from LLM assignment_rules.
    LLM rules are authoritative; warn if count < user expectation.
    """
    parts_path = out_dir / "participants.tsv"
    if parts_path.exists():
        parts_path.unlink()

    rules  = plan_yaml.get("assignment_rules", [])
    labels = plan_yaml.get("subjects", {}).get("labels", [])

    seen:    set       = set()
    ordered: List[str] = []
    for rule in rules:
        sid = str(rule.get("subject", ""))
        if sid and sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    if not ordered:
        ordered = [str(lbl) for lbl in labels]

    if user_n_subjects and len(ordered) < user_n_subjects:
        warn(f"  ⚠ participants.tsv has {len(ordered)} subjects "
             f"but user specified {user_n_subjects}. "
             f"LLM assignment_rules may be incomplete — check BIDSPlan.yaml.")

    metadata      = plan_yaml.get("participant_metadata", {})
    extra_columns = _collect_extra_columns(metadata)
    columns       = ["participant_id"] + extra_columns

    def _sort_key(sid: str):
        try:    return (0, int(sid))
        except: return (1, sid)

    lines = ["\t".join(columns) + "\n"]
    for sid in sorted(ordered, key=_sort_key):
        meta = metadata.get(sid, {})
        row  = [f"sub-{sid}"] + [str(meta.get(col, "n/a")) for col in extra_columns]
        lines.append("\t".join(row) + "\n")

    parts_path.write_text("".join(lines))
    info(f"  ✓ participants.tsv: {len(ordered)} subjects, columns: {columns}")


def _merge_participants_from_llm_metadata(
    plan_yaml: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Append any extra columns from participant_metadata to existing participants.tsv."""
    parts_path = out_dir / "participants.tsv"
    if not parts_path.exists():
        return

    metadata      = plan_yaml.get("participant_metadata", {})
    extra_columns = _collect_extra_columns(metadata)
    if not extra_columns:
        info("  No extra columns from LLM metadata")
        return

    existing = parts_path.read_text().splitlines()
    if not existing:
        return

    header   = existing[0].split("\t")
    new_cols = [c for c in extra_columns if c not in header]
    if not new_cols:
        info("  participants.tsv already has all metadata columns")
        return

    info(f"  Adding columns to participants.tsv: {new_cols}")
    new_lines = ["\t".join(header + new_cols)]
    for line in existing[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        sid   = cells[0].replace("sub-", "")
        meta  = metadata.get(sid, {})
        new_lines.append("\t".join(cells + [str(meta.get(c, "n/a")) for c in new_cols]))

    parts_path.write_text("\n".join(new_lines) + "\n")
    info(f"  ✓ participants.tsv updated with {len(new_cols)} new column(s)")


# ============================================================================
# Main entry point
# ============================================================================

def build_bids_plan(model: str, planning_inputs: Dict[str, Any],
                    out_dir: Path, id_strategy: str = "auto") -> Dict[str, Any]:
    """
    Build BIDS conversion plan (LLM-first, Python-validates).

    Steps:
      1. Python extracts subject hints from directory/filename structure (advisory).
      2. Build LLM payload — data files only, non-data files filtered out.
      3. Call LLM to generate full BIDSPlan (assignment_rules, mappings, metadata).
      4. Validate subject count: trust LLM rules, only update count field if needed.
      5. Write participants.tsv from LLM plan.
      6. Merge any extra metadata columns from LLM.
      7. Save BIDSPlan.yaml.
    """
    info("=== Generating Unified BIDS Plan ===")

    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files       = evidence_bundle.get("all_files", [])
    user_hints      = evidence_bundle.get("user_hints", {})
    user_n_subjects = user_hints.get("n_subjects")

    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Python structural hints (advisory only) ───────────────
    info("Step 1: Python extracting structural hints...")
    subject_info = _extract_subjects_from_directory_structure(all_files)
    if not subject_info["success"]:
        info("  Directory-level detection failed, trying flat filename analysis...")
        subject_info = _extract_subjects_from_flat_filenames(all_files)

    python_subject_count = subject_info.get("subject_count", 0)
    info(f"  Python hint: {python_subject_count} subjects "
         f"(method: {subject_info.get('method', 'unknown')})")

    # ── Step 2: Build LLM payload (data files only) ───────────────────
    info("\nStep 2: Building LLM payload...")
    data_files = [f for f in all_files if _is_data_file(f)]

    if len(data_files) <= 200:
        sample_files = data_files
    else:
        n       = len(data_files)
        indices = (list(range(0, min(50, n))) +
                   list(range(n // 2 - 25, n // 2 + 25)) +
                   list(range(max(0, n - 50), n)))
        sample_files = [data_files[i] for i in sorted(set(indices)) if i < n]

    info(f"  Data files for LLM: {len(sample_files)} "
         f"(filtered from {len(all_files)} total)")

    optimized_bundle = {
        "root":          evidence_bundle.get("root"),
        "counts_by_ext": {
            k: v for k, v in evidence_bundle.get("counts_by_ext", {}).items()
            if k.lower() in _DATA_EXTS
        },
        "user_hints":    user_hints,
        "total_files":   len(all_files),
        "data_files":    len(data_files),
        "sample_files":  sample_files,
        "python_subject_analysis": {
            "success":       subject_info["success"],
            "method":        subject_info.get("method"),
            "subject_count": python_subject_count,
            "subject_examples": [
                {
                    "original":   rec["original_id"],
                    "numeric_id": rec.get("numeric_id"),
                    "site":       rec.get("site"),
                }
                for rec in subject_info.get("subject_records", [])[:20]
            ],
            "note": (
                "This is a HINT from Python's heuristic detection. "
                "Trust user_hints.n_subjects over this count. "
                "Use your own analysis of sample_files to determine "
                "the true subject structure."
            ),
        },
    }

    # ── Step 3: Call LLM ──────────────────────────────────────────────
    info("\nStep 3: Calling LLM for full plan generation...")
    evidence_json = json.dumps(optimized_bundle, indent=2)
    plan_response = llm_bids_plan(model, evidence_json)

    if not plan_response:
        fatal("LLM returned empty response for BIDS plan")

    try:
        text = plan_response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        plan_yaml = yaml.safe_load(text.strip())
    except yaml.YAMLError as e:
        fatal(f"BIDS plan YAML parsing failed: {e}")

    if not isinstance(plan_yaml, dict):
        fatal("BIDS plan is not a valid YAML dict")

    # ── Step 4: Validate subject count ────────────────────────────────
    info("\nStep 4: Validating subject count...")
    llm_count = plan_yaml.get("subjects", {}).get("count", 0)
    info(f"  LLM produced:  {llm_count} subjects")
    info(f"  User provided: {user_n_subjects} subjects (--nsubjects)")

    if user_n_subjects and llm_count != user_n_subjects:
        warn(f"  ⚠ LLM subject count ({llm_count}) ≠ user-provided count "
             f"({user_n_subjects}). Trusting LLM assignment_rules; updating count only.")
        plan_yaml["subjects"]["count"] = user_n_subjects

    # ── Step 5: Write participants.tsv ────────────────────────────────
    info("\nStep 5: Generating participants.tsv from LLM plan...")
    _write_participants_from_plan(plan_yaml, out_dir, user_n_subjects)

    # ── Step 6: Merge extra metadata columns ──────────────────────────
    info("\nStep 6: Merging participant metadata...")
    if "participant_metadata" in plan_yaml:
        _merge_participants_from_llm_metadata(plan_yaml, out_dir)

    # ── Step 7: Save plan ─────────────────────────────────────────────
    plan_yaml["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "model":        model,
        "id_strategy":  id_strategy,
    }
    plan_path = staging_dir / BIDS_PLAN
    write_yaml(plan_path, plan_yaml)
    info(f"\n✓ Plan saved: {plan_path}")

    final_count = plan_yaml.get("subjects", {}).get("count", llm_count)
    info(f"\n=== Complete: {final_count} subjects ===")
    return {"status": "ok", "warnings": [], "questions": []}


# ============================================================================
# NIRS header planning (separate from main BIDS plan)
# ============================================================================

def nirs_plan_headers(model: str, planning_inputs: Dict[str, Any],
                      out_dir: Path) -> Dict[str, Any]:
    """Plan fNIRS header mappings via two-step LLM draft/normalize."""
    info("=== Planning NIRS headers ===")

    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    evidence_json   = json.dumps(evidence_bundle, indent=2)

    draft_response = llm_nirs_draft(model, evidence_json)
    if not draft_response:
        return {"warnings": [], "questions": []}

    draft = _parse_llm_json_response(draft_response, "nirs_draft")
    if not draft:
        return {"warnings": [], "questions": []}

    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    write_json(staging_dir / HEADERS_DRAFT, draft)

    normalized_response = llm_nirs_normalize(model, json.dumps(draft, indent=2))
    if not normalized_response:
        return {"warnings": [], "questions": []}

    normalized = _parse_llm_json_response(normalized_response, "nirs_normalize")
    if not normalized:
        return {"warnings": [], "questions": []}

    write_json(staging_dir / HEADERS_NORMALIZED, normalized)
    info("✓ NIRS headers saved")
    return {"warnings": [], "questions": []}


def mri_plan_voxel_mappings(model: str, planning_inputs: Dict[str, Any],
                             out_dir: Path) -> Dict[str, Any]:
    """MRI voxel mapping planning (stub)."""
    return {"warnings": [], "questions": []}