# converters/planner.py
# UNIVERSAL ID EXTRACTION: Use ALL files, not just samples

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import re
from datetime import datetime
from collections import defaultdict
from autobidsify.utils import write_json, read_json, write_yaml, info, warn, fatal, write_text
from autobidsify.constants import SEVERITY_BLOCK
from autobidsify.llm import llm_nirs_draft, llm_nirs_normalize, llm_mri_voxel_draft, llm_mri_voxel_final, llm_bids_plan

HEADERS_DRAFT = "nirs_headers_draft.json"
HEADERS_NORMALIZED = "nirs_headers_normalized.json"
VOXEL_DRAFT = "mri_voxel_draft.json"
VOXEL_FINAL_PLAN = "mri_voxel_final.json"
BIDS_PLAN = "BIDSPlan.yaml"


def _parse_llm_json_response(response_text: str, step_name: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response with fallback handling."""
    if not response_text or not response_text.strip():
        warn(f"{step_name}: LLM returned empty response")
        return None
    
    text = response_text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    elif text.startswith("```"):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            try:
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(text)
                return obj
            except:
                pass
        
        warn(f"{step_name}: Failed to parse JSON: {e}")
        return None


def _extract_numeric_id_from_identifier(identifier: str) -> Optional[str]:
    """
    Extract numeric ID from identifier, preserving leading zeros.

    Examples:
        BZZ003  → '003'   (preserve leading zeros from original)
        BZZ021  → '021'
        sub-01  → '01'    (handled by already_bids, but safe here too)
        patient001 → '001'

    NOTE: We preserve the original zero-padded string, NOT int conversion.
    """
    numbers = re.findall(r'\d+', identifier)
    if not numbers:
        return None
    # Return last numeric sequence as-is (preserving leading zeros)
    return numbers[-1]


def _generate_subject_id_mapping(subject_info: Dict[str, Any],
                                 user_hints: Dict[str, Any],
                                 id_strategy: str = "auto") -> Dict[str, Any]:
    """
    Generate subject ID mapping.

    Key rule: ALWAYS detect already-BIDS format and override strategy.
    sub-01 → '01'  (preserve leading zeros, never convert to int)
    """
    subject_records = subject_info.get("subject_records", [])

    if not subject_records:
        return {
            "id_mapping": {},
            "reverse_mapping": {},
            "strategy_used": "none",
            "metadata_columns": []
        }

    # ----------------------------------------------------------------
    # CRITICAL: Always detect already-BIDS format first, override any strategy
    # This handles: sub-01, sub-02, ... sub-10
    # ----------------------------------------------------------------
    all_already_bids = all(
        re.match(r'^sub-\w+$', rec["original_id"], re.IGNORECASE)
        for rec in subject_records
    )

    if all_already_bids:
        id_strategy = "already_bids"
        info(f"  Detected existing BIDS sub-XX format → using 'already_bids' strategy")

    elif id_strategy == "auto":
        has_site = subject_info.get("has_site_info", False)
        all_simple = all(
            rec["original_id"].isdigit() or len(rec["original_id"]) <= 3
            for rec in subject_records
        )
        if has_site:
            id_strategy = "semantic"
        elif len(subject_records) <= 10 and not all_simple:
            id_strategy = "semantic"
        else:
            id_strategy = "numeric"
        info(f"  Auto-selected ID strategy: {id_strategy}")

    id_mapping = {}
    reverse_mapping = {}
    metadata_columns = []

    # ----------------------------------------------------------------
    # already_bids: sub-01 → '01', sub-10 → '10'
    # Strip 'sub-' prefix, preserve leading zeros exactly as-is
    # ----------------------------------------------------------------
    if id_strategy == "already_bids":
        for rec in subject_records:
            original_id = rec["original_id"]
            # Strip 'sub-' prefix only, keep the rest verbatim
            bids_id = re.sub(r'^sub-', '', original_id)   # 'sub-01' → '01'
            id_mapping[original_id] = bids_id
            reverse_mapping[bids_id] = original_id
        metadata_columns = []

    # ----------------------------------------------------------------
    # numeric: extract trailing numbers, preserve leading zeros
    # BZZ003 → '003', patient021 → '021'
    # ----------------------------------------------------------------
    elif id_strategy == "numeric":
        extracted_numbers = {}
        for rec in subject_records:
            original_id = rec["original_id"]
            numeric_id = _extract_numeric_id_from_identifier(original_id)
            if numeric_id:
                extracted_numbers[original_id] = numeric_id

        if extracted_numbers and len(set(extracted_numbers.values())) == len(extracted_numbers):
            info(f"  ✓ Using extracted numeric IDs (with original zero-padding)")
            for original_id, numeric_id in extracted_numbers.items():
                id_mapping[original_id] = numeric_id
                reverse_mapping[numeric_id] = original_id
        else:
            info(f"  → Using sequential numbering (extracted IDs not unique)")
            for i, rec in enumerate(subject_records, 1):
                original_id = rec["original_id"]
                bids_id = str(i)
                id_mapping[original_id] = bids_id
                reverse_mapping[bids_id] = original_id

        metadata_columns = ["original_id"]
        if subject_info.get("has_site_info"):
            metadata_columns.append("site")

    # ----------------------------------------------------------------
    # semantic: remove special chars, preserve full identifier
    # NewYork_sub04856 → 'NewYorksub04856'
    # ----------------------------------------------------------------
    elif id_strategy == "semantic":
        for rec in subject_records:
            original_id = rec["original_id"]
            bids_id = re.sub(r'[^a-zA-Z0-9]', '', original_id)
            id_mapping[original_id] = bids_id
            reverse_mapping[bids_id] = original_id
        metadata_columns = []
        if subject_info.get("has_site_info"):
            metadata_columns.append("site")

    return {
        "id_mapping": id_mapping,
        "reverse_mapping": reverse_mapping,
        "strategy_used": id_strategy,
        "metadata_columns": metadata_columns
    }


def _extract_subjects_from_directory_structure(all_files: List[str]) -> Dict[str, Any]:
    """Extract subjects from directory structure."""
    patterns = [
        (r'([A-Za-z]+)_sub(\d+)', True, 2, 1, "site_prefixed"),
        (r'sub-(\d+)', False, 1, None, "standard_bids"),
        (r'subject[_-]?(\d+)', False, 1, None, "simple"),
        (r'^(\d{3,})$', False, 1, None, "numeric_only"),
    ]
    
    subject_records = []
    seen_ids = set()
    
    for filepath in all_files:
        parts = filepath.split('/')
        
        for part in parts[:2]:
            for pattern_regex, has_site, id_group, site_group, pattern_name in patterns:
                match = re.match(pattern_regex, part, re.IGNORECASE)
                if match:
                    numeric_id = match.group(id_group)
                    original_id = match.group(0)
                    
                    if original_id in seen_ids:
                        break
                    
                    seen_ids.add(original_id)
                    
                    site = match.group(site_group) if has_site and site_group else None
                    
                    subject_records.append({
                        "original_id": original_id,
                        "numeric_id": numeric_id,
                        "site": site,
                        "pattern_name": pattern_name
                    })
                    break
    
    if len(subject_records) == 0:
        return {"success": False, "method": "directory_structure"}
    
    subject_records.sort(key=lambda x: int(x["numeric_id"]) if x["numeric_id"].isdigit() else 0)
    has_sites = any(rec["site"] is not None for rec in subject_records)
    
    return {
        "success": True,
        "method": "directory_structure",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "has_site_info": has_sites
    }


def _extract_subjects_from_flat_filenames(all_files: List[str], 
                                          filename_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract subjects from flat structure - UNIVERSAL VERSION.
    
    CRITICAL: Use ALL files, not just samples!
    
    Strategy:
    1. Extract base identifier from every filename
    2. Group by unique identifiers
    3. Sort by extracted numeric ID if present
    """
    identifier_to_files = defaultdict(list)
    
    # CRITICAL: Use ALL files, not just samples
    for filepath in all_files:
        filename = filepath.split('/')[-1]
        name_no_ext = filename.rsplit('.', 1)[0]
        
        # Extract base identifier (alphanumeric before first underscore or complete)
        # Examples: BZZ003 → BZZ003, patient001_rest → patient001
        base_id_match = re.match(r'^([A-Za-z0-9\-]+)', name_no_ext)
        
        if base_id_match:
            identifier = base_id_match.group(1)
            identifier_to_files[identifier].append(filepath)
    
    if not identifier_to_files:
        return {"success": False, "method": "flat_filename"}
    
    # Sort identifiers by extracted numeric ID if possible
    def sort_key(identifier):
        numeric_id = _extract_numeric_id_from_identifier(identifier)
        return int(numeric_id) if numeric_id else 999999
    
    sorted_identifiers = sorted(identifier_to_files.keys(), key=sort_key)
    
    # Generate subject records
    subject_records = []
    for i, identifier in enumerate(sorted_identifiers, 1):
        files = identifier_to_files[identifier]
        
        subject_records.append({
            "original_id": identifier,
            "numeric_id": str(i),  # Sequential for now (will be updated by ID mapping)
            "site": None,
            "pattern_name": "filename_identifier",
            "file_count": len(files)
        })
    
    info(f"  Detected {len(subject_records)} unique identifiers:")
    for rec in subject_records[:10]:
        info(f"    '{rec['original_id']}': {rec['file_count']} file(s)")
    if len(subject_records) > 10:
        info(f"    ... and {len(subject_records) - 10} more")
    
    return {
        "success": True,
        "method": "flat_filename_identifiers",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "has_site_info": False,
        "variants_by_subject": {},
        "python_generated_filename_rules": []
    }


def _update_participants_with_metadata(plan: Dict[str, Any], out_dir: Path) -> None:
    """Update participants.tsv with metadata from LLM plan.
    
    Skips original_id column when its value is identical to participant_id
    (i.e. already_bids strategy where sub-01 → sub-01 adds no information).
    """
    participants_path = out_dir / 'participants.tsv'
    participant_metadata = plan.get('participant_metadata', {})

    if not participant_metadata:
        return

    # Collect all additional columns from metadata
    first_subject = list(participant_metadata.values())[0]
    additional_columns = list(first_subject.keys())

    # FIX: Remove original_id column if it's redundant
    # (i.e. original_id == participant_id for all subjects)
    def _is_redundant(col):
        if col != 'original_id':
            return False
        for bids_id, metadata in participant_metadata.items():
            orig = metadata.get('original_id', '')
            # participant_id in TSV is f"sub-{bids_id}"
            if orig != f"sub-{bids_id}":
                return False
        return True

    additional_columns = [c for c in additional_columns if not _is_redundant(c)]

    if not additional_columns:
        info("  ✓ No additional metadata columns needed (original_id is redundant)")
        return

    columns = ['participant_id'] + additional_columns
    lines = ['\t'.join(columns) + '\n']

    subject_ids = sorted(participant_metadata.keys(),
                        key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    for subj_id in subject_ids:
        metadata = participant_metadata[subj_id]
        bids_id = f"sub-{subj_id}"
        row = [bids_id] + [str(metadata.get(col, 'n/a')) for col in additional_columns]
        lines.append('\t'.join(row) + '\n')

    participants_path.write_text(''.join(lines))
    info(f"  ✓ Updated participants.tsv ({len(subject_ids)} subjects, "
         f"columns: {additional_columns})")


def _generate_participants_tsv_from_python(subject_info: Dict[str, Any],
                                            out_dir: Path,
                                            id_mapping: Dict[str, str],
                                            metadata_columns: List[str]) -> None:
    """Generate participants.tsv. Always regenerate to avoid stale data."""
    participants_path = out_dir / 'participants.tsv'

    if participants_path.exists():
        info("  Removing old participants.tsv to regenerate")
        participants_path.unlink()

    subject_records = subject_info.get("subject_records", [])

    columns = ['participant_id']
    if 'original_id' in metadata_columns:
        columns.append('original_id')
    if 'site' in metadata_columns:
        columns.append('site')

    lines = ['\t'.join(columns) + '\n']

    # Safe sort: try numeric first, fallback to string
    def _safe_sort_key(rec):
        bids_id = id_mapping.get(rec['original_id'], rec.get('numeric_id', '0'))
        try:
            return (0, int(bids_id))
        except (ValueError, TypeError):
            return (1, str(bids_id))

    sorted_records = sorted(subject_records, key=_safe_sort_key)

    for rec in sorted_records:
        original_id = rec['original_id']
        bids_id = id_mapping.get(original_id, rec.get('numeric_id', '?'))

        row = [f"sub-{bids_id}"]
        if 'original_id' in columns:
            row.append(original_id)
        if 'site' in columns:
            row.append(rec.get('site', 'unknown'))

        lines.append('\t'.join(row) + '\n')

    participants_path.write_text(''.join(lines))

    all_bids_ids = [id_mapping.get(r['original_id'], '?') for r in sorted_records]
    info(f"  ✓ Generated participants.tsv ({len(sorted_records)} subjects)")
    info(f"    BIDS IDs: {all_bids_ids[:5]}{'...' if len(all_bids_ids) > 5 else ''}")


def _apply_python_rules_to_plan(plan_yaml: Dict[str, Any], 
                                subject_info: Dict[str, Any],
                                id_mapping_info: Dict[str, Any]) -> None:
    """Apply Python's subject detection."""
    subject_records = subject_info.get("subject_records", [])
    id_mapping = id_mapping_info.get('id_mapping', {})
    
    subject_labels = [
        id_mapping.get(rec["original_id"], rec["numeric_id"])
        for rec in subject_records
    ]
    
    plan_yaml['subjects'] = {
        'labels': subject_labels,
        'count': len(subject_labels),
        'source': 'python_extracted',
        'id_strategy': id_mapping_info.get('strategy_used')
    }
    
    plan_yaml['assignment_rules'] = []
    for rec in subject_records:
        original_id = rec["original_id"]
        bids_id = id_mapping.get(original_id, rec["numeric_id"])
        
        plan_yaml['assignment_rules'].append({
            'subject': bids_id,
            'original': original_id,
            'match': [f"*{original_id}*"]
        })
    
    info(f"  → Applied {len(subject_labels)} subjects")


def _enhance_plan_with_python_details(plan_yaml: Dict[str, Any], 
                                      subject_info: Dict[str, Any],
                                      id_mapping_info: Dict[str, Any]) -> None:
    """Enhance LLM plan with Python analysis."""
    if 'assignment_rules' not in plan_yaml or not plan_yaml['assignment_rules']:
        _apply_python_rules_to_plan(plan_yaml, subject_info, id_mapping_info)


def build_bids_plan(model: str, planning_inputs: Dict[str, Any], 
                   out_dir: Path, id_strategy: str = "auto") -> Dict[str, Any]:
    """Build BIDS plan."""
    info("=== Generating Unified BIDS Plan ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files = evidence_bundle.get("all_files", [])
    
    staging_dir = out_dir / '_staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    info("Step 1: Python extracting subjects from ALL files...")
    
    subject_info = _extract_subjects_from_directory_structure(all_files)
    
    if not subject_info["success"]:
        info("  Trying flat filename analysis on ALL files...")
        # CRITICAL: Pass ALL files, not just samples
        subject_info = _extract_subjects_from_flat_filenames(all_files, {})
    
    python_subject_count = subject_info.get("subject_count", 0)
    
    if subject_info["success"]:
        info(f"  ✓ Extracted {python_subject_count} subjects")
    
    info("\nStep 1.5: Generating ID mapping...")
    id_mapping_info = _generate_subject_id_mapping(
        subject_info, 
        evidence_bundle.get("user_hints", {}), 
        id_strategy
    )
    
    info(f"  Strategy: {id_mapping_info['strategy_used']}")
    if id_mapping_info['id_mapping']:
        sample_mappings = list(id_mapping_info['id_mapping'].items())[:10]
        info(f"  ID Mappings:")
        for orig, bids in sample_mappings:
            info(f"    '{orig}' → sub-{bids}")
        if len(id_mapping_info['id_mapping']) > 10:
            info(f"    ... and {len(id_mapping_info['id_mapping']) - 10} more")
    
    subject_analysis_path = staging_dir / 'subject_analysis.json'
    subject_info['id_mapping'] = id_mapping_info
    write_json(subject_analysis_path, subject_info)
    
    info("\nStep 2: Generating participants.tsv...")
    
    if subject_info["success"]:
        _generate_participants_tsv_from_python(
            subject_info,
            out_dir,
            id_mapping_info['id_mapping'],
            id_mapping_info['metadata_columns']
        )
    
    info("\nStep 3: Calling LLM...")
    
    optimized_bundle = {
        "root": evidence_bundle.get("root"),
        "counts_by_ext": evidence_bundle.get("counts_by_ext", {}),
        "user_hints": evidence_bundle.get("user_hints", {}),
        "file_count": len(all_files),
        "sample_files": [s['relpath'] for s in evidence_bundle.get('samples', [])],
        "python_subject_analysis": {
            "success": subject_info["success"],
            "method": subject_info.get("method"),
            "subject_count": python_subject_count,
            "subject_examples": [
                {"original": rec["original_id"], "bids_id": id_mapping_info['id_mapping'].get(rec["original_id"])}
                for rec in subject_info.get("subject_records", [])[:10]
            ],
            "id_mapping": id_mapping_info
        }
    }
    
    evidence_json = json.dumps(optimized_bundle, indent=2)
    plan_response = llm_bids_plan(model, evidence_json)
    
    if not plan_response:
        fatal("LLM failed")
    
    try:
        text = plan_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        plan_yaml = yaml.safe_load(text.strip())
    except yaml.YAMLError as e:
        fatal(f"YAML error: {e}")
    
    info("\nStep 4: Validating...")
    
    llm_count = plan_yaml.get('subjects', {}).get('count', 0)
    user_count = evidence_bundle.get("user_hints", {}).get("n_subjects")
    
    info(f"  LLM: {llm_count}, Python: {python_subject_count}, User: {user_count}")
    
    # Use Python's analysis (most reliable for flat structures)
    if python_subject_count > 0:
        info("  → Using Python's complete file analysis")
        _apply_python_rules_to_plan(plan_yaml, subject_info, id_mapping_info)
    
    plan_yaml['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'model': model,
        'id_strategy': id_mapping_info.get('strategy_used')
    }
    
    plan_path = staging_dir / BIDS_PLAN
    write_yaml(plan_path, plan_yaml)
    info(f"\n✓ Plan saved: {plan_path}")
    
    info("\nStep 5: Updating participants.tsv...")
    if 'participant_metadata' in plan_yaml:
        _update_participants_with_metadata(plan_yaml, out_dir)
    
    info(f"\n=== Complete: {plan_yaml.get('subjects', {}).get('count', 0)} subjects ===")
    
    return {"status": "ok", "warnings": [], "questions": []}


def nirs_plan_headers(model: str, planning_inputs: Dict[str, Any], 
                     out_dir: Path) -> Dict[str, Any]:
    """NIRS header planning."""
    info("=== Planning NIRS headers ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    evidence_json = json.dumps(evidence_bundle, indent=2)
    
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
    info(f"✓ NIRS headers saved")
    
    return {"warnings": [], "questions": []}


def mri_plan_voxel_mappings(model: str, planning_inputs: Dict[str, Any],
                           out_dir: Path) -> Dict[str, Any]:
    """MRI voxel mapping planning."""
    return {"warnings": [], "questions": []}
