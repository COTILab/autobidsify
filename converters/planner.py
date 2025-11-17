# converters/planner.py

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import re
from datetime import datetime
from collections import defaultdict
from utils import write_json, read_json, write_yaml, info, warn, fatal, write_text
from constants import SEVERITY_BLOCK
from llm import llm_nirs_draft, llm_nirs_normalize, llm_mri_voxel_draft, llm_mri_voxel_final, llm_bids_plan

# Local constants
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
        warn(f"Preview: {text[:500]}...")
        return None


# ============================================================================
# Python-First Subject Detection
# ============================================================================

def _extract_subjects_from_directory_structure(all_files: List[str]) -> Dict[str, Any]:
    """Extract subjects from directory structure (hierarchical datasets)."""
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


def _extract_subjects_from_flat_filenames(sample_files: List[str], 
                                          filename_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract subjects and variants from flat structure."""
    stats = filename_analysis.get('python_statistics', {})
    dominant_prefixes = stats.get('dominant_prefixes', [])
    
    if not dominant_prefixes:
        return {"success": False, "method": "flat_filename"}
    
    subject_records = []
    for i, prefix_info in enumerate(dominant_prefixes, 1):
        prefix = prefix_info['prefix']
        subject_records.append({
            "original_id": prefix,
            "numeric_id": str(i),
            "site": None,
            "pattern_name": "filename_prefix"
        })
    
    variants_by_prefix = defaultdict(set)
    
    for filepath in sample_files:
        filename = filepath.split('/')[-1]
        
        matched_prefix = None
        for prefix_info in dominant_prefixes:
            if filename.startswith(prefix_info['prefix']):
                matched_prefix = prefix_info['prefix']
                break
        
        if not matched_prefix:
            continue
        
        variant = None
        match = re.search(r'-([A-Za-z]+)\s', filename)
        if match:
            variant = match.group(1)
        
        if not variant:
            match = re.search(r'_([A-Za-z]+)(?:_|\.)', filename)
            if match:
                variant = match.group(1)
        
        if not variant:
            keywords = ['rest', 'task', 'head', 'hip', 'brain', 'shoulder', 'knee', 'ankle', 'pelvis']
            for kw in keywords:
                if kw in filename.lower():
                    variant = kw
                    break
        
        if variant:
            variants_by_prefix[matched_prefix].add(variant)
    
    python_generated_rules = []
    
    for i, prefix_info in enumerate(dominant_prefixes, 1):
        prefix = prefix_info['prefix']
        subject_id = str(i)
        variants = sorted(variants_by_prefix.get(prefix, []))
        
        for variant in variants:
            rule = {
                "match_pattern": f"{prefix}.*{variant}.*\\.dcm",
                "bids_template": f"sub-{subject_id}_acq-ct{variant.lower()}_T1w.nii.gz"
            }
            python_generated_rules.append(rule)
    
    info(f"  Python generated {len(python_generated_rules)} filename rules from samples")
    
    return {
        "success": True,
        "method": "flat_filename",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "has_site_info": False,
        "variants_by_subject": {k: sorted(v) for k, v in variants_by_prefix.items()},
        "python_generated_filename_rules": python_generated_rules
    }


# ============================================================================
# Participant Metadata Handling
# ============================================================================

def _update_participants_with_metadata(plan: Dict[str, Any], out_dir: Path) -> None:
    """
    Use LLM participant_metadata extracted by LLM to update participants.tsv
    
    including provenance tracking
    """
    participants_path = out_dir / 'participants.tsv'
    participant_metadata = plan.get('participant_metadata', {})
    metadata_provenance = plan.get('metadata_provenance', {})
    
    if not participant_metadata:
        info("  No participant metadata in plan")
        
        # check whether it's insufficient_evidence
        if metadata_provenance.get('status') == 'insufficient_evidence':
            info("  ℹ LLM reported: insufficient evidence for metadata extraction")
            info(f"    Reason: {metadata_provenance.get('reasoning', 'N/A')}")
        
        return
    
    # check whether it contains complete participants.tsv
    if participants_path.exists():
        existing_content = participants_path.read_text()
        first_line = existing_content.split('\n')[0]
        
        # check whether it contains metadata cols
        existing_columns = set(first_line.split('\t'))
        new_columns = set(list(list(participant_metadata.values())[0].keys()))
        
        if new_columns.issubset(existing_columns):
            info("  ✓ participants.tsv already contains all metadata columns")
            return
        
        info("  Updating existing participants.tsv with additional metadata columns...")
    else:
        info("  Creating participants.tsv with metadata...")
    
    # obtain all columns
    first_subject = list(participant_metadata.values())[0]
    additional_columns = list(first_subject.keys())
    columns = ['participant_id'] + additional_columns
    
    # generate TSV
    lines = ['\t'.join(columns) + '\n']
    
    # order in terms of subject ID
    subject_ids = sorted(participant_metadata.keys(), 
                        key=lambda x: int(x) if x.isdigit() else 0)
    
    for subj_id in subject_ids:
        metadata = participant_metadata[subj_id]
        bids_id = f"sub-{subj_id}"
        
        row = [bids_id]
        for col in additional_columns:
            value = metadata.get(col, 'n/a')
            row.append(str(value))
        
        lines.append('\t'.join(row) + '\n')
    
    participants_path.write_text(''.join(lines))
    
    info(f"  ✓ Updated participants.tsv with {len(additional_columns)} metadata column(s):")
    info(f"    Columns: {', '.join(columns)}")
    info(f"    Subjects: {len(subject_ids)}")
    
    # show metadata provenance information
    if metadata_provenance:
        info(f"\n  Metadata provenance information:")
        for field, prov_info in metadata_provenance.items():
            if isinstance(prov_info, dict) and 'final_confidence' in prov_info:
                confidence = prov_info.get('final_confidence', 0)
                action = prov_info.get('recommended_action', 'unknown')
                info(f"    {field}: confidence={confidence:.2f}, action={action}")
    
    # show first-5-row preview
    if len(subject_ids) <= 5:
        info(f"\n  Content preview:")
        for line in lines:
            info(f"    {line.rstrip()}")


def _generate_participants_tsv_from_python(subject_info: Dict[str, Any], 
                                            out_dir: Path) -> None:
    """Generate basic participants.tsv from Python's subject records."""
    participants_path = out_dir / 'participants.tsv'
    
    if participants_path.exists():
        info("  ✓ participants.tsv already exists")
        return
    
    subject_records = subject_info.get("subject_records", [])
    has_site_info = subject_info.get("has_site_info", False)
    
    columns = ['participant_id']
    if has_site_info:
        columns.append('site')
    
    lines = ['\t'.join(columns) + '\n']
    
    for rec in subject_records:
        bids_id = f"sub-{rec['numeric_id']}"
        row = [bids_id]
        
        if has_site_info:
            row.append(rec.get('site', 'unknown'))
        
        lines.append('\t'.join(row) + '\n')
    
    participants_path.write_text(''.join(lines))
    info(f"  ✓ Generated participants.tsv (basic version, {len(subject_records)} subjects)")
    info(f"    Note: Will be updated with metadata in Step 8 if available")


# ============================================================================
# Main Planning Function
# ============================================================================

def build_bids_plan(model: str, planning_inputs: Dict[str, Any], 
                   out_dir: Path) -> Dict[str, Any]:
    """Build BIDS plan with evidence-based participant metadata extraction."""
    info("=== Generating Unified BIDS Plan ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files = evidence_bundle.get("all_files", [])
    
    staging_dir = out_dir / '_staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Python extracts subjects (deterministic)
    # ========================================================================
    info("Step 1: Python extracting subjects...")
    
    subject_info = _extract_subjects_from_directory_structure(all_files)
    
    if not subject_info["success"]:
        info("  No subjects in directory structure, trying filename analysis...")
        
        filename_analysis = evidence_bundle.get("filename_analysis", {})
        sample_file_objects = evidence_bundle.get('samples', [])
        sample_files = [s['relpath'] for s in sample_file_objects]
        
        subject_info = _extract_subjects_from_flat_filenames(sample_files, filename_analysis)
    
    if subject_info["success"]:
        info(f"  ✓ Extracted {subject_info['subject_count']} subjects via {subject_info['method']}")
        
        if 'variants_by_subject' in subject_info:
            for prefix, variants in subject_info['variants_by_subject'].items():
                info(f"    {prefix}: {len(variants)} variants ({', '.join(variants[:5])}{'...' if len(variants) > 5 else ''})")
    else:
        warn("  ⚠ Python extraction failed, will rely on LLM")
    
    subject_analysis_path = staging_dir / 'subject_analysis.json'
    write_json(subject_analysis_path, subject_info)
    
    # ========================================================================
    # STEP 2: Generate basic participants.tsv
    # ========================================================================
    info("\nStep 2: Generating participants.tsv (basic version)...")
    
    if subject_info["success"] and subject_info.get("subject_records"):
        _generate_participants_tsv_from_python(subject_info, out_dir)
    else:
        info("  Deferring to LLM")
    
    # ========================================================================
    # STEP 3: Prepare LLM payload (inluding multi-source evidence)
    # ========================================================================
    info("\nStep 3: Preparing LLM payload with evidence...")
    
    sample_file_objects = evidence_bundle.get('samples', [])
    sample_files = [s['relpath'] for s in sample_file_objects]
    
    # NEW: include participant_metadata_evidence
    participant_evidence = evidence_bundle.get("participant_metadata_evidence", {})
    evidence_summary = participant_evidence.get("summary", {})
    
    info(f"  Evidence types found: {evidence_summary.get('total_evidence_types_found', 0)}/5")
    for evidence_type in evidence_summary.get('evidence_types', []):
        info(f"    ✓ {evidence_type}")
    
    optimized_bundle = {
        "root": evidence_bundle.get("root"),
        "counts_by_ext": evidence_bundle.get("counts_by_ext", {}),
        "user_hints": evidence_bundle.get("user_hints", {}),
        "file_count": len(all_files),
        "sample_files": sample_files,
        
        # NEW: add complete participant metadata evidence
        "participant_metadata_evidence": participant_evidence,
        
        "python_subject_analysis": {
            "success": subject_info["success"],
            "method": subject_info.get("method"),
            "subject_count": subject_info.get("subject_count", 0),
            "has_site_info": subject_info.get("has_site_info", False),
            "subject_examples": [
                {"original": rec["original_id"], "numeric": rec["numeric_id"]}
                for rec in subject_info.get("subject_records", [])[:5]
            ],
            "variants": subject_info.get("variants_by_subject", {}),
            "python_generated_filename_rules": subject_info.get("python_generated_filename_rules", [])
        }
    }
    
    info(f"  Payload includes {len(subject_info.get('python_generated_filename_rules', []))} Python-generated rules")
    
    # ========================================================================
    # STEP 4: Call LLM
    # ========================================================================
    info("\nStep 4: Calling LLM for plan generation...")
    
    evidence_json = json.dumps(optimized_bundle, indent=2)
    plan_response = llm_bids_plan(model, evidence_json)
    
    if not plan_response:
        fatal("LLM failed to generate BIDS plan")
    
    try:
        text = plan_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        
        plan_yaml = yaml.safe_load(text.strip())
    except yaml.YAMLError as e:
        warn(f"YAML parse error: {e}")
        fatal("Failed to parse BIDS plan YAML")
    
    # ========================================================================
    # STEP 5: Merge Python's results into plan
    # ========================================================================
    info("\nStep 5: Merging Python analysis into plan...")
    
    if subject_info["success"]:
        subject_labels = [rec["numeric_id"] for rec in subject_info.get("subject_records", [])]
        
        plan_yaml['subjects'] = {
            'labels': subject_labels,
            'count': len(subject_labels),
            'source': 'python_extracted'
        }
        
        plan_yaml['assignment_rules'] = []
        for rec in subject_info.get("subject_records", []):
            rule = {
                'subject': rec["numeric_id"],
                'original': rec["original_id"],
                'match': [f"**/{rec['original_id']}*", f"**/{rec['original_id']}/**"]
            }
            if rec.get("site"):
                rule['site'] = rec["site"]
            
            if subject_info.get("method") == "flat_filename":
                rule['prefix'] = rec["original_id"]
            
            plan_yaml['assignment_rules'].append(rule)
        
        info(f"  ✓ Added {len(subject_labels)} subjects to plan")
        
        python_rules = subject_info.get("python_generated_filename_rules", [])
        if python_rules:
            mappings = plan_yaml.get('mappings', [])
            if mappings:
                llm_rules = mappings[0].get('filename_rules', [])
                
                if not llm_rules:
                    mappings[0]['filename_rules'] = python_rules
                    info(f"  ✓ Using {len(python_rules)} Python-generated filename rules")
                else:
                    mappings[0]['filename_rules'] = python_rules
                    info(f"  ✓ Replaced LLM rules with {len(python_rules)} Python-generated rules")
    
    # ========================================================================
    # STEP 6: Add metadata
    # ========================================================================
    plan_yaml['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'model': model,
        'python_analysis': {
            'method': subject_info.get('method'),
            'success': subject_info.get('success'),
            'subject_count': subject_info.get('subject_count', 0)
        }
    }
    
    # ========================================================================
    # STEP 7: Save plan
    # ========================================================================
    plan_path = staging_dir / BIDS_PLAN
    write_yaml(plan_path, plan_yaml)
    info(f"\n✓ BIDS Plan saved: {plan_path}")
    
    if not plan_path.exists():
        fatal(f"BIDS Plan file was not created at: {plan_path}")
    
    # ========================================================================
    # STEP 8: Update participants.tsv with metadata (NEW!)
    # ========================================================================
    info("\nStep 8: Updating participants.tsv with LLM-extracted metadata...")
    
    if 'participant_metadata' in plan_yaml:
        _update_participants_with_metadata(plan_yaml, out_dir)
        info("  ✓ Participant metadata successfully applied")
    elif 'metadata_provenance' in plan_yaml:
        # Check whether is insufficient_evidence
        if plan_yaml['metadata_provenance'].get('status') == 'insufficient_evidence':
            info("  ℹ No metadata extracted (insufficient evidence)")
        else:
            info("  ℹ No participant metadata in plan")
    else:
        info("  ℹ No participant metadata or provenance information")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    info("\n=== BIDS Plan Generation Complete ===")
    info(f"Plan location: {plan_path}")
    info(f"Subjects detected: {subject_info.get('subject_count', 0)}")
    
    # FIXED: Check if participant_metadata is not None and not empty
    participant_metadata = plan_yaml.get('participant_metadata')
    if participant_metadata and isinstance(participant_metadata, dict) and len(participant_metadata) > 0:
        first_subject_metadata = list(participant_metadata.values())[0]
        if first_subject_metadata and isinstance(first_subject_metadata, dict):
            metadata_keys = list(first_subject_metadata.keys())
            info(f"Participant columns: participant_id, {', '.join(metadata_keys)}")
    
    if 'metadata_provenance' in plan_yaml:
        info(f"Metadata provenance tracking: ✓ enabled")
    
    return {
        "status": "ok",
        "warnings": plan_yaml.get("questions", []),
        "questions": [],
        "subject_info": subject_info,
        "metadata_extracted": bool(participant_metadata and isinstance(participant_metadata, dict) and len(participant_metadata) > 0)
    }


# ============================================================================
# Legacy Functions
# ============================================================================

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
    
    return {
        "warnings": normalized.get("warnings", []),
        "questions": normalized.get("questions", [])
    }


def mri_plan_voxel_mappings(model: str, planning_inputs: Dict[str, Any],
                           out_dir: Path) -> Dict[str, Any]:
    """MRI voxel mapping planning."""
    info("=== Planning MRI voxel mappings ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    evidence_json = json.dumps(evidence_bundle, indent=2)
    
    draft_response = llm_mri_voxel_draft(model, evidence_json)
    if not draft_response:
        return {"warnings": [], "questions": []}
    
    draft = _parse_llm_json_response(draft_response, "mri_voxel_draft")
    if not draft:
        return {"warnings": [], "questions": []}
    
    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    write_json(staging_dir / VOXEL_DRAFT, draft)
    
    final_response = llm_mri_voxel_final(model, json.dumps(draft, indent=2))
    if not final_response:
        return {"warnings": [], "questions": []}
    
    final = _parse_llm_json_response(final_response, "mri_voxel_final")
    if not final:
        return {"warnings": [], "questions": []}
    
    write_json(staging_dir / VOXEL_FINAL_PLAN, final)
    info(f"✓ MRI voxel mappings saved")
    
    return {
        "warnings": final.get("warnings", []),
        "questions": final.get("questions", [])
    }
