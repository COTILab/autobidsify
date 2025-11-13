# converters/planner.py
# Python-first planning: deterministic extraction + LLM validation

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
    """
    Extract subjects from directory structure (hierarchical datasets).
    
    Returns:
        {
            "success": True/False,
            "method": "directory_structure",
            "subject_records": [...],
            "has_site_info": True/False
        }
    """
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
        
        for part in parts[:2]:  # Check first 2 levels
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
    """
    Extract subjects and variants from flat structure (all files in one directory).
    
    Strategy:
    1. Use filename_analysis to get dominant prefixes (subjects)
    2. Extract all variants (body parts, tasks) from sample filenames
    3. Generate complete filename_rules (prefix × variant combinations)
    
    Args:
        sample_files: Sampled file paths
        filename_analysis: From evidence bundle
    
    Returns:
        {
            "success": True/False,
            "method": "flat_filename",
            "subject_records": [...],
            "variants_by_subject": {...},
            "python_generated_filename_rules": [...]
        }
    """
    # Get dominant prefixes from token analysis
    stats = filename_analysis.get('python_statistics', {})
    dominant_prefixes = stats.get('dominant_prefixes', [])
    
    if not dominant_prefixes:
        return {"success": False, "method": "flat_filename"}
    
    # Create subject records
    subject_records = []
    for i, prefix_info in enumerate(dominant_prefixes, 1):
        prefix = prefix_info['prefix']
        subject_records.append({
            "original_id": prefix,
            "numeric_id": str(i),
            "site": None,
            "pattern_name": "filename_prefix"
        })
    
    # Extract variants from sample filenames
    variants_by_prefix = defaultdict(set)
    
    for filepath in sample_files:
        filename = filepath.split('/')[-1]
        
        # Match to subject prefix
        matched_prefix = None
        for prefix_info in dominant_prefixes:
            if filename.startswith(prefix_info['prefix']):
                matched_prefix = prefix_info['prefix']
                break
        
        if not matched_prefix:
            continue
        
        # Extract variant (body part, task, etc.)
        # Common patterns:
        # - VHMCT1mm-Hip (134).dcm → "Hip"
        # - subject01_rest.nii → "rest"
        # - patient_A_task_motor.nii → "motor"
        
        variant = None
        
        # Pattern 1: Dash-separated (VHMCT1mm-Hip)
        match = re.search(r'-([A-Za-z]+)\s', filename)
        if match:
            variant = match.group(1)
        
        # Pattern 2: Underscore-separated (subject01_rest)
        if not variant:
            match = re.search(r'_([A-Za-z]+)(?:_|\.)', filename)
            if match:
                variant = match.group(1)
        
        # Pattern 3: Common keywords
        if not variant:
            keywords = ['rest', 'task', 'head', 'hip', 'brain', 'shoulder', 'knee', 'ankle', 'pelvis']
            for kw in keywords:
                if kw in filename.lower():
                    variant = kw
                    break
        
        if variant:
            variants_by_prefix[matched_prefix].add(variant)
    
    # Generate filename_rules for ALL combinations
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
# Main Planning Function
# ============================================================================

def build_bids_plan(model: str, planning_inputs: Dict[str, Any], 
                   out_dir: Path) -> Dict[str, Any]:
    """Build BIDS plan with Python-first subject detection."""
    info("=== Generating Unified BIDS Plan ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files = evidence_bundle.get("all_files", [])
    
    staging_dir = out_dir / '_staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Python extracts subjects (deterministic)
    # ========================================================================
    info("Step 1: Python extracting subjects...")
    
    # Try directory structure first
    subject_info = _extract_subjects_from_directory_structure(all_files)
    
    # If failed, try flat filename extraction
    if not subject_info["success"]:
        info("  No subjects in directory structure, trying filename analysis...")
        
        filename_analysis = evidence_bundle.get("filename_analysis", {})
        sample_file_objects = evidence_bundle.get('samples', [])
        sample_files = [s['relpath'] for s in sample_file_objects]
        
        subject_info = _extract_subjects_from_flat_filenames(sample_files, filename_analysis)
    
    # Report results
    if subject_info["success"]:
        info(f"  ✓ Extracted {subject_info['subject_count']} subjects via {subject_info['method']}")
        
        # Show variants if available
        if 'variants_by_subject' in subject_info:
            for prefix, variants in subject_info['variants_by_subject'].items():
                info(f"    {prefix}: {len(variants)} variants ({', '.join(variants[:5])}{'...' if len(variants) > 5 else ''})")
    else:
        warn("  ⚠ Python extraction failed, will rely on LLM")
    
    # Save subject analysis
    subject_analysis_path = staging_dir / 'subject_analysis.json'
    write_json(subject_analysis_path, subject_info)
    
    # ========================================================================
    # STEP 2: Generate participants.tsv (if Python succeeded)
    # ========================================================================
    info("Step 2: Generating participants.tsv...")
    
    if subject_info["success"] and subject_info.get("subject_records"):
        _generate_participants_tsv_from_python(subject_info, out_dir)
    else:
        info("  Deferring to LLM")
    
    # ========================================================================
    # STEP 3: Prepare LLM payload
    # ========================================================================
    info("Step 3: Preparing LLM payload...")
    
    # Use evidence bundle samples
    sample_file_objects = evidence_bundle.get('samples', [])
    sample_files = [s['relpath'] for s in sample_file_objects]
    
    # Build payload
    optimized_bundle = {
        "root": evidence_bundle.get("root"),
        "counts_by_ext": evidence_bundle.get("counts_by_ext", {}),
        "user_hints": evidence_bundle.get("user_hints", {}),
        "file_count": len(all_files),
        "sample_files": sample_files,
        
        # Python's analysis results
        "python_subject_analysis": {
            "success": subject_info["success"],
            "method": subject_info.get("method"),
            "subject_count": subject_info.get("subject_count", 0),
            "has_site_info": subject_info.get("has_site_info", False),
            
            # Sample subject IDs (not all)
            "subject_examples": [
                {"original": rec["original_id"], "numeric": rec["numeric_id"]}
                for rec in subject_info.get("subject_records", [])[:5]
            ],
            
            # Variants extracted by Python
            "variants": subject_info.get("variants_by_subject", {}),
            
            # Pre-generated filename rules (if any)
            "python_generated_filename_rules": subject_info.get("python_generated_filename_rules", [])
        }
    }
    
    info(f"  Payload includes {len(subject_info.get('python_generated_filename_rules', []))} Python-generated rules")
    
    # ========================================================================
    # STEP 4: Call LLM (for validation and semantic understanding)
    # ========================================================================
    info("Step 4: Calling LLM for validation and semantic analysis...")
    
    evidence_json = json.dumps(optimized_bundle, indent=2)
    plan_response = llm_bids_plan(model, evidence_json)
    
    if not plan_response:
        fatal("LLM failed to generate BIDS plan")
    
    try:
        # Remove markdown fences if present
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
    # STEP 5: Merge Python's results into LLM's plan
    # ========================================================================
    info("Step 5: Merging Python analysis into plan...")
    
    if subject_info["success"]:
        # Use Python's subject list (complete)
        subject_labels = [rec["numeric_id"] for rec in subject_info.get("subject_records", [])]
        
        plan_yaml['subjects'] = {
            'labels': subject_labels,
            'count': len(subject_labels),
            'source': 'python_extracted'
        }
        
        # Generate assignment_rules from Python's data
        plan_yaml['assignment_rules'] = []
        for rec in subject_info.get("subject_records", []):
            rule = {
                'subject': rec["numeric_id"],
                'original': rec["original_id"],
                'match': [f"**/{rec['original_id']}*", f"**/{rec['original_id']}/**"]
            }
            if rec.get("site"):
                rule['site'] = rec["site"]
            
            plan_yaml['assignment_rules'].append(rule)
        
        info(f"  ✓ Added {len(subject_labels)} subjects to plan")
        
        # Use Python-generated filename_rules if available
        python_rules = subject_info.get("python_generated_filename_rules", [])
        if python_rules:
            # Find the first mapping and inject Python's rules
            mappings = plan_yaml.get('mappings', [])
            if mappings:
                # Replace or merge filename_rules
                llm_rules = mappings[0].get('filename_rules', [])
                
                if not llm_rules:
                    # LLM didn't generate rules, use Python's
                    mappings[0]['filename_rules'] = python_rules
                    info(f"  ✓ Using {len(python_rules)} Python-generated filename rules")
                else:
                    # LLM generated some rules, merge with Python's
                    # Python's rules are more complete, use them
                    mappings[0]['filename_rules'] = python_rules
                    info(f"  ✓ Replaced LLM rules with {len(python_rules)} Python-generated rules")
    
    # Generate participants.tsv from LLM metadata (if provided)
    if 'participant_metadata' in plan_yaml and not subject_info.get("has_site_info"):
        _generate_participants_from_llm_metadata(plan_yaml, out_dir)
    
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
    info(f"✓ BIDS Plan saved: {plan_path}")
    
    if not plan_path.exists():
        fatal(f"BIDS Plan file was not created at: {plan_path}")
    
    return {
        "warnings": plan_yaml.get("questions", []),
        "questions": [],
        "subject_info": subject_info
    }


def _generate_participants_tsv_from_python(subject_info: Dict[str, Any], 
                                            out_dir: Path) -> None:
    """Generate participants.tsv from Python's subject records."""
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
    info(f"  ✓ Generated participants.tsv ({len(subject_records)} subjects)")


def _generate_participants_from_llm_metadata(plan: Dict[str, Any], out_dir: Path) -> None:
    """Generate participants.tsv from LLM's participant_metadata."""
    participants_path = out_dir / 'participants.tsv'
    
    if participants_path.exists():
        return
    
    participant_metadata = plan.get('participant_metadata', {})
    if not participant_metadata:
        return
    
    first_subject = list(participant_metadata.values())[0]
    columns = ['participant_id'] + list(first_subject.keys())
    
    lines = ['\t'.join(columns) + '\n']
    
    for subj_id in sorted(participant_metadata.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        metadata = participant_metadata[subj_id]
        bids_id = f"sub-{subj_id}"
        
        row = [bids_id]
        for col in columns[1:]:
            row.append(str(metadata.get(col, 'n/a')))
        
        lines.append('\t'.join(row) + '\n')
    
    participants_path.write_text(''.join(lines))
    info(f"  ✓ Generated participants.tsv with metadata")


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
