# converters/planner.py
# MODIFIED: Added JNIfTI detection and LLM hint support

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


def _update_participants_with_metadata(plan: Dict[str, Any], out_dir: Path) -> None:
    """Update participants.tsv with LLM-extracted metadata."""
    participants_path = out_dir / 'participants.tsv'
    participant_metadata = plan.get('participant_metadata', {})
    metadata_provenance = plan.get('metadata_provenance', {})
    
    if not participant_metadata:
        info("  No participant metadata in plan")
        
        if metadata_provenance.get('status') == 'insufficient_evidence':
            info("  ℹ LLM reported: insufficient evidence for metadata extraction")
            info(f"    Reason: {metadata_provenance.get('reasoning', 'N/A')}")
        
        return
    
    if participants_path.exists():
        existing_content = participants_path.read_text()
        first_line = existing_content.split('\n')[0]
        
        existing_columns = set(first_line.split('\t'))
        new_columns = set(list(list(participant_metadata.values())[0].keys()))
        
        if new_columns.issubset(existing_columns):
            info("  ✓ participants.tsv already contains all metadata columns")
            return
        
        info("  Updating existing participants.tsv with additional metadata columns...")
    else:
        info("  Creating participants.tsv with metadata...")
    
    first_subject = list(participant_metadata.values())[0]
    additional_columns = list(first_subject.keys())
    columns = ['participant_id'] + additional_columns
    
    lines = ['\t'.join(columns) + '\n']
    
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
    
    if metadata_provenance:
        info(f"\n  Metadata provenance information:")
        for field, prov_info in metadata_provenance.items():
            if isinstance(prov_info, dict) and 'final_confidence' in prov_info:
                confidence = prov_info.get('final_confidence', 0)
                action = prov_info.get('recommended_action', 'unknown')
                info(f"    {field}: confidence={confidence:.2f}, action={action}")
    
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


def build_bids_plan(model: str, planning_inputs: Dict[str, Any], 
                   out_dir: Path) -> Dict[str, Any]:
    """
    Build BIDS plan with LLM-first decision making.
    
    CRITICAL CHANGE: When LLM and Python disagree on subject count,
    trust LLM by default (it has full context including user description).
    """
    info("=== Generating Unified BIDS Plan ===")
    
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files = evidence_bundle.get("all_files", [])
    
    staging_dir = out_dir / '_staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Python analysis (提供统计数据)
    info("Step 1: Python extracting subjects...")
    
    subject_info = _extract_subjects_from_directory_structure(all_files)
    
    if not subject_info["success"]:
        info("  No subjects in directory structure, trying filename analysis...")
        
        filename_analysis = evidence_bundle.get("filename_analysis", {})
        sample_file_objects = evidence_bundle.get('samples', [])
        sample_files = [s['relpath'] for s in sample_file_objects]
        
        subject_info = _extract_subjects_from_flat_filenames(sample_files, filename_analysis)
    
    python_subject_count = subject_info.get("subject_count", 0)
    python_confidence = subject_info.get("success", False)
    
    if subject_info["success"]:
        info(f"  ✓ Python extracted {python_subject_count} subjects via {subject_info['method']}")
    else:
        warn("  ⚠ Python extraction failed")
    
    # 保存 Python 分析结果（供调试）
    subject_analysis_path = staging_dir / 'subject_analysis.json'
    write_json(subject_analysis_path, subject_info)
    
    # Step 2: 生成基础 participants.tsv（如果需要）
    info("\nStep 2: Generating participants.tsv (basic version)...")
    
    if subject_info["success"] and subject_info.get("subject_records"):
        _generate_participants_tsv_from_python(subject_info, out_dir)
    else:
        info("  Deferring to LLM")
    
    # Step 3: 准备 LLM payload
    info("\nStep 3: Preparing LLM payload with evidence...")
    
    sample_file_objects = evidence_bundle.get('samples', [])
    sample_files = [s['relpath'] for s in sample_file_objects]
    
    participant_evidence = evidence_bundle.get("participant_metadata_evidence", {})
    evidence_summary = participant_evidence.get("summary", {})
    
    info(f"  Evidence types found: {evidence_summary.get('total_evidence_types_found', 0)}/5")
    
    # Detect JNIfTI files
    counts_by_ext = evidence_bundle.get("counts_by_ext", {})
    jnifti_count = counts_by_ext.get('.jnii', 0) + counts_by_ext.get('.bnii', 0)
    
    if jnifti_count > 0:
        info(f"  Detected {jnifti_count} JNIfTI files (.jnii/.bnii)")
    
    # Build optimized payload
    optimized_bundle = {
        "root": evidence_bundle.get("root"),
        "counts_by_ext": counts_by_ext,
        "user_hints": evidence_bundle.get("user_hints", {}),
        "file_count": len(all_files),
        "sample_files": sample_files,
        
        "jnifti_hint": {
            "count": jnifti_count,
            "requires_conversion": jnifti_count > 0,
            "target_format": "nifti",
            "note": "JNIfTI files must be converted to NIfTI before BIDS compliance"
        } if jnifti_count > 0 else None,
        
        "participant_metadata_evidence": participant_evidence,
        
        # CRITICAL: Python 分析作为参考，不强制使用
        "python_subject_analysis": {
            "success": subject_info["success"],
            "method": subject_info.get("method"),
            "subject_count": python_subject_count,
            "confidence": "high" if python_confidence and python_subject_count > 1 else "low",
            "has_site_info": subject_info.get("has_site_info", False),
            "subject_examples": [
                {"original": rec["original_id"], "numeric": rec["numeric_id"]}
                for rec in subject_info.get("subject_records", [])[:5]
            ],
            "variants": subject_info.get("variants_by_subject", {}),
            "python_generated_filename_rules": subject_info.get("python_generated_filename_rules", []),
            # NEW: 标记这只是建议
            "note": "This is Python's statistical analysis. LLM should validate with full context."
        }
    }
    
    info(f"  Python suggests {python_subject_count} subjects (confidence: {'high' if python_confidence else 'low'})")
    if jnifti_count > 0:
        info(f"  Payload includes JNIfTI conversion hint ({jnifti_count} files)")
    
    # Step 4: Call LLM
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
    
    # ===================================================================
    # Step 5: LLM-FIRST DECISION LOGIC (CRITICAL CHANGE)
    # ===================================================================
    info("\nStep 5: Validating LLM plan with Python analysis...")
    
    # Extract subject counts
    llm_subjects = plan_yaml.get('subjects', {})
    llm_subject_count = llm_subjects.get('count', 0)
    llm_subject_labels = llm_subjects.get('labels', [])
    
    user_provided_count = evidence_bundle.get("user_hints", {}).get("n_subjects")
    
    info(f"  LLM analysis: {llm_subject_count} subjects")
    info(f"  Python analysis: {python_subject_count} subjects")
    if user_provided_count:
        info(f"  User provided: {user_provided_count} subjects")
    
    # Decision logic
    if llm_subject_count == 0:
        # LLM 没有检测到主题（异常情况）
        warn("  ⚠ LLM did not detect subjects!")
        
        if python_subject_count > 0:
            info("  → Using Python's analysis as fallback")
            # 使用 Python 的结果填充计划
            subject_labels = [rec["numeric_id"] for rec in subject_info.get("subject_records", [])]
            plan_yaml['subjects'] = {
                'labels': subject_labels,
                'count': len(subject_labels),
                'source': 'python_fallback'
            }
            _apply_python_rules_to_plan(plan_yaml, subject_info)
        else:
            fatal("  ✗ Both LLM and Python failed to detect subjects")
    
    elif llm_subject_count == python_subject_count:
        # 两者一致
        info(f"  ✓ LLM and Python agree: {llm_subject_count} subjects")
        info("  → Using LLM's plan (with Python validation)")
        # 保持 LLM 的计划，但可以补充 Python 的细节
        _enhance_plan_with_python_details(plan_yaml, subject_info)
    
    elif user_provided_count:
        # 用户明确指定，以用户为准
        info(f"  → User explicitly specified {user_provided_count} subjects")
        
        if llm_subject_count == user_provided_count:
            info("  ✓ LLM matches user specification")
            # 使用 LLM 的计划
            pass
        elif python_subject_count == user_provided_count:
            info("  ✓ Python matches user specification")
            info("  → Using Python's analysis")
            _apply_python_rules_to_plan(plan_yaml, subject_info)
        else:
            warn(f"  ⚠ Neither LLM ({llm_subject_count}) nor Python ({python_subject_count}) matches user ({user_provided_count})")
            info("  → Trusting user specification, using LLM's interpretation")
            # 保持 LLM 的计划，它可能理解了用户的意图
    
    else:
        # LLM 和 Python 不一致，且用户未指定
        warn(f"  ⚠ Conflict: LLM says {llm_subject_count}, Python says {python_subject_count}")
        
        # 信任 LLM（它有完整上下文，包括用户描述）
        if llm_subject_count > python_subject_count:
            info(f"  → Trusting LLM (it has user description and document context)")
            info(f"  → LLM likely identified semantic groups Python missed")
            # 使用 LLM 的计划
            pass
        else:
            # Python 检测到更多主题（罕见情况）
            info(f"  → Python found more subjects than LLM")
            info(f"  → Using Python's analysis (may need manual review)")
            _apply_python_rules_to_plan(plan_yaml, subject_info)
    
    # Add metadata
    plan_yaml['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'model': model,
        'decision_logic': 'llm_first',
        'python_analysis': {
            'method': subject_info.get('method'),
            'success': subject_info.get('success'),
            'subject_count': python_subject_count
        },
        'llm_analysis': {
            'subject_count': llm_subject_count
        },
        'final_decision': {
            'subject_count': plan_yaml.get('subjects', {}).get('count', 0),
            'source': plan_yaml.get('subjects', {}).get('source', 'llm')
        }
    }
    
    # Save plan
    plan_path = staging_dir / BIDS_PLAN
    write_yaml(plan_path, plan_yaml)
    info(f"\n✓ BIDS Plan saved: {plan_path}")
    
    if not plan_path.exists():
        fatal(f"BIDS Plan file was not created at: {plan_path}")
    
    # Step 8: Update participants.tsv with metadata
    info("\nStep 8: Updating participants.tsv with LLM-extracted metadata...")
    
    if 'participant_metadata' in plan_yaml:
        _update_participants_with_metadata(plan_yaml, out_dir)
        info("  ✓ Participant metadata successfully applied")
    else:
        info("  ℹ No participant metadata in plan")
    
    info("\n=== BIDS Plan Generation Complete ===")
    info(f"Plan location: {plan_path}")
    
    final_count = plan_yaml.get('subjects', {}).get('count', 0)
    info(f"Subjects detected: {final_count}")
    
    participant_metadata = plan_yaml.get('participant_metadata')
    if participant_metadata:
        first_subject_metadata = list(participant_metadata.values())[0]
        if first_subject_metadata and isinstance(first_subject_metadata, dict):
            metadata_keys = list(first_subject_metadata.keys())
            info(f"Participant columns: participant_id, {', '.join(metadata_keys)}")
    
    return {
        "status": "ok",
        "warnings": plan_yaml.get("questions", []),
        "questions": [],
        "subject_info": subject_info,
        "llm_subject_count": llm_subject_count,
        "python_subject_count": python_subject_count,
        "metadata_extracted": bool(participant_metadata)
    }


def _apply_python_rules_to_plan(plan_yaml: Dict[str, Any], subject_info: Dict[str, Any]) -> None:
    """Apply Python's subject detection results to the plan."""
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
    
    # Apply Python-generated filename rules
    python_rules = subject_info.get("python_generated_filename_rules", [])
    if python_rules:
        if 'mappings' in plan_yaml and len(plan_yaml['mappings']) > 0:
            plan_yaml['mappings'][0]['filename_rules'] = python_rules
            info(f"  → Applied {len(python_rules)} Python-generated filename rules")


def _enhance_plan_with_python_details(plan_yaml: Dict[str, Any], subject_info: Dict[str, Any]) -> None:
    """Enhance LLM's plan with Python's detailed analysis (when they agree)."""
    # 只在缺失时补充，不覆盖 LLM 的决策
    
    if 'assignment_rules' not in plan_yaml or not plan_yaml['assignment_rules']:
        info("  → Adding Python's assignment rules (LLM didn't provide them)")
        _apply_python_rules_to_plan(plan_yaml, subject_info)
    
    # 补充 Python 生成的 filename rules（如果有）
    python_rules = subject_info.get("python_generated_filename_rules", [])
    if python_rules:
        if 'mappings' in plan_yaml and len(plan_yaml['mappings']) > 0:
            llm_rules = plan_yaml['mappings'][0].get('filename_rules', [])
            
            if not llm_rules:
                plan_yaml['mappings'][0]['filename_rules'] = python_rules
                info(f"  → Added {len(python_rules)} Python-generated filename rules")


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
