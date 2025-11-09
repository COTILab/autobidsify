# planner.py
# LLM-First architecture: LLM makes ALL decisions, Python only executes

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import re
from datetime import datetime
from utils import write_json, read_json, write_yaml, info, warn, fatal, write_text
from constants import HEADERS_DRAFT, HEADERS_NORMALIZED, VOXEL_DRAFT, VOXEL_FINAL_PLAN, BIDS_PLAN, SEVERITY_BLOCK
from llm import llm_nirs_draft, llm_nirs_normalize, llm_mri_voxel_draft, llm_mri_voxel_final, llm_bids_plan

def fix_yaml_regex_escapes(yaml_text: str) -> str:
    """Fix regex escape sequences for YAML compatibility."""
    lines = yaml_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        if ('pattern:' in line or 'match_pattern:' in line or 'extraction_pattern:' in line) and ('"' in line or "'" in line):
            if '"' in line:
                parts = line.split('"')
                if len(parts) >= 3:
                    pattern_value = parts[1]
                    fixed_pattern = pattern_value.replace('\\', '\\\\')
                    line = parts[0] + '"' + fixed_pattern + '"' + '"'.join(parts[2:])
            elif "'" in line:
                parts = line.split("'")
                if len(parts) >= 3:
                    pattern_value = parts[1]
                    fixed_pattern = pattern_value.replace('\\', '\\\\')
                    line = parts[0] + "'" + fixed_pattern + "'" + "'".join(parts[2:])
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def _parse_llm_json_response(response_text: str, step_name: str) -> Optional[Dict[str, Any]]:
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

def _extract_python_observations(all_files: List[str]) -> Dict[str, Any]:
    """
    Python observes patterns WITHOUT making decisions.
    Just notes what it sees - LLM will interpret.
    """
    observations = {
        "unique_prefixes": set(),
        "unique_keywords": set(),
        "directory_structure": "flat",
        "patterns_noticed": []
    }
    
    # Collect prefixes (first few chars before special chars)
    for filepath in all_files[:min(100, len(all_files))]:
        filename = filepath.split('/')[-1]
        
        # Extract prefix (alphanumeric before special char)
        match = re.match(r'([A-Za-z]+)', filename)
        if match:
            prefix = match.group(1)[:5]  # First 5 chars max
            observations["unique_prefixes"].add(prefix)
        
        # Extract keywords (capitalized words)
        words = re.findall(r'[A-Z][a-z]+', filename)
        observations["unique_keywords"].update(words[:3])
    
    # Check directory structure
    has_dirs = any('/' in f for f in all_files)
    if has_dirs:
        max_depth = max(len(f.split('/')) for f in all_files)
        observations["directory_structure"] = f"hierarchical (max depth: {max_depth})"
    
    # Convert sets to lists for JSON serialization
    observations["unique_prefixes"] = sorted(list(observations["unique_prefixes"]))[:20]
    observations["unique_keywords"] = sorted(list(observations["unique_keywords"]))[:20]
    
    return observations

def build_bids_plan(model: str, planning_inputs: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    info("Generating BIDS Plan (LLM-First architecture)...")
    
    planning_inputs["timestamp"] = datetime.now().isoformat()
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files = evidence_bundle.get("all_files", [])
    user_text = evidence_bundle.get("user_hints", {}).get("user_text", "")
    user_n_subjects = evidence_bundle.get("user_hints", {}).get("n_subjects")
    
    # Python observations (NOT decisions!)
    python_obs = _extract_python_observations(all_files)
    
    info(f"Dataset scan: {len(all_files)} files")
    if python_obs['unique_prefixes']:
        info(f"  Python noticed prefixes: {python_obs['unique_prefixes'][:5]}")
    if python_obs['unique_keywords']:
        info(f"  Python noticed keywords: {python_obs['unique_keywords'][:5]}")
    
    # Determine how many files to send based on dataset size
    if len(all_files) <= 100:
        file_list_for_llm = all_files
        info(f"Small dataset: sending all {len(all_files)} files to LLM")
    elif len(all_files) <= 3000:
        file_list_for_llm = all_files
        info(f"Medium dataset: sending all {len(all_files)} files to LLM for complete analysis")
    else:
        # Strategic sampling for very large datasets
        sample_files = []
        sample_files.extend(all_files[:100])
        step = max(1, len(all_files) // 200)
        for i in range(100, len(all_files) - 100, step):
            sample_files.append(all_files[i])
            if len(sample_files) >= 300:
                break
        sample_files.extend(all_files[-100:])
        file_list_for_llm = sample_files
        info(f"Large dataset: sending {len(file_list_for_llm)} representative files to LLM")
    
    # Build LLM-friendly payload
    llm_payload = {
        "file_structure": {
            "all_files": file_list_for_llm,
            "total_files": len(all_files),
            "counts_by_ext": evidence_bundle.get("counts_by_ext", {})
        },
        "user_context": {
            "description": user_text if user_text else "No user description provided",
            "n_subjects_hint": user_n_subjects,
            "modality_hint": evidence_bundle.get("user_hints", {}).get("modality_hint")
        },
        "documents": evidence_bundle.get("documents", []),
        "python_observations": python_obs
    }
    
    payload_json = json.dumps(llm_payload, ensure_ascii=False)
    payload_size = len(payload_json)
    info(f"Payload size: {payload_size:,} characters")
    
    if user_text:
        info(f"✓ User description: {len(user_text)} chars")
    
    # Call LLM (IT makes all decisions!)
    try:
        response_text = llm_bids_plan(model, payload_json)
        
        text = response_text.strip()
        if text.startswith("```yaml") or text.startswith("```yml") or text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Fix regex escapes for YAML
        text = fix_yaml_regex_escapes(text)
        
        plan = yaml.safe_load(text)
        
    except yaml.YAMLError as e:
        fatal(f"YAML parsing failed: {e}")
        fatal("LLM returned invalid YAML format")
        return {"status": "error"}
    except Exception as e:
        fatal(f"BIDS plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error"}
    
    # Validate LLM output
    if not plan.get("subjects"):
        warn("WARNING: No subjects defined in plan")
        return {"status": "error", "message": "Invalid plan structure"}
    
    info(f"✓ LLM generated plan for {len(plan.get('subjects', {}).get('labels', []))} subjects")
    
    # Execute LLM's grouping strategy
    subject_info = _execute_llm_grouping_strategy(all_files, plan, evidence_bundle)
    
    # Create participants.tsv based on LLM's metadata
    participant_metadata = plan.get("participant_metadata", {})
    if participant_metadata:
        _create_participants_with_llm_metadata(subject_info, participant_metadata, out_dir)
    elif subject_info.get("subject_records"):
        _create_simple_participants_tsv(subject_info, out_dir)
    
    # Add fingerprints
    from utils import sha256_full
    plan["fingerprints"] = {
        "evidence_bundle_sha": sha256_full(json.dumps(evidence_bundle.get("counts_by_ext", {}), sort_keys=True)),
        "timestamp": planning_inputs["timestamp"]
    }
    
    # Save outputs
    write_json(Path(out_dir) / "_staging" / "subject_analysis.json", subject_info)
    write_yaml(Path(out_dir) / "_staging" / "BIDSPlan.yaml", plan)
    
    if subject_info.get("subject_count", 0) > 0:
        info(f"✓ Analysis: {subject_info['subject_count']} subjects")
    info(f"✓ BIDS Plan saved")
    
    return {"status": "ok", "plan": plan}

def _execute_llm_grouping_strategy(all_files: List[str], plan: Dict[str, Any], evidence: Dict) -> Dict[str, Any]:
    """
    Execute LLM's subject grouping strategy.
    Python just follows LLM's instructions.
    """
    grouping = plan.get("subject_grouping", {})
    method = grouping.get("method")
    
    subject_records = []
    
    if method == "prefix_based":
        # LLM provided prefix-to-subject mapping
        rules = grouping.get("rules", [])
        
        for rule in rules:
            subject_id = rule.get("maps_to_subject")
            prefix = rule.get("prefix")
            metadata = rule.get("metadata", {})
            
            subject_records.append({
                "numeric_id": subject_id,
                "original_id": prefix,
                "prefix": prefix,
                "pattern_name": "llm_prefix",
                "metadata": metadata
            })
        
        info(f"Applied prefix_based grouping: {len(subject_records)} subjects")
    
    elif method == "directory_based":
        # LLM provided extraction pattern
        extraction_pattern = grouping.get("extraction_pattern")
        subject_group = grouping.get("subject_from_group", 1)
        site_group = grouping.get("site_from_group")
        
        seen_ids = set()
        for filepath in all_files:
            parts = filepath.split('/')
            for part in parts[:3]:
                match = re.search(extraction_pattern, part)
                if match and len(match.groups()) >= subject_group:
                    subject_id = match.group(subject_group)
                    if subject_id not in seen_ids:
                        seen_ids.add(subject_id)
                        
                        site = None
                        if site_group and len(match.groups()) >= site_group:
                            site = match.group(site_group)
                        
                        subject_records.append({
                            "numeric_id": subject_id,
                            "original_id": match.group(0),
                            "site": site,
                            "pattern_name": "llm_directory"
                        })
                    break
        
        info(f"Applied directory_based grouping: {len(subject_records)} subjects")
    
    elif method == "filename_pattern":
        # LLM provided filename extraction regex
        extraction_regex = grouping.get("extraction_regex")
        subject_group = grouping.get("subject_from_group", 1)
        
        seen_ids = set()
        for filepath in all_files:
            match = re.search(extraction_regex, filepath)
            if match and len(match.groups()) >= subject_group:
                subject_id = match.group(subject_group)
                if subject_id not in seen_ids:
                    seen_ids.add(subject_id)
                    subject_records.append({
                        "numeric_id": subject_id,
                        "original_id": subject_id,
                        "pattern_name": "llm_filename"
                    })
        
        info(f"Applied filename_pattern grouping: {len(subject_records)} subjects")
    
    elif method == "single_subject":
        # All files belong to one subject
        subject_records.append({
            "numeric_id": "01",
            "original_id": "01",
            "pattern_name": "single"
        })
        info("Single subject dataset")
    
    else:
        # Use whatever LLM provided in subjects.labels
        labels = plan.get("subjects", {}).get("labels", [])
        for label in labels:
            subject_records.append({
                "numeric_id": str(label),
                "original_id": str(label),
                "pattern_name": "llm_provided"
            })
        info(f"Using LLM-provided subject list: {len(subject_records)} subjects")
    
    return {
        "extraction_method": method or "llm_directed",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "grouping_strategy": grouping
    }

def _create_participants_with_llm_metadata(subject_info: Dict, metadata: Dict, out_dir: Path) -> None:
    """
    Create participants.tsv using LLM-provided participant metadata.
    
    Args:
        subject_info: Subject records from grouping
        metadata: Dict[subject_id, Dict[column, value]]
            Example: {"1": {"sex": "M", "group": "cadaver"}, "2": {"sex": "F", "group": "cadaver"}}
    """
    parts_path = out_dir / "participants.tsv"
    
    if parts_path.exists():
        info(f"✓ participants.tsv already exists, preserving")
        return
    
    subject_records = subject_info["subject_records"]
    
    # Collect all column names from metadata
    all_columns = set()
    for subj_meta in metadata.values():
        all_columns.update(subj_meta.keys())
    
    columns = ["participant_id"] + sorted(list(all_columns))
    
    # Build TSV
    lines = ["\t".join(columns) + "\n"]
    
    for rec in subject_records:
        bids_id = f"sub-{rec['numeric_id']}"
        row = [bids_id]
        
        subj_id = rec['numeric_id']
        subj_meta = metadata.get(subj_id, {})
        
        for col in columns[1:]:
            value = subj_meta.get(col, "n/a")
            row.append(str(value))
        
        lines.append("\t".join(row) + "\n")
    
    content = "".join(lines)
    write_text(parts_path, content)
    
    info(f"✓ Created participants.tsv ({len(subject_records)} subjects)")
    info(f"  Columns: {', '.join(columns)}")

def _create_simple_participants_tsv(subject_info: Dict, out_dir: Path) -> None:
    """Create simple participants.tsv with only participant_id."""
    parts_path = out_dir / "participants.tsv"
    
    if parts_path.exists():
        info(f"✓ participants.tsv already exists, preserving")
        return
    
    subject_records = subject_info.get("subject_records", [])
    
    lines = ["participant_id\n"]
    for rec in subject_records:
        bids_id = f"sub-{rec['numeric_id']}"
        lines.append(f"{bids_id}\n")
    
    content = "".join(lines)
    write_text(parts_path, content)
    info(f"✓ Created participants.tsv ({len(subject_records)} subjects)")

def nirs_tables_to_normalized(model: str, evidence: Dict[str, Any], user_edited_draft: Optional[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    if user_edited_draft is None:
        info("Generating NIRS headers draft...")
        payload = json.dumps({"samples": evidence.get("samples", []), "documents": evidence.get("documents", []), "user_text": evidence.get("user_hints", {}).get("user_text", "")}, ensure_ascii=False)
        try:
            response_text = llm_nirs_draft(model, payload)
            draft_obj = _parse_llm_json_response(response_text, "NIRS_draft")
            if draft_obj is None:
                return {"status": "error", "questions": []}
        except Exception as e:
            fatal(f"NIRS draft failed: {e}")
            return {"status": "error", "questions": []}
        write_json(Path(out_dir) / HEADERS_DRAFT, draft_obj)
        info(f"✓ Draft saved")
        return {"status": "draft_written", "questions": draft_obj.get("questions", [])}
    else:
        info("Normalizing NIRS headers...")
        payload = json.dumps(user_edited_draft, ensure_ascii=False)
        try:
            response_text = llm_nirs_normalize(model, payload)
            normalized_obj = _parse_llm_json_response(response_text, "NIRS_normalize")
            if normalized_obj is None:
                return {"status": "error", "questions": []}
        except Exception as e:
            fatal(f"NIRS normalize failed: {e}")
            return {"status": "error", "questions": []}
        write_json(Path(out_dir) / HEADERS_NORMALIZED, normalized_obj)
        info(f"✓ Normalized saved")
        questions = normalized_obj.get("questions", [])
        has_blocks = any(q.get("severity") == SEVERITY_BLOCK for q in questions)
        return {"status": "blocked" if has_blocks else "normalized", "normalized": normalized_obj, "questions": questions}

def mri_arrays_to_final_plan(model: str, arrays_meta: Dict[str, Any], user_edited_draft: Optional[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    if user_edited_draft is None:
        info("Generating MRI voxel draft...")
        try:
            response_text = llm_mri_voxel_draft(model, json.dumps(arrays_meta))
            draft_obj = _parse_llm_json_response(response_text, "MRI_draft")
            if draft_obj is None:
                return {"status": "error"}
        except Exception as e:
            fatal(f"MRI draft failed: {e}")
            return {"status": "error"}
        write_json(Path(out_dir) / VOXEL_DRAFT, draft_obj)
        return {"status": "draft_written", "questions": draft_obj.get("questions", [])}
    else:
        info("Generating MRI final plan...")
        try:
            response_text = llm_mri_voxel_final(model, json.dumps(user_edited_draft))
            final_obj = _parse_llm_json_response(response_text, "MRI_final")
            if final_obj is None:
                return {"status": "error"}
        except Exception as e:
            fatal(f"MRI final failed: {e}")
            return {"status": "error"}
        write_json(Path(out_dir) / VOXEL_FINAL_PLAN, final_obj)
        return {"status": "final_ready", "final_plan": final_obj}
