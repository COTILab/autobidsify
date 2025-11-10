# planner.py v2
# LLM-First with Python enhancement - uses universal_core analysis

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from datetime import datetime
from utils import write_json, read_json, write_yaml, info, warn, fatal, write_text
from constants import BIDS_PLAN
from llm import llm_bids_plan
from universal_core import extract_subject_ids_from_paths

def _create_participants_json(metadata: Dict, out_dir: Path) -> None:
    """Generate participants.json to describe custom columns"""
    json_path = out_dir / "participants.json"
    
    if json_path.exists():
        info(f"✓ participants.json already exists")
        return
    
    # Collect all columns and values
    all_columns = {}
    
    for subj_meta in metadata.values():
        for col, value in subj_meta.items():
            if col not in all_columns:
                all_columns[col] = {"values": set()}
            all_columns[col]["values"].add(str(value))
    
    # Build data dictionary
    data_dict = {}
    
    for col, info_dict in all_columns.items():
        values = sorted(list(info_dict["values"]))
        
        if col == "sex":
            data_dict[col] = {
                "Description": "Biological sex of the participant",
                "Levels": {"M": "Male", "F": "Female", "n/a": "Not available"}
            }
        elif col == "age":
            data_dict[col] = {"Description": "Age of the participant", "Units": "years"}
        elif col == "group":
            levels = {v: v.capitalize() for v in values if v != "n/a"}
            levels["n/a"] = "Not available"
            data_dict[col] = {"Description": "Experimental group", "Levels": levels}
        elif col == "site":
            levels = {v: f"Data collection site: {v}" for v in values if v != "n/a"}
            levels["n/a"] = "Not available"
            data_dict[col] = {"Description": "Data collection site", "Levels": levels}
        else:
            if len(values) <= 10:
                levels = {v: v for v in values if v != "n/a"}
                levels["n/a"] = "Not available"
                data_dict[col] = {"Description": f"Participant {col}", "Levels": levels}
            else:
                data_dict[col] = {"Description": f"Participant {col}"}
    
    write_json(json_path, data_dict)
    info(f"✓ Created participants.json")

def build_bids_plan(model: str, planning_inputs: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Build BIDS plan - v2 simplified, relies on evidence analysis.
    
    Key changes:
    1. Uses structure_analysis from evidence bundle
    2. Builds compressed payload for LLM
    3. Python enhances LLM's plan with full file analysis
    4. Subject detection uses evidence's result directly
    """
    info("Generating BIDS Plan (v2: LLM strategy + Python execution)...")
    
    planning_inputs["timestamp"] = datetime.now().isoformat()
    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    
    # Extract key information
    all_files = evidence_bundle.get("all_files", [])
    structure_analysis = evidence_bundle.get("structure_analysis", {})
    user_text = evidence_bundle.get("user_hints", {}).get("user_text", "")
    user_n_subjects = evidence_bundle.get("user_hints", {}).get("n_subjects")
    
    subject_detection = structure_analysis.get("subject_detection", {})
    tree_summary = structure_analysis.get("tree_summary_for_llm", {})
    
    info(f"Dataset: {len(all_files)} files")
    info(f"Structure: {structure_analysis.get('directory_structure', {}).get('structure_template', 'unknown')}")
    
    # === Build Optimized LLM Payload ===
    # Strategy: Send compressed structure, not full file list
    
    llm_payload = {
        "structure_summary": tree_summary,  # Compressed! Only 50 subjects
        "total_files": len(all_files),
        "total_subjects": subject_detection.get("best_candidate", {}).get("count", user_n_subjects),
        "subject_pattern": subject_detection.get("best_candidate", {}).get("pattern_display", "unknown"),
        "directory_template": structure_analysis.get("directory_structure", {}).get("structure_template"),
        "sample_files": all_files[:50] + all_files[-50:],  # First and last 50
        "user_context": {
            "description": user_text,
            "n_subjects_hint": user_n_subjects,
            "modality_hint": evidence_bundle.get("user_hints", {}).get("modality_hint")
        },
        "documents": evidence_bundle.get("documents", []),
        "file_counts": evidence_bundle.get("counts_by_ext", {})
    }
    
    payload_json = json.dumps(llm_payload, ensure_ascii=False)
    payload_size = len(payload_json)
    estimated_tokens = payload_size // 4
    
    info(f"Payload size: {payload_size:,} characters (~{estimated_tokens:,} tokens)")
    
    if estimated_tokens > 25000:
        warn(f"⚠ Payload approaching token limit, results may be slower")
    
    # === Call LLM ===
    try:
        response_text = llm_bids_plan(model, payload_json)
        
        # Clean response
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        plan = yaml.safe_load(text)
        
    except yaml.YAMLError as e:
        fatal(f"YAML parsing failed: {e}")
        return {"status": "error"}
    except Exception as e:
        fatal(f"BIDS plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error"}
    
    if not plan.get("subjects"):
        warn("WARNING: No subjects in plan")
        return {"status": "error"}
    
    info(f"✓ LLM generated strategy for {len(plan.get('subjects', {}).get('labels', []))} subjects")
    
    # === CRITICAL: Python Enhances Plan with Full Analysis ===
    # Don't rely on LLM's subject detection - use evidence's result!
    
    subject_info = _build_subject_records_from_evidence(
        all_files=all_files,
        structure_analysis=structure_analysis,
        llm_plan=plan
    )
    
    info(f"✓ Python enhanced: {subject_info['subject_count']} subjects confirmed")
    
    # === Generate participants.tsv ===
    participant_metadata = plan.get("participant_metadata", {})
    
    if subject_info["subject_count"] > 0:
        if participant_metadata:
            _create_participants_with_metadata(subject_info, participant_metadata, out_dir)
        else:
            _create_simple_participants_tsv(subject_info, out_dir)
    else:
        warn("⚠ No subjects detected, skipping participants.tsv")
    
    # === Save Outputs ===
    from utils import sha256_full
    plan["fingerprints"] = {
        "evidence_bundle_sha": sha256_full(json.dumps(evidence_bundle.get("counts_by_ext", {}), sort_keys=True)),
        "timestamp": planning_inputs["timestamp"]
    }
    
    write_json(Path(out_dir) / "_staging" / "subject_analysis.json", subject_info)
    write_yaml(Path(out_dir) / "_staging" / "BIDSPlan.yaml", plan)
    
    info(f"✓ BIDS Plan saved")
    
    return {"status": "ok", "plan": plan}

def _build_subject_records_from_evidence(all_files: List[str], structure_analysis: Dict, 
                                         llm_plan: Dict) -> Dict[str, Any]:
    """
    Build subject records using evidence's analysis (NOT LLM's!)
    
    This is the key fix: we trust Python's analysis over LLM's extraction.
    
    Args:
        all_files: Complete file list
        structure_analysis: From evidence bundle
        llm_plan: LLM's plan (for metadata only)
    
    Returns:
        Subject records with full information
    """
    subject_detection = structure_analysis.get("subject_detection", {})
    best_candidate = subject_detection.get("best_candidate")
    
    if not best_candidate:
        warn("⚠ No subject pattern detected in evidence analysis")
        # Try to use LLM's labels as fallback
        labels = llm_plan.get("subjects", {}).get("labels", [])
        if labels:
            return {
                "extraction_method": "llm_labels_fallback",
                "subject_records": [{"numeric_id": str(l), "original_id": str(l)} for l in labels],
                "subject_count": len(labels)
            }
        else:
            return {
                "extraction_method": "failed",
                "subject_records": [],
                "subject_count": 0
            }
    
    # Use evidence's best candidate to extract subjects
    extraction_regex = best_candidate["extraction_regex"]
    subject_group = best_candidate["subject_group"]
    site_group = best_candidate.get("site_group")
    
    info(f"  Using pattern: {best_candidate['pattern_display']}")
    info(f"  Extraction regex: {extraction_regex}")
    
    # CRITICAL: Use the helper function that fixes regex escapes!
    subject_records_raw = extract_subject_ids_from_paths(
        all_files,
        extraction_regex,
        subject_group,
        site_group
    )
    
    # Convert to standard format
    subject_records = []
    for rec in subject_records_raw:
        subject_records.append({
            "numeric_id": rec["subject_id"],
            "original_id": rec["original_dirname"],
            "site": rec.get("site"),
            "pattern_name": best_candidate["pattern_name"],
            "metadata": {"site": rec.get("site")} if rec.get("site") else {}
        })
    
    return {
        "extraction_method": best_candidate["pattern_name"],
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "pattern_used": best_candidate["pattern_display"],
        "confidence": subject_detection.get("confidence", "unknown")
    }

def _create_participants_with_metadata(subject_info: Dict, metadata: Dict, out_dir: Path) -> None:
    """Create participants.tsv with LLM-provided metadata"""
    parts_path = out_dir / "participants.tsv"
    
    if parts_path.exists():
        info(f"✓ participants.tsv already exists")
        return
    
    subject_records = subject_info["subject_records"]
    
    if len(subject_records) == 0:
        warn("⚠ No subject records to write")
        return
    
    # Collect columns
    all_columns = set()
    for subj_meta in metadata.values():
        all_columns.update(subj_meta.keys())
    
    columns = ["participant_id"] + sorted(list(all_columns))
    
    # Build TSV
    lines = ["\t".join(columns) + "\n"]
    
    for rec in subject_records:
        bids_id = f"sub-{rec['numeric_id']}"
        row = [bids_id]
        
        subj_meta = metadata.get(rec['numeric_id'], {})
        
        for col in columns[1:]:
            value = subj_meta.get(col, "n/a")
            row.append(str(value))
        
        lines.append("\t".join(row) + "\n")
    
    content = "".join(lines)
    write_text(parts_path, content)
    
    info(f"✓ Created participants.tsv ({len(subject_records)} subjects)")
    info(f"  Columns: {', '.join(columns)}")
    
    # Auto-generate participants.json
    _create_participants_json(metadata, out_dir)

def _create_simple_participants_tsv(subject_info: Dict, out_dir: Path) -> None:
    """Create simple participants.tsv (participant_id only)"""
    parts_path = out_dir / "participants.tsv"
    
    if parts_path.exists():
        info(f"✓ participants.tsv already exists")
        return
    
    subject_records = subject_info.get("subject_records", [])
    
    if len(subject_records) == 0:
        warn("⚠ No subject records")
        return
    
    lines = ["participant_id\n"]
    for rec in subject_records:
        lines.append(f"sub-{rec['numeric_id']}\n")
    
    write_text(parts_path, "".join(lines))
    info(f"✓ Created participants.tsv ({len(subject_records)} subjects)")
