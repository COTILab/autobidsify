# executor.py v3
# Execute LLM's plan - CRITICAL: Use LLM's decisions, don't re-detect!

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import re
from utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn
from converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available

def _build_ascii_tree(root: Path, max_depth: int = 3) -> str:
    """Build ASCII tree visualization"""
    lines = [root.name + "/"]
    
    def walk(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            entries = sorted([p for p in directory.iterdir()], 
                           key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        max_items = 15 if depth == 0 else 8
        entries = entries[:max_items]
        
        for i, path in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            line = prefix + connector + path.name
            if path.is_dir():
                line += "/"
            lines.append(line)
            
            if path.is_dir() and depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                walk(path, prefix + extension, depth + 1)
    
    walk(root)
    return "\n".join(lines)


def _match_file_to_subject(filename: str, assignment_rules: List[Dict]) -> Optional[str]:
    """
    Match file to subject based on assignment_rules from BIDS Plan.
    
    Args:
        filename: Filename (not full path)
        assignment_rules: List of rules from plan
    
    Returns:
        Subject ID if matched, None otherwise
    """
    for rule in assignment_rules:
        subject_id = rule.get('subject')
        prefix = rule.get('prefix')
        original = rule.get('original')
        
        # Method 1: Prefix matching (for flat structures)
        if prefix and filename.startswith(prefix):
            return subject_id
        
        # Method 2: Original ID matching (for hierarchical)
        if original and original in filename:
            return subject_id
    
    return None


def _extract_bids_entities_from_filename(filename: str, filename_rules: List[Dict]) -> Optional[Dict[str, str]]:
    """
    Extract BIDS entities from filename using LLM's filename_rules.
    
    Args:
        filename: Original filename
        filename_rules: Rules from mapping in BIDS Plan
    
    Returns:
        {
            "bids_template": "sub-1_acq-cthip_T1w.nii.gz",
            "entities": {"subject": "1", "acquisition": "cthip"}
        }
    """
    for rule in filename_rules:
        match_pattern = rule.get('match_pattern', '')
        
        # Remove YAML double backslashes
        pattern_fixed = match_pattern.replace(r'\\', '\\')
        
        try:
            if re.search(pattern_fixed, filename, re.IGNORECASE):
                return {
                    "bids_template": rule.get('bids_template'),
                    "entities": rule.get('extract_entities', {})
                }
        except re.error:
            # If regex fails, try simple string matching
            keywords = re.findall(r'[A-Za-z]+', match_pattern)
            if all(kw.lower() in filename.lower() for kw in keywords if len(kw) > 2):
                return {
                    "bids_template": rule.get('bids_template'),
                    "entities": rule.get('extract_entities', {})
                }
    
    return None


def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute BIDS plan v3 - Use LLM's decisions from plan.
    
    CRITICAL CHANGE: Don't re-detect subjects, use LLM's assignment_rules!
    
    Args:
        input_root: Input data directory
        output_dir: Output directory
        plan: BIDS plan from planner (contains LLM's decisions)
        aux_inputs: Auxiliary inputs
    
    Returns:
        Execution summary
    """
    info("Executing BIDS Plan (v3: using LLM's subject grouping)...")
    
    bids_root = Path(output_dir) / "bids_compatible"
    derivatives_dir = bids_root / "derivatives"
    ensure_dir(bids_root)
    ensure_dir(derivatives_dir)
    
    logs = []
    successes = 0
    failures = 0
    
    # === Step 1: Copy trio files ===
    info("\n[1/3] Organizing trio files...")
    trio_files = ["dataset_description.json", "README.md", "participants.tsv", "participants.json"]
    
    for trio_file in trio_files:
        src = output_dir / trio_file
        if not src.exists():
            src = input_root / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")
    
    # === Step 2: Process data files using LLM's plan ===
    info("\n[2/3] Processing data files using LLM's grouping rules...")
    
    # Get all files
    all_files_paths = list_all_files(input_root)
    all_files_str = [str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    
    info(f"Total files in input: {len(all_files_str)}")
    
    # CRITICAL: Read LLM's decisions from plan
    subject_grouping = plan.get("subject_grouping", {})
    assignment_rules = plan.get("assignment_rules", [])
    mappings = plan.get("mappings", [])
    
    grouping_method = subject_grouping.get('method', 'unknown')
    info(f"  Grouping method from plan: {grouping_method}")
    info(f"  Assignment rules: {len(assignment_rules)} rules")
    
    # DEBUG: Show assignment rules
    if assignment_rules:
        info(f"  Subject assignments:")
        for rule in assignment_rules[:5]:
            prefix = rule.get('prefix', rule.get('original', 'N/A'))
            subject = rule.get('subject')
            info(f"    '{prefix}' -> sub-{subject}")
        if len(assignment_rules) > 5:
            info(f"    ... and {len(assignment_rules) - 5} more")
    else:
        warn("  ⚠ WARNING: No assignment_rules in plan!")
        warn("  This will cause all files to be assigned to sub-01")
    
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        filename_rules = mapping.get("filename_rules", [])
        
        info(f"\n  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        info(f"    Match patterns: {patterns}")
        info(f"    Filename rules: {len(filename_rules)} rules")
        
        # Simple pattern matching
        matched_files = []
        for filepath_str in all_files_str:
            filename = filepath_str.split('/')[-1]
            
            # Check if matches any pattern
            for pattern in patterns:
                if pattern == "**/*.dcm" and filename.endswith('.dcm'):
                    matched_files.append(filepath_str)
                    break
                elif pattern == "**/*.nii.gz" and filename.endswith('.nii.gz'):
                    matched_files.append(filepath_str)
                    break
        
        if not matched_files:
            warn(f"    ⚠ No files matched patterns")
            continue
        
        info(f"    ✓ Matched: {len(matched_files)} files")
        
        # Group by subject + entity (using LLM's rules)
        file_groups = {}
        
        for filepath_str in matched_files:
            filename = filepath_str.split('/')[-1]
            
            # Match to subject
            subject_id = _match_file_to_subject(filename, assignment_rules)
            if not subject_id:
                subject_id = "unknown"
            
            # Extract BIDS entities from filename
            entity_info = _extract_bids_entities_from_filename(filename, filename_rules)
            
            if entity_info:
                bids_template = entity_info['bids_template']
                group_key = f"{subject_id}_{bids_template}"
            else:
                # Fallback: generic grouping
                group_key = f"{subject_id}_generic"
                bids_template = f"sub-{subject_id}_unknown.nii.gz"
            
            if group_key not in file_groups:
                file_groups[group_key] = {
                    'subject_id': subject_id,
                    'bids_template': bids_template,
                    'files': [],
                    'entity_info': entity_info
                }
            
            file_groups[group_key]['files'].append(filepath_str)
        
        info(f"    Grouped into: {len(file_groups)} scan groups")
        
        # === Process each group ===
        if format_ready and convert_to == "none":
            # Files already in correct format
            info(f"    Organizing {len(file_groups)} file groups...")
            
            for group_key, group_data in file_groups.items():
                try:
                    # Take first file from group (or could copy all)
                    filepath_str = group_data['files'][0]
                    filepath = path_str_to_path.get(filepath_str)
                    
                    if not filepath:
                        continue
                    
                    subject_id = group_data['subject_id']
                    bids_filename = group_data['bids_template']
                    
                    # Determine subdirectory (anat for MRI/CT)
                    subdir = "anat"
                    
                    dst = bids_root / f"sub-{subject_id}" / subdir / bids_filename
                    ensure_dir(dst.parent)
                    copy_file(filepath, dst)
                    
                    logs.append({
                        "source": filepath_str,
                        "destination": f"sub-{subject_id}/{subdir}/{bids_filename}",
                        "action": "organize",
                        "status": "success"
                    })
                    successes += 1
                    
                except Exception as e:
                    warn(f"      Failed: {e}")
                    failures += 1
        
        elif not format_ready and convert_to == "dicom_to_nifti":
            # DICOM conversion
            if not check_dcm2niix_available():
                warn(f"    dcm2niix not available, skipping")
                failures += len(matched_files)
                continue
            
            temp_base = Path(output_dir) / "_staging" / "dicom_temp"
            
            info(f"    Converting {len(file_groups)} DICOM volume groups...")
            
            progress_count = 0
            progress_interval = max(1, len(file_groups) // 10)
            
            for group_key, group_data in file_groups.items():
                try:
                    file_paths_str = group_data['files']
                    file_paths = [path_str_to_path[s] for s in file_paths_str if s in path_str_to_path]
                    
                    subject_id = group_data['subject_id']
                    bids_filename = group_data['bids_template']
                    
                    subdir = "anat"
                    output_path = bids_root / f"sub-{subject_id}" / subdir / bids_filename
                    
                    temp_dir = temp_base / group_key
                    ensure_dir(temp_dir)
                    
                    result = run_dcm2niix_batch(file_paths, output_path, temp_dir, quiet=True)
                    
                    if result:
                        logs.append({
                            "source": f"{len(file_paths)} DICOM files",
                            "destination": f"sub-{subject_id}/{subdir}/{bids_filename}",
                            "action": "dicom_to_nifti",
                            "status": "success"
                        })
                        successes += 1
                    else:
                        failures += 1
                    
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    progress_count += 1
                    if progress_count % progress_interval == 0 or progress_count == len(file_groups):
                        percent = (progress_count / len(file_groups)) * 100
                        info(f"      Progress: {progress_count}/{len(file_groups)} ({percent:.0f}%) - Latest: {bids_filename}")
                        
                except Exception as e:
                    warn(f"      Conversion failed: {e}")
                    failures += 1
            
            if temp_base.exists():
                shutil.rmtree(temp_base, ignore_errors=True)
    
    # === Step 3: Finalize ===
    info("\n[3/3] Finalizing...")
    
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    
    manifest_files = [str(p.relative_to(bids_root)).replace("\\", "/") 
                      for p in bids_root.rglob("*") if p.is_file() and not p.name.startswith('.')]
    
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files": sorted(manifest_files),
        "tree": _build_ascii_tree(bids_root)
    })
    
    # === Verification ===
    subject_dirs = list(bids_root.glob("sub-*"))
    
    info(f"\n✓ BIDS Dataset Created")
    info(f"Location: {bids_root}")
    info(f"Files processed: {successes}")
    info(f"Failed: {failures}")
    
    if len(subject_dirs) > 0:
        info(f"\nCreated {len(subject_dirs)} subject directories:")
        for subj_dir in sorted(subject_dirs)[:10]:
            nii_count = len(list(subj_dir.rglob("*.nii.gz")))
            info(f"  {subj_dir.name}: {nii_count} NIfTI file(s)")
        if len(subject_dirs) > 10:
            info(f"  ... and {len(subject_dirs) - 10} more subjects")
    else:
        warn("⚠ WARNING: No subject directories created!")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root),
        "subject_count": len(subject_dirs)
    }
