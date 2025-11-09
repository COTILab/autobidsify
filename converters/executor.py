# executor.py
# Execute LLM's decisions with automatic DICOM conversion

from pathlib import Path
from typing import Dict, Any, List
import json
import re
import shutil
from fnmatch import fnmatch
from utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn
from converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available

def _match_files_with_patterns(all_files: List[Path], patterns: List[str], input_root: Path) -> List[Path]:
    """Match files against glob patterns."""
    matched = []
    
    for file_path in all_files:
        try:
            rel_path = file_path.relative_to(input_root)
            rel_path_str = str(rel_path).replace("\\", "/")
        except ValueError:
            continue
        
        for pattern in patterns:
            if "**" in pattern:
                pattern_parts = pattern.split("**")
                if len(pattern_parts) == 2:
                    prefix = pattern_parts[0].strip("/")
                    suffix = pattern_parts[1].strip("/")
                    
                    if (not prefix or rel_path_str.startswith(prefix) or fnmatch(rel_path_str, prefix + "*")):
                        if fnmatch(file_path.name, suffix) or fnmatch(rel_path_str, "*" + suffix):
                            matched.append(file_path)
                            break
            else:
                if fnmatch(rel_path_str, pattern) or fnmatch(file_path.name, pattern):
                    matched.append(file_path)
                    break
    
    return matched

def _apply_llm_filename_rule(src: Path, rule: Dict) -> tuple:
    """
    Apply a single LLM-generated filename rule.
    
    Returns:
        (bids_filename, subject_id, entities_dict) or (None, None, None) if no match
    """
    match_pattern = rule.get("match_pattern")
    bids_template = rule.get("bids_template")
    extract_entities = rule.get("extract_entities", {})
    
    if not match_pattern or not bids_template:
        return None, None, None
    
    src_name = src.name
    
    try:
        match = re.search(match_pattern, src_name, re.IGNORECASE)
        if not match:
            return None, None, None
        
        # Build entity replacements
        entities = extract_entities.copy()
        
        # Extract capture groups if any
        if match.groups():
            for i, group_val in enumerate(match.groups(), 1):
                if group_val and not entities.get(f"group{i}"):
                    entities[f"group{i}"] = group_val
        
        # Replace placeholders in template
        result = bids_template
        for key, value in entities.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        # Clean up empty placeholders
        result = re.sub(r'\{[^}]+\}', '', result)
        result = re.sub(r'_+', '_', result)
        result = re.sub(r'_\\.', '.', result)
        
        subject_id = entities.get("subject")
        
        return result, subject_id, entities
        
    except re.error as e:
        warn(f"Invalid regex in rule: {match_pattern}: {e}")
        return None, None, None

def _group_dicom_files_by_rule(matched_files: List[Path], filename_rules: List[Dict]) -> Dict[str, Dict]:
    """
    Group DICOM files according to LLM's filename rules.
    
    Files that match the same rule and belong to same subject/acquisition
    should be grouped together for conversion.
    
    Returns:
        Dict[group_key, {"files": [...], "bids_name": "...", "subject": "...", "entities": {...}}]
    """
    groups = {}
    
    for dcm_file in matched_files:
        # Try to match against rules
        matched_rule = False
        
        for rule in filename_rules:
            bids_name, subject_id, entities = _apply_llm_filename_rule(dcm_file, rule)
            
            if bids_name and subject_id:
                # Create group key from subject + bids_name (without extension)
                # This groups all slices of same acquisition together
                base_name = bids_name.replace('.nii.gz', '').replace('.nii', '')
                group_key = f"{subject_id}_{base_name}"
                
                if group_key not in groups:
                    groups[group_key] = {
                        "files": [],
                        "bids_name": bids_name,
                        "subject": subject_id,
                        "entities": entities
                    }
                
                groups[group_key]["files"].append(dcm_file)
                matched_rule = True
                break
        
        if not matched_rule:
            # No rule matched - create fallback group
            warn(f"  No rule matched for {dcm_file.name}, grouping as unknown")
            group_key = f"unknown_{dcm_file.stem}"
            if group_key not in groups:
                groups[group_key] = {
                    "files": [],
                    "bids_name": dcm_file.stem + ".nii.gz",
                    "subject": "01",
                    "entities": {}
                }
            groups[group_key]["files"].append(dcm_file)
    
    return groups

def _infer_subdirectory(bids_name: str, modality: str) -> str:
    """Infer BIDS subdirectory from filename and modality."""
    name_lower = bids_name.lower()
    
    if modality == "mri":
        if any(kw in name_lower for kw in ['t1w', 't2w', 'flair', 'ct', 'anat']):
            return 'anat'
        elif any(kw in name_lower for kw in ['bold', 'task']):
            return 'func'
        elif 'dwi' in name_lower:
            return 'dwi'
        elif 'fmap' in name_lower or 'fieldmap' in name_lower:
            return 'fmap'
        else:
            return 'anat'
    elif modality == "nirs":
        return 'nirs'
    else:
        return modality

def _build_ascii_tree(root: Path, max_depth: int = 5) -> str:
    """Build ASCII tree representation."""
    lines = [root.name + "/"]
    
    def walk(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            entries = sorted([p for p in directory.iterdir()], key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for i, path in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            line = prefix + connector + path.name
            if path.is_dir():
                line += "/"
            lines.append(line)
            
            if path.is_dir():
                extension = "    " if is_last else "│   "
                walk(path, prefix + extension, depth + 1)
    
    walk(root)
    return "\n".join(lines)

def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute LLM's BIDS plan with automatic DICOM conversion.
    """
    info("Executing BIDS Plan (following LLM's instructions)...")
    
    bids_root = Path(output_dir) / "bids_compatible"
    derivatives_dir = bids_root / "derivatives"
    ensure_dir(bids_root)
    ensure_dir(derivatives_dir)
    
    logs = []
    successes = 0
    failures = 0
    assignment_rules = plan.get("assignment_rules", [])
    
    # Step 1: Copy trio files
    info("\n[1/3] Organizing trio files...")
    trio_files = ["dataset_description.json", "README.md", "participants.tsv"]
    
    for trio_file in trio_files:
        src = output_dir / trio_file
        if not src.exists():
            src = input_root / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")
    
    # Step 2: Process data files according to LLM's mappings
    info("\n[2/3] Organizing BIDS data files...")
    
    mappings = plan.get("mappings", [])
    all_files = list_all_files(input_root)
    
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        filename_rules = mapping.get("filename_rules", [])
        
        info(f"  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        info(f"    Format: {'ready' if format_ready else 'needs conversion (' + convert_to + ')'}")
        
        matched_files = _match_files_with_patterns(all_files, patterns, input_root)
        
        if not matched_files:
            info(f"    No files matched")
            continue
        
        info(f"    Matched: {len(matched_files)} files")
        
        # Handle based on format and conversion needs
        if format_ready and convert_to == "none":
            # Files are ready, just organize
            _process_ready_files(matched_files, filename_rules, assignment_rules, 
                                modality, input_root, bids_root, logs, info)
            successes += len(matched_files)
        
        elif not format_ready and convert_to == "dicom_to_nifti":
            # DICOM conversion needed
            if not check_dcm2niix_available():
                warn(f"    dcm2niix not found - skipping {len(matched_files)} DICOM files")
                warn(f"    Install: apt-get install dcm2niix (Ubuntu) or brew install dcm2niix (macOS)")
                failures += len(matched_files)
                continue
            
            info(f"    Converting DICOM files with dcm2niix...")
            
            # Group DICOM files by LLM rules
            dicom_groups = _group_dicom_files_by_rule(matched_files, filename_rules)
            
            info(f"    Grouped into {len(dicom_groups)} volume(s)")
            
            # Convert each group
            temp_base = Path(output_dir) / "_staging" / "dicom_temp"
            
            for group_key, group_info in dicom_groups.items():
                try:
                    dicom_files = group_info["files"]
                    bids_name = group_info["bids_name"]
                    subject_id = group_info["subject"]
                    
                    # Determine output path
                    subdir = _infer_subdirectory(bids_name, modality)
                    output_path = bids_root / f"sub-{subject_id}" / subdir / bids_name
                    
                    # Create temp dir for this group
                    temp_dir = temp_base / group_key
                    ensure_dir(temp_dir)
                    
                    # Convert
                    info(f"      Converting {len(dicom_files)} files → {bids_name}")
                    result = run_dcm2niix_batch(dicom_files, output_path, temp_dir)
                    
                    if result:
                        logs.append({
                            "source": f"{len(dicom_files)} DICOM files",
                            "destination": str(output_path.relative_to(bids_root)),
                            "action": "dicom_to_nifti",
                            "status": "success",
                            "group": group_key
                        })
                        successes += 1
                    else:
                        failures += 1
                    
                    # Cleanup temp dir
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
                except Exception as e:
                    warn(f"      Conversion failed for {group_key}: {e}")
                    failures += 1
            
            # Cleanup temp base
            if temp_base.exists():
                shutil.rmtree(temp_base, ignore_errors=True)
        
        else:
            warn(f"    Conversion type '{convert_to}' not implemented")
            failures += len(matched_files)
    
    # Step 3: Derivatives
    info("\n[3/3] Organizing derivative files...")
    processed_ext = set()
    for mapping in mappings:
        for pattern in mapping.get("match", []):
            if "." in pattern:
                ext = "." + pattern.split(".")[-1].replace("*", "").replace("]", "")
                processed_ext.add(ext)
    
    for item in input_root.iterdir():
        if item.is_file():
            if item.name in trio_files:
                continue
            if any(item.suffix.endswith(ext) for ext in processed_ext):
                continue
            shutil.copy2(item, derivatives_dir / item.name)
            info(f"  ✓ {item.name}")
    
    # Save reports
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    
    manifest_files = [str(p.relative_to(bids_root)).replace("\\", "/") 
                      for p in bids_root.rglob("*") if p.is_file() and not p.name.startswith('.')]
    
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files": sorted(manifest_files),
        "tree": _build_ascii_tree(bids_root)
    })
    
    info(f"")
    info(f"✓ BIDS Dataset Created")
    info(f"Location: {bids_root}")
    info(f"Successful conversions: {successes}")
    info(f"Failed conversions: {failures}")
    info(f"")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root)
    }

def _process_ready_files(matched_files: List[Path], filename_rules: List[Dict], 
                        assignment_rules: List[Dict], modality: str,
                        input_root: Path, bids_root: Path, logs: List, info_func) -> None:
    """Process files that are already in correct format (NIfTI, SNIRF, etc.)."""
    for src in matched_files:
        try:
            # Apply LLM's filename rules
            bids_name, subject_id, entities = None, None, None
            
            for rule in filename_rules:
                bids_name, subject_id, entities = _apply_llm_filename_rule(src, rule)
                if bids_name:
                    break
            
            # Fallback if no rule matched
            if not subject_id:
                subject_id = _infer_subject_from_assignment_rules(src, assignment_rules)
            
            if not bids_name:
                bids_name = src.name
                warn(f"      No rule matched for {src.name}, keeping original name")
            
            # Determine subdirectory
            subdir = _infer_subdirectory(bids_name, modality)
            
            # Build path
            rel_bids_path = Path(f"sub-{subject_id}/{subdir}/{bids_name}")
            dst = bids_root / rel_bids_path
            
            # Show transformation
            if src.name != bids_name:
                info_func(f"      {src.name}")
                info_func(f"        → {bids_name}")
            else:
                info_func(f"      {src.name}")
            
            ensure_dir(dst.parent)
            copy_file(src, dst)
            
            # Copy sidecar JSON if exists
            if src.suffix == '.gz' and src.stem.endswith('.nii'):
                json_src = src.parent / (src.stem[:-4] + '.json')
            else:
                json_src = src.with_suffix('.json')
            
            if json_src.exists():
                if dst.suffix == '.gz' and dst.stem.endswith('.nii'):
                    json_dst = dst.parent / (dst.stem[:-4] + '.json')
                else:
                    json_dst = dst.with_suffix('.json')
                copy_file(json_src, json_dst)
            
            logs.append({
                "source": str(src.relative_to(input_root)),
                "destination": str(rel_bids_path),
                "action": "organize",
                "status": "success"
            })
            
        except Exception as e:
            warn(f"      Failed: {src.name}: {e}")

def _infer_subject_from_assignment_rules(src: Path, assignment_rules: List[Dict]) -> str:
    """Find subject ID using LLM's assignment rules."""
    src_str = str(src).replace("\\", "/")
    src_name = src.name
    
    for rule in assignment_rules:
        patterns = rule.get("match", [])
        subject = rule.get("subject")
        prefix = rule.get("prefix")
        
        # Try pattern matching
        for pattern in patterns:
            if fnmatch(src_str, pattern) or fnmatch(src_name, pattern):
                return str(subject)
        
        # Try prefix matching
        if prefix and src_name.startswith(prefix):
            return str(subject)
    
    return "01"
