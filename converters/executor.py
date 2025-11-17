# executor.py v5
# UNIVERSAL PATH-BASED APPROACH: Everything inferred from full filepath

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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


# ============================================================================
# UNIVERSAL PATH ANALYZER - Works for ANY dataset structure
# ============================================================================

def analyze_filepath_universal(filepath: str, assignment_rules: List[Dict], 
                               filename_rules: List[Dict]) -> Dict[str, Any]:
    """
    Universal filepath analyzer - extracts ALL information from full path.
    
    Strategy:
    1. Extract subject ID from filepath (using assignment_rules)
    2. Infer scan type from filepath (path keywords + filename)
    3. Generate BIDS-compliant filename
    
    Works for:
    - Hierarchical: "Site_subID/anat/scan.nii.gz"
    - Flat: "VHM-Hip-134.dcm"
    - Mixed: any combination
    
    Args:
        filepath: Full relative path (e.g., "Newark_sub41006/anat_mprage/scan.nii.gz")
        assignment_rules: Subject assignment rules from plan
        filename_rules: Filename rules from plan (used as hints)
    
    Returns:
        {
            "subject_id": "41006",
            "scan_type_suffix": "acq-anonymized_T1w",
            "bids_filename": "sub-41006_acq-anonymized_T1w.nii.gz",
            "subdirectory": "anat",
            "original_filepath": filepath
        }
    """
    filename = filepath.split('/')[-1]
    path_parts = filepath.split('/')
    
    # ========== Step 1: Extract subject ID from filepath ==========
    subject_id = None
    
    # Try assignment_rules first
    for rule in assignment_rules:
        original = rule.get('original')
        prefix = rule.get('prefix')
        
        # Check full filepath for original ID (hierarchical structures)
        if original and original in filepath:
            subject_id = rule.get('subject')
            break
        
        # Check filename for prefix (flat structures)
        if prefix and filename.startswith(prefix):
            subject_id = rule.get('subject')
            break
    
    # Fallback: try to extract from path parts
    if not subject_id:
        for part in path_parts:
            # Pattern: Site_subID or sub-ID
            match = re.search(r'sub[_-]?(\d+)', part, re.IGNORECASE)
            if match:
                subject_id = match.group(1)
                break
    
    if not subject_id:
        subject_id = "unknown"
    
    # ========== Step 2: Infer scan type from filepath ==========
    scan_info = infer_scan_type_from_filepath(filepath, filename_rules)
    
    # ========== Step 3: Generate BIDS filename ==========
    bids_filename = f"sub-{subject_id}_{scan_info['suffix']}.nii.gz"
    
    return {
        "subject_id": subject_id,
        "scan_type_suffix": scan_info['suffix'],
        "bids_filename": bids_filename,
        "subdirectory": scan_info['subdirectory'],
        "scan_category": scan_info['category'],
        "original_filepath": filepath
    }


def infer_scan_type_from_filepath(filepath: str, 
                                  filename_rules: List[Dict]) -> Dict[str, str]:
    """
    Infer BIDS scan type from FULL filepath (path + filename).
    
    Strategy (in order of priority):
    1. Try filename_rules first (if LLM provided specific rules)
    2. Check path keywords (anat, func, dwi, etc.)
    3. Check processing state (anonymized, skullstripped, etc.)
    4. Fallback to generic unknown
    
    Examples:
        "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz"
        → suffix: "acq-anonymized_T1w", subdirectory: "anat"
        
        "Berlin_sub06204/func_rest/NIfTI/scan_rest.nii.gz"
        → suffix: "task-rest_bold", subdirectory: "func"
        
        "VHMCT1mm-Hip (134).dcm"
        → suffix: "acq-cthip_T1w", subdirectory: "anat"
    
    Args:
        filepath: Full relative path
        filename_rules: LLM's filename rules (used as hints)
    
    Returns:
        {
            "suffix": BIDS suffix (e.g., "T1w", "task-rest_bold"),
            "subdirectory": BIDS subdirectory (e.g., "anat", "func"),
            "category": General category (e.g., "anatomical", "functional")
        }
    """
    path_lower = filepath.lower()
    filename = filepath.split('/')[-1].lower()
    
    # ========== Priority 1: Try filename_rules ==========
    for rule in filename_rules:
        match_pattern = rule.get('match_pattern', '')
        pattern_fixed = match_pattern.replace(r'\\', '\\')
        
        try:
            if re.search(pattern_fixed, filename, re.IGNORECASE):
                template = rule.get('bids_template', '')
                # Extract suffix from template: "sub-1_acq-cthip_T1w.nii.gz" → "acq-cthip_T1w"
                suffix_match = re.search(r'sub-\d+_(.*?)\.nii', template)
                if suffix_match:
                    suffix = suffix_match.group(1)
                    subdir = infer_subdirectory_from_suffix(suffix)
                    return {
                        "suffix": suffix,
                        "subdirectory": subdir,
                        "category": categorize_scan_type(suffix)
                    }
        except:
            continue
    
    # ========== Priority 2: Analyze path keywords ==========
    
    # Anatomical scans
    if any(kw in path_lower for kw in ['anat', 'mprage', 't1w', 't1 ', '/t1/']):
        # Check processing state
        if 'anonymized' in path_lower or 'anonymised' in path_lower:
            suffix = "acq-anonymized_T1w"
        elif 'skullstripped' in path_lower or 'skullstrip' in path_lower:
            suffix = "acq-skullstripped_T1w"
        elif 'normalized' in path_lower or 'normalised' in path_lower:
            suffix = "acq-normalized_T1w"
        else:
            suffix = "T1w"
        
        return {
            "suffix": suffix,
            "subdirectory": "anat",
            "category": "anatomical"
        }
    
    elif any(kw in path_lower for kw in ['t2w', 't2 ', '/t2/', 'space']):
        suffix = "T2w"
        return {
            "suffix": suffix,
            "subdirectory": "anat",
            "category": "anatomical"
        }
    
    # Functional scans
    elif any(kw in path_lower for kw in ['func', 'bold', 'rest', 'task']):
        if 'rest' in path_lower:
            suffix = "task-rest_bold"
        elif 'movie' in path_lower:
            suffix = "task-movie_bold"
        elif 'sensorimotor' in path_lower:
            suffix = "task-sensorimotor_bold"
        else:
            suffix = "task-unknown_bold"
        
        return {
            "suffix": suffix,
            "subdirectory": "func",
            "category": "functional"
        }
    
    # Diffusion
    elif any(kw in path_lower for kw in ['dwi', 'diffusion', 'dti']):
        suffix = "dwi"
        return {
            "suffix": suffix,
            "subdirectory": "dwi",
            "category": "diffusion"
        }
    
    # Fieldmap
    elif any(kw in path_lower for kw in ['fmap', 'fieldmap', 'b0']):
        suffix = "fieldmap"
        return {
            "suffix": suffix,
            "subdirectory": "fmap",
            "category": "fieldmap"
        }
    
    # CT scans (special handling)
    elif 'ct' in path_lower or 'ct' in filename:
        # Extract body part if available
        body_parts = ['hip', 'head', 'shoulder', 'knee', 'ankle', 'pelvis', 'chest', 'abdomen']
        
        for part in body_parts:
            if part in path_lower or part in filename:
                suffix = f"acq-ct{part}_T1w"
                return {
                    "suffix": suffix,
                    "subdirectory": "anat",
                    "category": "ct_scan"
                }
        
        # Generic CT
        suffix = "acq-ct_T1w"
        return {
            "suffix": suffix,
            "subdirectory": "anat",
            "category": "ct_scan"
        }
    
    # Fallback: unknown
    else:
        return {
            "suffix": "unknown",
            "subdirectory": "anat",
            "category": "unknown"
        }


def infer_subdirectory_from_suffix(suffix: str) -> str:
    """
    Infer BIDS subdirectory from suffix.
    
    Examples:
        "T1w" → "anat"
        "task-rest_bold" → "func"
        "dwi" → "dwi"
    """
    suffix_lower = suffix.lower()
    
    if 't1w' in suffix_lower or 't2w' in suffix_lower or 'flair' in suffix_lower:
        return "anat"
    elif 'bold' in suffix_lower or 'task' in suffix_lower:
        return "func"
    elif 'dwi' in suffix_lower:
        return "dwi"
    elif 'fieldmap' in suffix_lower or 'epi' in suffix_lower:
        return "fmap"
    else:
        return "anat"


def categorize_scan_type(suffix: str) -> str:
    """Categorize scan type for logging/debugging"""
    suffix_lower = suffix.lower()
    
    if 't1w' in suffix_lower or 't2w' in suffix_lower:
        return "anatomical"
    elif 'bold' in suffix_lower or 'task' in suffix_lower:
        return "functional"
    elif 'dwi' in suffix_lower:
        return "diffusion"
    elif 'fieldmap' in suffix_lower:
        return "fieldmap"
    else:
        return "unknown"


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], 
                     aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute BIDS plan v5 - UNIVERSAL path-based approach.
    
    Key Innovation: ALL information extracted from full filepath.
    No separate handling for hierarchical vs flat structures.
    
    Args:
        input_root: Input data directory
        output_dir: Output directory
        plan: BIDS plan from planner
        aux_inputs: Auxiliary inputs
    
    Returns:
        Execution summary
    """
    info("Executing BIDS Plan (v5: universal path-based approach)...")
    
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
    
    # === Step 2: Universal filepath analysis ===
    info("\n[2/3] Analyzing and organizing files...")
    
    all_files_paths = list_all_files(input_root)
    all_files_str = [str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    
    info(f"Total files in input: {len(all_files_str)}")
    
    assignment_rules = plan.get("assignment_rules", [])
    mappings = plan.get("mappings", [])
    
    info(f"  Assignment rules: {len(assignment_rules)} subjects")
    
    if assignment_rules:
        sample_rules = assignment_rules[:3]
        for rule in sample_rules:
            original = rule.get('original', rule.get('prefix', 'N/A'))
            subject = rule.get('subject')
            info(f"    '{original}' → sub-{subject}")
        if len(assignment_rules) > 3:
            info(f"    ... and {len(assignment_rules) - 3} more")
    
    # Process each mapping
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        filename_rules = mapping.get("filename_rules", [])
        
        info(f"\n  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        
        # Match files by extension
        matched_files = []
        for filepath_str in all_files_str:
            filename = filepath_str.split('/')[-1]
            
            for pattern in patterns:
                if pattern == "**/*.dcm" and filename.endswith('.dcm'):
                    matched_files.append(filepath_str)
                    break
                elif pattern == "**/*.nii.gz" and filename.endswith('.nii.gz'):
                    matched_files.append(filepath_str)
                    break
        
        if not matched_files:
            warn(f"    ⚠ No files matched")
            continue
        
        info(f"    ✓ Matched: {len(matched_files)} files")
        
        # ========== UNIVERSAL ANALYSIS: Analyze each filepath ==========
        file_analyses = []
        
        for filepath_str in matched_files:
            analysis = analyze_filepath_universal(filepath_str, assignment_rules, filename_rules)
            file_analyses.append(analysis)
        
        # Group by subject + scan type
        file_groups = {}
        
        for analysis in file_analyses:
            subject_id = analysis['subject_id']
            scan_suffix = analysis['scan_type_suffix']
            bids_filename = analysis['bids_filename']
            subdirectory = analysis['subdirectory']
            
            group_key = f"{subject_id}_{scan_suffix}"
            
            if group_key not in file_groups:
                file_groups[group_key] = {
                    'subject_id': subject_id,
                    'scan_suffix': scan_suffix,
                    'bids_filename': bids_filename,
                    'subdirectory': subdirectory,
                    'files': []
                }
            
            file_groups[group_key]['files'].append(analysis['original_filepath'])
        
        info(f"    Grouped into: {len(file_groups)} scan groups")
        
        # Show grouping summary
        subject_groups = {}
        for group_key, group_data in file_groups.items():
            subj_id = group_data['subject_id']
            subject_groups[subj_id] = subject_groups.get(subj_id, 0) + 1
        
        unique_subjects = len(subject_groups)
        avg_scans_per_subject = sum(subject_groups.values()) / unique_subjects if unique_subjects > 0 else 0
        
        info(f"    Subject distribution: {unique_subjects} subjects, avg {avg_scans_per_subject:.1f} scan types/subject")
        
        # Check for unknown subjects
        unknown_count = sum(1 for gk in file_groups.keys() if 'unknown' in gk)
        if unknown_count > 0:
            warn(f"    ⚠ {unknown_count} groups with unknown subject ID")
        
        # === Process each group ===
        if format_ready and convert_to == "none":
            info(f"    Organizing {len(file_groups)} scan groups...")
            
            progress_count = 0
            progress_interval = max(1, len(file_groups) // 10) if len(file_groups) > 10 else 1
            
            for group_key, group_data in file_groups.items():
                try:
                    filepath_str = group_data['files'][0]
                    filepath = path_str_to_path.get(filepath_str)
                    
                    if not filepath:
                        failures += 1
                        continue
                    
                    subject_id = group_data['subject_id']
                    bids_filename = group_data['bids_filename']
                    subdirectory = group_data['subdirectory']
                    
                    dst = bids_root / f"sub-{subject_id}" / subdirectory / bids_filename
                    ensure_dir(dst.parent)
                    copy_file(filepath, dst)
                    
                    logs.append({
                        "source": filepath_str,
                        "destination": f"sub-{subject_id}/{subdirectory}/{bids_filename}",
                        "action": "organize",
                        "status": "success"
                    })
                    successes += 1
                    
                    progress_count += 1
                    if progress_count % progress_interval == 0 or progress_count == len(file_groups):
                        percent = (progress_count / len(file_groups)) * 100
                        info(f"      Progress: {progress_count}/{len(file_groups)} ({percent:.0f}%)")
                    
                except Exception as e:
                    warn(f"      Failed: {e}")
                    failures += 1
        
        elif not format_ready and convert_to == "dicom_to_nifti":
            if not check_dcm2niix_available():
                warn(f"    dcm2niix not available")
                failures += len(matched_files)
                continue
            
            temp_base = Path(output_dir) / "_staging" / "dicom_temp"
            
            info(f"    Converting {len(file_groups)} DICOM groups...")
            
            progress_count = 0
            progress_interval = max(1, len(file_groups) // 10)
            
            for group_key, group_data in file_groups.items():
                try:
                    file_paths_str = group_data['files']
                    file_paths = [path_str_to_path[s] for s in file_paths_str if s in path_str_to_path]
                    
                    subject_id = group_data['subject_id']
                    bids_filename = group_data['bids_filename']
                    subdirectory = group_data['subdirectory']
                    
                    output_path = bids_root / f"sub-{subject_id}" / subdirectory / bids_filename
                    
                    temp_dir = temp_base / group_key
                    ensure_dir(temp_dir)
                    
                    result = run_dcm2niix_batch(file_paths, output_path, temp_dir, quiet=True)
                    
                    if result:
                        logs.append({
                            "source": f"{len(file_paths)} DICOM files",
                            "destination": f"sub-{subject_id}/{subdirectory}/{bids_filename}",
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
                        info(f"      Progress: {progress_count}/{len(file_groups)} ({percent:.0f}%)")
                        
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
        warn("This indicates a problem with subject matching or file processing")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root),
        "subject_count": len(subject_dirs)
    }
