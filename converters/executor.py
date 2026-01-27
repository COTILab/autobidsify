# executor.py v6
# UNIVERSAL PATH-BASED APPROACH: Everything inferred from full filepath
# CRITICAL FIX: Use 'match' patterns for subject assignment, not 'original' field

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil
import re
from utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn
from converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available
from converters.jnifti_converter import convert_jnifti_to_nifti, check_jnifti_support


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


def _match_glob_pattern(filepath: str, pattern: str) -> bool:
    """
    Universal glob pattern matcher.
    
    Handles:
    - Simple globs: *.nii, *.dcm
    - Wildcard globs: *neo*, *1yr*
    - Path globs: **/sub-01/**, **/anat/*
    
    Args:
        filepath: File path (relative)
        pattern: Glob pattern
    
    Returns:
        True if filepath matches pattern
    """
    # Convert glob to regex-like matching
    filepath_lower = filepath.lower()
    pattern_lower = pattern.lower()
    
    # Handle ** (match any directory depth)
    if '**/' in pattern_lower:
        pattern_lower = pattern_lower.replace('**/', '')
    
    # Simple wildcard matching
    if pattern_lower.startswith('*') and pattern_lower.endswith('*'):
        # Pattern like '*neo*'
        token = pattern_lower.strip('*')
        return token in filepath_lower
    
    elif pattern_lower.startswith('*.'):
        # Pattern like '*.nii'
        ext = pattern_lower[1:]  # Remove leading *
        return filepath_lower.endswith(ext)
    
    elif pattern_lower.endswith('*'):
        # Pattern like 'infant-*'
        prefix = pattern_lower.rstrip('*')
        filename = filepath.split('/')[-1].lower()
        return filename.startswith(prefix)
    
    else:
        # Direct substring match
        return pattern_lower in filepath_lower


def analyze_filepath_universal(filepath: str, assignment_rules: List[Dict], 
                               filename_rules: List[Dict]) -> Dict[str, Any]:
    """
    Universal filepath analyzer - extracts ALL information from full path.
    
    CRITICAL FIX: Use 'match' patterns for subject assignment, not 'original'.
    """
    filename = filepath.split('/')[-1]
    path_parts = filepath.split('/')
    
    subject_id = None
    
    # ===================================================================
    # PRIMARY MATCHING: Use 'match' patterns (CRITICAL FIX)
    # ===================================================================
    for rule in assignment_rules:
        match_patterns = rule.get('match', [])
        
        # Try each match pattern
        for pattern in match_patterns:
            if _match_glob_pattern(filepath, pattern):
                subject_id = rule.get('subject')
                break
        
        if subject_id:
            break
    
    # ===================================================================
    # FALLBACK 1: Try 'original' field (for backward compatibility)
    # ===================================================================
    if not subject_id:
        for rule in assignment_rules:
            original = rule.get('original')
            
            if original and original.lower() in filepath.lower():
                subject_id = rule.get('subject')
                break
    
    # ===================================================================
    # FALLBACK 2: Try 'prefix' field (for flat structures)
    # ===================================================================
    if not subject_id:
        for rule in assignment_rules:
            prefix = rule.get('prefix')
            
            if prefix and filename.lower().startswith(prefix.lower()):
                subject_id = rule.get('subject')
                break
    
    # ===================================================================
    # FALLBACK 3: Try standard BIDS pattern in path
    # ===================================================================
    if not subject_id:
        for part in path_parts:
            match = re.search(r'sub[_-]?(\w+)', part, re.IGNORECASE)
            if match:
                subject_id = match.group(1)
                break
    
    # ===================================================================
    # LAST RESORT: Mark as unknown
    # ===================================================================
    if not subject_id:
        subject_id = "unknown"
    
    # Infer scan type
    scan_info = infer_scan_type_from_filepath(filepath, filename_rules)
    
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
    
    Strategy:
    1. Try filename_rules from LLM (most specific)
    2. Fallback to heuristic detection
    """
    path_lower = filepath.lower()
    filename = filepath.split('/')[-1].lower()
    
    # ===================================================================
    # PRIMARY: Use LLM-generated filename_rules
    # ===================================================================
    for rule in filename_rules:
        match_pattern = rule.get('match_pattern', '')
        
        # Fix YAML escaping (double backslash -> single)
        pattern_fixed = match_pattern.replace(r'\\', '\\')
        
        try:
            if re.search(pattern_fixed, filename, re.IGNORECASE):
                template = rule.get('bids_template', '')
                
                # Extract suffix from template
                # Template: "sub-X_atlas-aal_T1w.nii.gz"
                # Extract: "atlas-aal_T1w"
                suffix_match = re.search(r'sub-[^_]+_(.*?)\.nii', template)
                if suffix_match:
                    suffix = suffix_match.group(1)
                    subdir = infer_subdirectory_from_suffix(suffix)
                    return {
                        "suffix": suffix,
                        "subdirectory": subdir,
                        "category": categorize_scan_type(suffix)
                    }
        except re.error as e:
            # Regex error, skip this rule
            warn(f"    Regex error in pattern '{match_pattern}': {e}")
            continue
        except Exception:
            continue
    
    # ===================================================================
    # FALLBACK: Heuristic detection
    # ===================================================================
    
    # Anatomical scans
    if any(kw in path_lower for kw in ['anat', 'mprage', 't1w', 't1 ', '/t1/']):
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
        body_parts = ['hip', 'head', 'shoulder', 'knee', 'ankle', 'pelvis', 'chest', 'abdomen']
        
        for part in body_parts:
            if part in path_lower or part in filename:
                suffix = f"acq-ct{part}_T1w"
                return {
                    "suffix": suffix,
                    "subdirectory": "anat",
                    "category": "ct_scan"
                }
        
        suffix = "acq-ct_T1w"
        return {
            "suffix": suffix,
            "subdirectory": "anat",
            "category": "ct_scan"
        }
    
    # Unknown
    else:
        return {
            "suffix": "unknown",
            "subdirectory": "anat",
            "category": "unknown"
        }


def infer_subdirectory_from_suffix(suffix: str) -> str:
    """Infer BIDS subdirectory from suffix."""
    suffix_lower = suffix.lower()
    
    if 't1w' in suffix_lower or 't2w' in suffix_lower or 'flair' in suffix_lower:
        return "anat"
    elif 'bold' in suffix_lower or 'task' in suffix_lower:
        return "func"
    elif 'dwi' in suffix_lower:
        return "dwi"
    elif 'fieldmap' in suffix_lower or 'epi' in suffix_lower:
        return "fmap"
    elif 'probseg' in suffix_lower or 'dseg' in suffix_lower:
        return "anat"  # Segmentation files go to anat
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
    elif 'probseg' in suffix_lower or 'dseg' in suffix_lower:
        return "segmentation"
    else:
        return "unknown"


def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], 
                     aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute BIDS plan v6 - UNIVERSAL approach with robust pattern matching.
    
    CRITICAL FIX: Use 'match' patterns for subject assignment
    """
    info("Executing BIDS Plan (v6: universal pattern matching)...")
    
    bids_root = Path(output_dir) / "bids_compatible"
    derivatives_dir = bids_root / "derivatives"
    ensure_dir(bids_root)
    ensure_dir(derivatives_dir)
    
    logs = []
    successes = 0
    failures = 0
    
    # ===================================================================
    # [1/3] Copy trio files
    # ===================================================================
    info("\n[1/3] Organizing trio files...")
    trio_files = ["dataset_description.json", "README.md", "participants.tsv", "participants.json"]
    
    for trio_file in trio_files:
        src = output_dir / trio_file
        if not src.exists():
            src = input_root / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")
    
    # ===================================================================
    # [2/3] Process data files
    # ===================================================================
    info("\n[2/3] Analyzing and organizing files...")
    
    all_files_paths = list_all_files(input_root)
    all_files_str = [str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    
    info(f"Total files in input: {len(all_files_str)}")
    
    assignment_rules = plan.get("assignment_rules", [])
    mappings = plan.get("mappings", [])
    
    info(f"  Assignment rules: {len(assignment_rules)} subjects")
    
    if assignment_rules:
        # Display assignment rules for debugging
        for rule in assignment_rules[:5]:
            subject = rule.get('subject')
            match_patterns = rule.get('match', [])
            info(f"    Subject '{subject}': {match_patterns}")
        if len(assignment_rules) > 5:
            info(f"    ... and {len(assignment_rules) - 5} more")
    
    # Process each mapping (modality)
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        filename_rules = mapping.get("filename_rules", [])
        
        info(f"\n  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        
        # Match files using mapping patterns
        matched_files = []
        for filepath_str in all_files_str:
            for pattern in patterns:
                if _match_glob_pattern(filepath_str, pattern):
                    matched_files.append(filepath_str)
                    break
        
        if not matched_files:
            warn(f"    ⚠ No files matched patterns: {patterns}")
            continue
        
        info(f"    ✓ Matched: {len(matched_files)} files")
        
        # Analyze each file
        file_analyses = []
        
        for filepath_str in matched_files:
            analysis = analyze_filepath_universal(filepath_str, assignment_rules, filename_rules)
            file_analyses.append(analysis)
        
        # Group by subject and scan type
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
        
        # Calculate subject distribution
        subject_groups = {}
        for group_key, group_data in file_groups.items():
            subj_id = group_data['subject_id']
            subject_groups[subj_id] = subject_groups.get(subj_id, 0) + 1
        
        unique_subjects = len(subject_groups)
        avg_scans_per_subject = sum(subject_groups.values()) / unique_subjects if unique_subjects > 0 else 0
        
        info(f"    Subject distribution: {unique_subjects} subjects, avg {avg_scans_per_subject:.1f} scan types/subject")
        
        # Debug: Show subject breakdown
        if unique_subjects <= 10:
            for subj_id in sorted(subject_groups.keys()):
                count = subject_groups[subj_id]
                info(f"      sub-{subj_id}: {count} scan types")
        
        unknown_count = sum(1 for gk in file_groups.keys() if 'unknown' in gk)
        if unknown_count > 0:
            warn(f"    ⚠ {unknown_count} groups with unknown subject ID")
        
        # ===================================================================
        # Execute based on conversion type
        # ===================================================================
        
        if format_ready and convert_to == "none":
            # Direct copy (already in correct format)
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
            # DICOM to NIfTI conversion
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
        
        elif not format_ready and convert_to == "jnifti_to_nifti":
            # JNIfTI to NIfTI conversion
            if not check_jnifti_support():
                warn(f"    JNIfTI conversion not available (missing nibabel)")
                warn(f"    Install with: pip install nibabel")
                failures += len(matched_files)
                continue
            
            info(f"    Converting {len(file_groups)} JNIfTI files...")
            
            progress_count = 0
            progress_interval = max(1, len(file_groups) // 10) if len(file_groups) > 10 else 1
            
            for group_key, group_data in file_groups.items():
                try:
                    file_paths_str = group_data['files']
                    
                    if not file_paths_str:
                        failures += 1
                        continue
                    
                    input_path_str = file_paths_str[0]
                    input_path = path_str_to_path.get(input_path_str)
                    
                    if not input_path or not input_path.exists():
                        warn(f"      File not found: {input_path_str}")
                        failures += 1
                        continue
                    
                    subject_id = group_data['subject_id']
                    bids_filename = group_data['bids_filename']
                    subdirectory = group_data['subdirectory']
                    
                    output_path = bids_root / f"sub-{subject_id}" / subdirectory / bids_filename
                    
                    result = convert_jnifti_to_nifti(input_path, output_path, quiet=True)
                    
                    if result:
                        logs.append({
                            "source": input_path_str,
                            "destination": f"sub-{subject_id}/{subdirectory}/{bids_filename}",
                            "action": "jnifti_to_nifti",
                            "status": "success"
                        })
                        successes += 1
                    else:
                        failures += 1
                    
                    progress_count += 1
                    if progress_count % progress_interval == 0 or progress_count == len(file_groups):
                        percent = (progress_count / len(file_groups)) * 100
                        info(f"      Progress: {progress_count}/{len(file_groups)} ({percent:.0f}%)")
                        
                except Exception as e:
                    warn(f"      JNIfTI conversion failed: {e}")
                    import traceback
                    traceback.print_exc()
                    failures += 1
    
    # ===================================================================
    # [3/3] Finalize
    # ===================================================================
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
        
        # Debug information
        if assignment_rules:
            warn("\nDebug: Assignment rules present but no matches found")
            warn("Check if 'match' patterns in BIDSPlan.yaml are correct")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root),
        "subject_count": len(subject_dirs)
    }
