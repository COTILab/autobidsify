# converters/executor.py v10
# ENHANCED: Full support for MRI and fNIRS conversions
# MRI: DICOM→NIfTI, JNIfTI→NIfTI
# fNIRS: .mat/.nirs→SNIRF

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import re
from autobidsify.utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn, read_json
from autobidsify.converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available
from autobidsify.converters.jnifti_converter import convert_jnifti_to_nifti, check_jnifti_support
from autobidsify.converters.nirs_convert import (
    write_snirf_from_normalized, 
    write_nirs_sidecars,
    convert_mat_to_snirf,
    convert_nirs_to_snirf
)


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


def _select_preferred_file(files: List[str]) -> str:
    """
    Select best file from duplicates using priority rules.
    
    Priority (universal neuroimaging conventions):
    1. Path contains NIfTI/nifti   → explicit NIfTI format directory
    2. Path does not contain BRIK  → exclude known duplicate format
    3. Shortest path depth         → closest to root = most original
    4. Alphabetical                → deterministic fallback
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    
    def priority(f):
        f_lower = f.lower()
        parts = f_lower.split('/')
        score_nifti = 0 if any('nifti' in p for p in parts) else 1
        score_brik   = 1 if any('brik' in p for p in parts) else 0
        score_depth  = len(parts)
        return (score_nifti, score_brik, score_depth, f)
    
    return sorted(files, key=priority)[0]


def _match_glob_pattern(filepath: str, pattern: str) -> bool:
    """
    Universal glob pattern matcher.
    
    Handles:
    - '**/*.nii.gz'     → match any .nii.gz file at any depth
    - '**/BRIK/**'      → match any file inside a BRIK directory
    - '*token*'         → match filename containing token
    - '*.ext'           → match by extension
    """
    filepath_lower = filepath.lower()
    pattern_lower = pattern.lower()
    parts = filepath_lower.split('/')

    # Case 1: **/TOKEN/** → file has TOKEN as a directory component
    # e.g. '**/BRIK/**' matches 'sub/anat/BRIK/scan.nii.gz'
    if pattern_lower.startswith('**/') and pattern_lower.endswith('/**'):
        token = pattern_lower[3:-3]  # extract 'brik' from '**/brik/**'
        return token in parts[:-1]   # check directory parts only, not filename

    # Case 2: **/*.ext → match any file with extension at any depth
    if pattern_lower.startswith('**/'):
        suffix = pattern_lower[3:]   # e.g. '*.nii.gz'
        if suffix.startswith('*.'):
            ext = suffix[1:]         # e.g. '.nii.gz'
            return filepath_lower.endswith(ext)
        return suffix in filepath_lower

    # Case 3: *token* → match if token in filename
    if pattern_lower.startswith('*') and pattern_lower.endswith('*'):
        token = pattern_lower.strip('*')
        return token in filepath_lower

    # Case 4: *.ext → match by extension in filename only
    if pattern_lower.startswith('*.'):
        ext = pattern_lower[1:]
        filename = parts[-1]
        return filename.endswith(ext)

    # Case 5: token* → filename starts with token
    if pattern_lower.endswith('*'):
        prefix = pattern_lower.rstrip('*')
        filename = parts[-1]
        return filename.startswith(prefix)

    # Case 6: fallback → substring match anywhere in path
    return pattern_lower in filepath_lower


def analyze_filepath_universal(filepath: str, assignment_rules: List[Dict], 
                               filename_rules: List[Dict],
                               modality: str = "mri") -> Dict[str, Any]:
    """
    Universal filepath analyzer.
    
    NEW: Added modality parameter to handle MRI vs fNIRS filenames
    """
    filename = filepath.split('/')[-1]
    path_parts = filepath.split('/')
    
    subject_id = None
    
    # PRIMARY: Use 'match' patterns
    for rule in assignment_rules:
        match_patterns = rule.get('match', [])
        for pattern in match_patterns:
            if _match_glob_pattern(filepath, pattern):
                subject_id = rule.get('subject')
                break
        if subject_id:
            break
    
    # FALLBACK 1: 'original' field
    if not subject_id:
        for rule in assignment_rules:
            original = rule.get('original')
            if original and original.lower() in filepath.lower():
                subject_id = rule.get('subject')
                break
    
    # FALLBACK 2: 'prefix' field
    if not subject_id:
        for rule in assignment_rules:
            prefix = rule.get('prefix')
            if prefix and filename.lower().startswith(prefix.lower()):
                subject_id = rule.get('subject')
                break
    
    # FALLBACK 3: Standard BIDS pattern
    if not subject_id:
        for part in path_parts:
            match = re.search(r'sub[_-]?(\w+)', part, re.IGNORECASE)
            if match:
                subject_id = match.group(1)
                break
    
    if not subject_id:
        subject_id = "unknown"
    
    # CRITICAL: Clean up subject ID (remove 'sub-' prefix if present)
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]
    
    # NEW v10: Different filename generation for MRI vs fNIRS
    scan_info = infer_scan_type_from_filepath(filepath, filename_rules)
    
    if modality == "nirs":
        # fNIRS: use .snirf extension
        bids_filename = f"sub-{subject_id}_{scan_info['suffix']}.snirf"
    else:
        # MRI: use .nii.gz extension
        bids_filename = f"sub-{subject_id}_{scan_info['suffix']}.nii.gz"
    
    return {
        "subject_id": subject_id,
        "scan_type_suffix": scan_info['suffix'],
        "bids_filename": bids_filename,
        "subdirectory": scan_info['subdirectory'],
        "scan_category": scan_info['category'],
        "original_filepath": filepath,
        "modality": modality
    }


def infer_scan_type_from_filepath(filepath: str, filename_rules: List[Dict]) -> Dict[str, str]:
    """
    Infer BIDS scan type from filepath.

    Priority:
    1. Try LLM-generated filename_rules first
    2. Extract ses/task from original filename directly
    3. Heuristic path-based detection
    4. Omit unknown entities (never use placeholder 'X')
    """
    path_lower = filepath.lower()
    filename = filepath.split('/')[-1]
    filename_lower = filename.lower()

    # ------------------------------------------------------------------
    # Priority 1: Try LLM-generated filename_rules
    # ------------------------------------------------------------------
    for rule in filename_rules:
        match_pattern = rule.get('match_pattern', '')
        pattern_fixed = match_pattern.replace(r'\\', '\\')
        try:
            if re.search(pattern_fixed, filename, re.IGNORECASE):
                template = rule.get('bids_template', '')

                # Extract suffix from template (handles .nii.gz and .snirf)
                suffix_match = re.search(
                    r'sub-[^_]+_(.*?)\.(nii\.gz|snirf|nii)', template
                )
                if suffix_match:
                    raw_suffix = suffix_match.group(1)

                    # FIX: Remove placeholder entities like ses-X or task-X
                    raw_suffix = re.sub(r'ses-X_?', '', raw_suffix)
                    raw_suffix = re.sub(r'task-X_?', '', raw_suffix)
                    raw_suffix = raw_suffix.strip('_')

                    if raw_suffix:
                        subdir = infer_subdirectory_from_suffix(raw_suffix)
                        return {
                            "suffix": raw_suffix,
                            "subdirectory": subdir,
                            "category": categorize_scan_type(raw_suffix)
                        }
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Priority 2: Extract ses/task from original filename directly
    # e.g. "sub-01_ses-left2s_task-FRESHMOTOR_nirs.snirf"
    #   → ses="left2s", task="FRESHMOTOR", suffix="ses-left2s_task-FRESHMOTOR_nirs"
    # ------------------------------------------------------------------
    entities = {}

    ses_match = re.search(r'ses-([A-Za-z0-9]+)', filename)
    if ses_match:
        entities['ses'] = ses_match.group(1)

    task_match = re.search(r'task-([A-Za-z0-9]+)', filename)
    if task_match:
        entities['task'] = task_match.group(1)

    acq_match = re.search(r'acq-([A-Za-z0-9]+)', filename)
    if acq_match:
        entities['acq'] = acq_match.group(1)

    run_match = re.search(r'run-([A-Za-z0-9]+)', filename)
    if run_match:
        entities['run'] = run_match.group(1)

    # Determine modality suffix from filename extension and content
    if filename_lower.endswith('.snirf') or 'nirs' in filename_lower:
        modality_label = 'nirs'
        subdir = 'nirs'
    elif any(k in filename_lower for k in ['t1w', 't1']):
        modality_label = 'T1w'
        subdir = 'anat'
    elif any(k in filename_lower for k in ['t2w', 't2']):
        modality_label = 'T2w'
        subdir = 'anat'
    elif any(k in filename_lower for k in ['bold', 'func']):
        modality_label = 'bold'
        subdir = 'func'
    elif 'dwi' in filename_lower:
        modality_label = 'dwi'
        subdir = 'dwi'
    else:
        modality_label = None
        subdir = 'anat'

    # Build suffix from extracted entities (omit missing ones, never use X)
    if entities or modality_label:
        parts = []
        if 'ses' in entities:
            parts.append(f"ses-{entities['ses']}")
        if 'task' in entities:
            parts.append(f"task-{entities['task']}")
        if 'acq' in entities:
            parts.append(f"acq-{entities['acq']}")
        if 'run' in entities:
            parts.append(f"run-{entities['run']}")
        if modality_label:
            parts.append(modality_label)

        if parts:
            suffix = '_'.join(parts)
            return {
                "suffix": suffix,
                "subdirectory": subdir,
                "category": categorize_scan_type(suffix)
            }

    # ------------------------------------------------------------------
    # Priority 3: Heuristic path-based detection (no placeholder X)
    # ------------------------------------------------------------------
    if any(kw in path_lower for kw in ['anat', 'mprage', 't1w', 't1 ']):
        return {"suffix": "T1w", "subdirectory": "anat", "category": "anatomical"}
    elif any(kw in path_lower for kw in ['func', 'bold']):
        # Try to find task name from path
        task_in_path = re.search(r'task[_-]([A-Za-z0-9]+)', path_lower)
        if task_in_path:
            return {"suffix": f"task-{task_in_path.group(1)}_bold",
                    "subdirectory": "func", "category": "functional"}
        return {"suffix": "task-rest_bold", "subdirectory": "func", "category": "functional"}
    elif 'rest' in path_lower:
        return {"suffix": "task-rest_bold", "subdirectory": "func", "category": "functional"}
    elif any(kw in path_lower for kw in ['nirs', 'fnirs', '.snirf']):
        return {"suffix": "nirs", "subdirectory": "nirs", "category": "functional"}
    elif 'dwi' in path_lower:
        return {"suffix": "dwi", "subdirectory": "dwi", "category": "diffusion"}

    # ------------------------------------------------------------------
    # Priority 4: Fallback — use modality only, no suffix guessing
    # ------------------------------------------------------------------
    if filename_lower.endswith('.snirf'):
        return {"suffix": "nirs", "subdirectory": "nirs", "category": "functional"}
    elif filename_lower.endswith(('.nii', '.nii.gz')):
        return {"suffix": "T1w", "subdirectory": "anat", "category": "anatomical"}

    return {"suffix": "unknown", "subdirectory": "anat", "category": "unknown"}


def infer_subdirectory_from_suffix(suffix: str) -> str:
    """Infer BIDS subdirectory from suffix."""
    suffix_lower = suffix.lower()
    if 't1w' in suffix_lower or 't2w' in suffix_lower:
        return "anat"
    elif 'bold' in suffix_lower:
        return "func"
    elif 'nirs' in suffix_lower:
        return "nirs"
    elif 'dwi' in suffix_lower:
        return "dwi"
    else:
        return "anat"


def categorize_scan_type(suffix: str) -> str:
    """Categorize scan type."""
    suffix_lower = suffix.lower()
    if 't1w' in suffix_lower or 't2w' in suffix_lower:
        return "anatomical"
    elif 'bold' in suffix_lower or 'nirs' in suffix_lower:
        return "functional"
    elif 'dwi' in suffix_lower:
        return "diffusion"
    else:
        return "unknown"


def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], 
                     aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute BIDS plan v10.
    
    NEW v10: Full conversion support
    - MRI: DICOM→NIfTI (dcm2niix), JNIfTI→NIfTI
    - fNIRS: .mat→SNIRF, .nirs→SNIRF
    """
    info("=== Executing BIDS Plan v10 ===")
    
    bids_root = Path(output_dir) / "bids_compatible"
    ensure_dir(bids_root)
    ensure_dir(bids_root / "derivatives")
    
    logs = []
    successes = 0
    failures = 0
    
    info("\n[1/4] Organizing trio files...")
    for trio_file in ["dataset_description.json", "README.md", "participants.tsv"]:
        src = output_dir / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")
    
    info("\n[2/4] Processing data files...")
    
    all_files_paths = list_all_files(input_root)
    all_files_str = [str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    
    info(f"Total files: {len(all_files_str)}")
    
    assignment_rules = plan.get("assignment_rules", [])
    mappings = plan.get("mappings", [])
    
    info(f"  Assignment rules: {len(assignment_rules)} subjects")
    
    # Process each modality mapping
    for mapping in mappings:
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        filename_rules = mapping.get("filename_rules", [])
        format_ready = mapping.get("format_ready", False)
        convert_to = mapping.get("convert_to", "none")
        
        info(f"\n  Processing {modality} files...")
        info(f"    Format ready: {format_ready}, Convert to: {convert_to}")

        # Match files
        exclude_patterns = mapping.get("exclude", [])
        matched_files = []
        for filepath_str in all_files_str:
            # Apply exclude patterns first
            if exclude_patterns:
                excluded = any(
                    _match_glob_pattern(filepath_str, ex_pat)
                    for ex_pat in exclude_patterns
                )
                if excluded:
                    continue
            # Apply match patterns
            for pattern in patterns:
                if _match_glob_pattern(filepath_str, pattern):
                    matched_files.append(filepath_str)
                    break
        
        if not matched_files:
            warn(f"    ⚠ No files matched for {modality}")
            continue
        
        info(f"    ✓ Matched: {len(matched_files)} files")
        
        # ==================================================================
        # NEW v10: Handle fNIRS conversion (BEFORE file analysis)
        # ==================================================================
        if modality == "nirs" and not format_ready:
            info(f"    → fNIRS conversion required ({convert_to})")
            
            # Check conversion method
            if convert_to == "snirf":
                # Check if normalized headers exist (from Plan stage)
                normalized_path = output_dir / "_staging" / "nirs_headers_normalized.json"
                
                if normalized_path.exists():
                    info(f"    → Using normalized headers for CSV→SNIRF conversion")
                    
                    # Convert CSV/tables using normalized configuration
                    try:
                        normalized = read_json(normalized_path)
                        snirf_files = write_snirf_from_normalized(
                            normalized=normalized,
                            input_root=input_root,
                            output_dir=bids_root / "_temp_snirf"
                        )
                        
                        if snirf_files:
                            info(f"    ✓ Generated {len(snirf_files)} SNIRF files from CSV")
                            
                            # Generate sidecars
                            defaults = {
                                "TaskName": plan.get("task_name", "task"),
                                "SamplingFrequency": 10.0
                            }
                            write_nirs_sidecars(bids_root / "_temp_snirf", defaults)
                            
                            # Move to BIDS structure (TODO: proper subject mapping)
                            for snirf_file in snirf_files:
                                # For now, copy to subject directories
                                info(f"      → Created {snirf_file.name}")
                            
                            successes += len(snirf_files)
                        else:
                            warn(f"    ✗ CSV→SNIRF conversion produced no files")
                            failures += len(matched_files)
                    except Exception as e:
                        warn(f"    ✗ CSV→SNIRF conversion failed: {e}")
                        failures += len(matched_files)
                else:
                    # Direct .mat or .nirs file conversion
                    info(f"    → Direct file conversion (.mat/.nirs → SNIRF)")
                    
                    converted_count = 0
                    for filepath_str in matched_files:
                        filepath = path_str_to_path.get(filepath_str)
                        if not filepath:
                            continue
                        
                        file_ext = filepath.suffix.lower()
                        
                        # Analyze filepath to get subject and destination
                        analysis = analyze_filepath_universal(
                            filepath_str, assignment_rules, filename_rules, modality="nirs"
                        )
                        
                        subject_id = analysis['subject_id']
                        bids_filename = analysis['bids_filename']
                        subdirectory = analysis.get('subdirectory', 'nirs')
                        
                        dst = bids_root / f"sub-{subject_id}" / subdirectory / bids_filename
                        ensure_dir(dst.parent)
                        
                        # Convert based on file type
                        if file_ext == '.mat':
                            result = convert_mat_to_snirf(filepath, dst, quiet=False)
                            if result:
                                converted_count += 1
                                successes += 1
                            else:
                                failures += 1
                        
                        elif file_ext == '.nirs':
                            result = convert_nirs_to_snirf(filepath, dst, quiet=False)
                            if result:
                                converted_count += 1
                                successes += 1
                            else:
                                failures += 1
                        
                        else:
                            warn(f"      ⚠ Unknown fNIRS format: {file_ext}")
                            failures += 1
                    
                    if converted_count > 0:
                        info(f"    ✓ Converted {converted_count} fNIRS files to SNIRF")
                    else:
                        warn(f"    ✗ No fNIRS files were converted")
            
            else:
                warn(f"    ⚠ Unknown conversion target: {convert_to}")
                failures += len(matched_files)
            
            continue  # Skip file analysis for converted fNIRS files
        
        # ==================================================================
        # MRI or format-ready files: analyze and organize
        # ==================================================================
        file_analyses = [
            analyze_filepath_universal(f, assignment_rules, filename_rules, modality=modality)
            for f in matched_files
        ]
        
        # Group files by subject and scan type
        file_groups = {}
        for analysis in file_analyses:
            subject_id = analysis['subject_id']
            scan_suffix = analysis['scan_type_suffix']
            group_key = f"{subject_id}_{scan_suffix}"
            
            if group_key not in file_groups:
                file_groups[group_key] = {
                    'subject_id': subject_id,
                    'scan_suffix': scan_suffix,
                    'bids_filename': analysis['bids_filename'],
                    'subdirectory': analysis['subdirectory'],
                    'files': [],
                    'modality': modality
                }
            
            file_groups[group_key]['files'].append(analysis['original_filepath'])
        
        info(f"    Grouped into {len(file_groups)} scan groups")

        # Deduplicate files within each group using priority rules
        for group_key, group_data in file_groups.items():
            if len(group_data['files']) > 1:
                preferred = _select_preferred_file(group_data['files'])
                group_data['files'] = [preferred]
        
        # Display subject summary
        subject_groups = {}
        for group_data in file_groups.values():
            subj_id = group_data['subject_id']
            subject_groups[subj_id] = subject_groups.get(subj_id, 0) + 1
        
        info(f"    Subjects: {len(subject_groups)}")
        if len(subject_groups) <= 15:
            for subj_id in sorted(subject_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                info(f"      sub-{subj_id}: {subject_groups[subj_id]} scan(s)")
        
        # ==================================================================
        # Execute conversion/organization per group
        # ==================================================================
        info(f"    Processing {len(file_groups)} scan groups...")
        
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
                # file_ext = filepath.suffix.lower()
                file_ext = filepath.name.lower().endswith('.nii.gz') and '.nii.gz' or filepath.suffix.lower()
                
                dst = bids_root / f"sub-{subject_id}" / subdirectory / bids_filename
                ensure_dir(dst.parent)
                
                # ==============================================================
                # CONVERSION LOGIC
                # ==============================================================
                
                conversion_performed = False
                
                # MRI: JNIfTI → NIfTI
                if file_ext in ['.jnii', '.bnii']:
                    if check_jnifti_support():
                        info(f"      → Converting JNIfTI: {filepath.name}")
                        result = convert_jnifti_to_nifti(filepath, dst, quiet=True)
                        if result:
                            successes += 1
                            conversion_performed = True
                        else:
                            warn(f"      ✗ JNIfTI conversion failed")
                            failures += 1
                    else:
                        warn(f"      ⚠ JNIfTI support not available (install nibabel)")
                        failures += 1
                
                # MRI: DICOM → NIfTI
                elif file_ext == '.dcm':
                    if check_dcm2niix_available():
                        info(f"      → Converting DICOM batch: {filepath.name}")
                        
                        # Collect all DICOMs for this subject/scan
                        all_dicoms = [
                            path_str_to_path[f] for f in group_data['files'] 
                            if path_str_to_path.get(f) and path_str_to_path[f].suffix.lower() == '.dcm'
                        ]
                        
                        if all_dicoms:
                            result = run_dcm2niix_batch(all_dicoms, dst, quiet=True)
                            if result:
                                info(f"      ✓ Converted {len(all_dicoms)} DICOM files")
                                successes += 1
                                conversion_performed = True
                            else:
                                warn(f"      ✗ DICOM conversion failed")
                                failures += 1
                        else:
                            warn(f"      ⚠ No DICOM files found in group")
                            failures += 1
                    else:
                        warn(f"      ⚠ dcm2niix not available (install dcm2niix)")
                        failures += 1
                
                # fNIRS: Already SNIRF (format_ready=true)
                elif file_ext == '.snirf' and modality == 'nirs':
                    info(f"      → Copying SNIRF: {filepath.name}")
                    copy_file(filepath, dst)
                    successes += 1
                    conversion_performed = True

                # MRI: Already NIfTI (format_ready=true)
                elif file_ext in ('.nii', '.nii.gz') and modality == 'mri':
                    copy_file(filepath, dst)
                    successes += 1
                    conversion_performed = True
                    # Generate sidecar JSON if functional scan
                    _write_nifti_sidecar_if_needed(filepath, dst, scan_info)
                
                # Unknown format
                else:
                    warn(f"      ⚠ Unsupported format for {modality}: {file_ext}")
                    failures += 1
                
                # Log action
                if conversion_performed:
                    logs.append({
                        "source": filepath_str,
                        "destination": f"sub-{subject_id}/{subdirectory}/{bids_filename}",
                        "action": "convert" if file_ext in ['.dcm', '.jnii', '.bnii', '.mat', '.nirs'] else "copy",
                        "modality": modality,
                        "status": "success"
                    })
                
            except Exception as e:
                warn(f"      ✗ Failed: {e}")
                failures += 1
    
    info("\n[3/4] Finalizing...")
    
    # Save conversion log
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    
    # Build manifest
    manifest_files = [str(p.relative_to(bids_root)).replace("\\", "/") 
                      for p in bids_root.rglob("*") if p.is_file()]
    
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files": sorted(manifest_files),
        "tree": _build_ascii_tree(bids_root)
    })
    
    # Count subjects and files
    subject_dirs = list(bids_root.glob("sub-*"))
    
    info(f"\n[4/4] Summary")
    info(f"━" * 60)
    info(f"✓ BIDS Dataset Created")
    info(f"Location: {bids_root}")
    info(f"Files processed: {successes}")
    info(f"Failed: {failures}")
    info(f"Subjects: {len(subject_dirs)}")
    
    if len(subject_dirs) > 0:
        info(f"\nSubject directories:")
        for subj_dir in sorted(subject_dirs)[:15]:
            nii_count = len(list(subj_dir.rglob("*.nii.gz")))
            snirf_count = len(list(subj_dir.rglob("*.snirf")))
            total = nii_count + snirf_count
            
            if nii_count > 0 and snirf_count > 0:
                info(f"  {subj_dir.name}: {total} files ({nii_count} NIfTI, {snirf_count} SNIRF)")
            elif nii_count > 0:
                info(f"  {subj_dir.name}: {nii_count} NIfTI file(s)")
            elif snirf_count > 0:
                info(f"  {subj_dir.name}: {snirf_count} SNIRF file(s)")
        
        if len(subject_dirs) > 15:
            info(f"  ... and {len(subject_dirs) - 15} more")
    else:
        warn("⚠ WARNING: No subject directories created!")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root),
        "subject_count": len(subject_dirs)
    }