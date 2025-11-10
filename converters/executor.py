# executor.py v2
# Execute LLM's plan using universal_core - completely rewritten for generality

from pathlib import Path
from typing import Dict, Any, List
import shutil
from utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn
from converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available
from universal_core import FileStructureAnalyzer, UniversalFileMatcher, SmartFileGrouper, build_bids_filename

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

def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute BIDS plan - v2 using universal_core.
    
    UNIVERSAL SOLUTION:
    - Works with ANY directory structure
    - Uses full file path matching
    - Automatic duplicate handling
    - Smart format selection (NIfTI over BRIK)
    
    Args:
        input_root: Input data directory
        output_dir: Output directory
        plan: BIDS plan from planner
        aux_inputs: Auxiliary inputs
    
    Returns:
        Execution summary
    """
    info("Executing BIDS Plan (v2: universal matching engine)...")
    
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
    
    # === Step 2: Process data files ===
    info("\n[2/3] Processing data files with universal matcher...")
    
    # Get all files and their paths
    all_files_paths = list_all_files(input_root)
    all_files_str = [str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths]
    
    # Create Path lookup
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    
    info(f"Total files in input: {len(all_files_str)}")
    
    # Create analyzer for grouping
    analyzer = FileStructureAnalyzer(all_files_str)
    subject_detection = analyzer.detect_subject_identifiers()
    
    grouper = SmartFileGrouper(analyzer)
    
    mappings = plan.get("mappings", [])
    
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        
        info(f"\n  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        info(f"    Patterns: {patterns}")
        
        # === Use UniversalFileMatcher ===
        # Auto-exclude BRIK if NIfTI exists
        exclude_patterns = ["**/BRIK/**", "**/brik/**"]
        
        matched_str = UniversalFileMatcher.match_files_batch(
            all_files_str, 
            patterns,
            exclude_patterns
        )
        
        if not matched_str:
            warn(f"    ⚠ No files matched")
            continue
        
        # Convert back to Path objects
        matched_paths = [path_str_to_path[s] for s in matched_str if s in path_str_to_path]
        
        info(f"    ✓ Matched: {len(matched_paths)} files")
        
        # === Group files intelligently ===
        groups = grouper.group_by_subject_and_scan(matched_str, subject_detection)
        
        info(f"    Grouped into: {len(groups)} scan groups")
        
        # === Process each group ===
        if format_ready and convert_to == "none":
            # Files already in correct format, just organize
            
            # Smart logging: show summary instead of every file
            total_groups = len(groups)
            info(f"    Processing {total_groups} file groups...")
            
            # Show progress every N groups
            progress_interval = max(1, total_groups // 10)  # Show ~10 progress updates
            processed = 0
            
            for group_key, group_data in groups.items():
                try:
                    preferred_path_str = group_data["preferred_file"]
                    preferred_path = path_str_to_path.get(preferred_path_str)
                    
                    if not preferred_path:
                        warn(f"      Preferred file not found: {preferred_path_str}")
                        continue
                    
                    subject_id = group_data["subject_id"]
                    bids_filename = group_data["bids_filename"]
                    scan_type = group_data["scan_type"]
                    
                    # Determine subdirectory
                    if scan_type == "func":
                        subdir = "func"
                    elif scan_type == "dwi":
                        subdir = "dwi"
                    elif scan_type == "fmap":
                        subdir = "fmap"
                    else:
                        subdir = "anat"
                    
                    dst = bids_root / f"sub-{subject_id}" / subdir / bids_filename
                    
                    ensure_dir(dst.parent)
                    copy_file(preferred_path, dst)
                    
                    # Copy JSON sidecar if exists
                    if preferred_path.suffix == '.gz' and preferred_path.stem.endswith('.nii'):
                        json_src = preferred_path.parent / (preferred_path.stem[:-4] + '.json')
                    else:
                        json_src = preferred_path.with_suffix('.json')
                    
                    if json_src.exists():
                        json_dst = dst.parent / (dst.stem.replace('.nii', '') + '.json')
                        copy_file(json_src, json_dst)
                    
                    logs.append({
                        "source": preferred_path_str,
                        "destination": f"sub-{subject_id}/{subdir}/{bids_filename}",
                        "action": "organize",
                        "status": "success",
                        "duplicates_resolved": len(group_data["files"]) > 1
                    })
                    successes += 1
                    processed += 1
                    
                    # Show progress
                    if processed % progress_interval == 0 or processed == total_groups:
                        percent = (processed / total_groups) * 100
                        info(f"      Progress: {processed}/{total_groups} ({percent:.0f}%)")
                    
                except Exception as e:
                    warn(f"      Failed: {e}")
                    failures += 1
        
        elif not format_ready and convert_to == "dicom_to_nifti":
            # DICOM conversion
            if not check_dcm2niix_available():
                warn(f"    dcm2niix not available, skipping")
                failures += len(matched_paths)
                continue
            
            temp_base = Path(output_dir) / "_staging" / "dicom_temp"
            
            total_groups = len(groups)
            info(f"    Converting {total_groups} DICOM volume groups...")
            
            progress_interval = max(1, total_groups // 10)
            processed = 0
            
            for group_key, group_data in groups.items():
                try:
                    file_paths_str = group_data["files"]
                    file_paths = [path_str_to_path[s] for s in file_paths_str if s in path_str_to_path]
                    
                    subject_id = group_data["subject_id"]
                    bids_filename = group_data["bids_filename"]
                    scan_type = group_data["scan_type"]
                    
                    subdir = "anat" if scan_type == "anat" else scan_type
                    output_path = bids_root / f"sub-{subject_id}" / subdir / bids_filename
                    
                    temp_dir = temp_base / group_key
                    ensure_dir(temp_dir)
                    
                    # Use quiet mode to suppress verbose dcm2niix output
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
                    
                    processed += 1
                    
                    # Show progress
                    if processed % progress_interval == 0 or processed == total_groups:
                        percent = (processed / total_groups) * 100
                        info(f"      Progress: {processed}/{total_groups} ({percent:.0f}%) - Latest: {bids_filename}")
                        
                except Exception as e:
                    warn(f"      Conversion failed: {e}")
                    failures += 1
            
            if temp_base.exists():
                shutil.rmtree(temp_base, ignore_errors=True)
    
    # === Step 3: Save outputs ===
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
