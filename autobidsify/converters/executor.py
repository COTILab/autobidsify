# converters/executor.py v10
# ENHANCED: Full support for MRI and fNIRS conversions
# MRI: DICOM→NIfTI, JNIfTI→NIfTI
# fNIRS: .mat/.nirs→SNIRF

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import re
from collections import defaultdict
from autobidsify.utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn, read_json
from autobidsify.converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available
from autobidsify.converters.jnifti_converter import convert_jnifti_to_nifti, check_jnifti_support
from autobidsify.converters.nirs_convert import (
    write_snirf_from_normalized,
    write_nirs_sidecars,
    convert_mat_to_snirf,
    convert_nirs_to_snirf,
)


# ============================================================================
# ASCII tree
# ============================================================================

def _build_ascii_tree(root: Path, max_depth: int = 3) -> str:
    """Build ASCII tree visualization of a directory."""
    lines = [root.name + "/"]

    def walk(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                list(directory.iterdir()),
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except PermissionError:
            return
        entries = entries[: (15 if depth == 0 else 8)]
        for i, path in enumerate(entries):
            is_last  = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + path.name + ("/" if path.is_dir() else ""))
            if path.is_dir() and depth < max_depth - 1:
                walk(path, prefix + ("    " if is_last else "│   "), depth + 1)

    walk(root)
    return "\n".join(lines)


# ============================================================================
# Filename helpers
# ============================================================================

def _normalize_filename(filepath: str) -> str:
    """
    Normalize a filename by stripping extensions and trailing sequence numbers.

    Used to:
    - Identify DICOM series (same series = same normalized name).
    - Detect format duplicates (same content in different format directories).

    Examples:
        'VHFCT1mm-Hip (134).dcm'        → 'vhfct1mm-hip'
        'scan_mprage_anonymized.nii.gz'  → 'scan_mprage_anonymized'
        'scan_001.dcm'                   → 'scan'
    """
    name = filepath.split("/")[-1]
    while "." in name and len(name.split(".")[-1]) <= 6:
        name = name.rsplit(".", 1)[0]
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)   # strip trailing " (N)"
    name = re.sub(r"[_\-]\d+$", "", name)         # strip trailing _NNN or -NNN
    return name.strip().lower()


def _extract_acq_label(normalized_fname: str) -> str:
    """
    Extract a short, clean acq- label from a normalized DICOM filename.

    Strategy: split on digit boundaries, keep the last alphabetic token
    that is longer than 2 characters and is not a known scanner/format prefix.
    This isolates the body-part or scan-descriptor word.

    Examples:
        'vhfct1mmankle' → 'ankle'
        'vhfct1mmhead'  → 'head'
        'vhmct1mmhip'   → 'hip'
        'scanmprage'    → 'mprage'

    FIX: previously the entire normalized name (e.g. 'vhfct1mmankle') was used
    as the acq- label, producing names that were both non-descriptive and too
    long for some validators.
    """
    skip = {"vhf", "vhm", "ct", "mr", "mri", "mm", "scan", "the"}
    tokens = re.findall(r"[a-z]+", normalized_fname)
    meaningful = [t for t in tokens if len(t) > 2 and t not in skip]
    if meaningful:
        return meaningful[-1]          # last meaningful token = body part
    return normalized_fname[:20]       # fallback: cap at 20 chars


def _select_preferred_file(files: List[str]) -> str:
    """
    Select the best representative file from a set of format duplicates.

    Priority:
    1. Path contains 'nifti'  → explicit NIfTI format directory (preferred).
    2. Path does NOT contain 'brik' → exclude known duplicate-format directory.
    3. Shortest path depth → closest to root = most original copy.
    4. Alphabetical → deterministic tiebreak.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    def priority(f):
        parts = f.lower().split("/")
        return (
            0 if any("nifti" in p for p in parts) else 1,
            1 if any("brik"  in p for p in parts) else 0,
            len(parts),
            f,
        )

    return sorted(files, key=priority)[0]


# ============================================================================
# Glob pattern matching
# ============================================================================

def _match_glob_pattern(filepath: str, pattern: str) -> bool:
    """
    Universal glob-style pattern matcher for relative file paths.

    Supported patterns:
        '**/*.nii.gz'   → any .nii.gz at any depth
        '**/BRIK/**'    → any file inside a BRIK directory
        '*token*'       → filepath contains token
        '*.ext'         → filename ends with extension
        'token*'        → filename starts with token
        'plain'         → substring anywhere in path (fallback)
    """
    fp  = filepath.lower()
    pat = pattern.lower()
    parts = fp.split("/")

    # **/TOKEN/** — directory component match
    if pat.startswith("**/") and pat.endswith("/**"):
        token = pat[3:-3]
        return token in parts[:-1]

    # **/*.ext — any depth extension match
    if pat.startswith("**/"):
        suffix = pat[3:]
        if suffix.startswith("*."):
            return fp.endswith(suffix[1:])
        return suffix in fp

    # *token* — substring in full path
    if pat.startswith("*") and pat.endswith("*"):
        return pat.strip("*") in fp

    # *.ext — extension match on filename only
    if pat.startswith("*."):
        return parts[-1].endswith(pat[1:])

    # token* — filename prefix
    if pat.endswith("*"):
        return parts[-1].startswith(pat.rstrip("*"))

    # fallback — substring anywhere
    return pat in fp


# ============================================================================
# Scan-type inference
# ============================================================================

def infer_scan_type_from_filepath(filepath: str, filename_rules: List[Dict]) -> Dict[str, str]:
    """
    Infer BIDS scan-type suffix and subdirectory from a file path.

    Priority:
    1. LLM-generated filename_rules (match_pattern → bids_template).
    2. BIDS entities already embedded in the filename (ses-, task-, acq-, run-).
    3. Heuristic keyword detection in the path.
    4. Extension-based fallback.
    """
    path_lower = filepath.lower()
    filename   = filepath.split("/")[-1]
    fname_low  = filename.lower()

    # ------------------------------------------------------------------
    # Priority 1: LLM filename_rules
    # ------------------------------------------------------------------
    for rule in filename_rules:
        mp = rule.get("match_pattern", "").replace(r"\\", "\\")
        try:
            if not re.search(mp, filename, re.IGNORECASE):
                continue
            template = rule.get("bids_template", "")
            m = re.search(r"sub-[^_]+_(.*?)\.(nii\.gz|snirf|nii)", template)
            if not m:
                continue
            raw = m.group(1)
            # Remove placeholder entities (ses-X, task-X)
            raw = re.sub(r"ses-X_?",  "", raw)
            raw = re.sub(r"task-X_?", "", raw)
            raw = raw.strip("_")
            # Remove spurious ses- if no ses- directory exists in path
            if re.search(r"ses-[A-Za-z0-9]+", raw):
                if not re.search(r"/ses-[A-Za-z0-9]+/", filepath):
                    raw = re.sub(r"ses-[A-Za-z0-9]+_?", "", raw).strip("_")
            if raw:
                subdir = infer_subdirectory_from_suffix(raw)
                return {"suffix": raw, "subdirectory": subdir,
                        "category": categorize_scan_type(raw)}
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Priority 2: Entities already in filename
    # ------------------------------------------------------------------
    entities: Dict[str, str] = {}
    for key, pattern in [("ses",  r"ses-([A-Za-z0-9]+)"),
                          ("task", r"task-([A-Za-z0-9]+)"),
                          ("acq",  r"acq-([A-Za-z0-9]+)"),
                          ("run",  r"run-([A-Za-z0-9]+)")]:
        m = re.search(pattern, filename)
        if m:
            entities[key] = m.group(1)

    # Infer task from filename keywords when no task- entity is present.
    # This handles datasets where files are named by task content rather than
    # BIDS convention (e.g. "2_finger_tapping.snirf", "3_walking.snirf").
    if "task" not in entities:
        fname_no_ext = fname_low.rsplit(".", 1)[0]
        if any(kw in fname_no_ext for kw in ("rest", "resting")):
            entities["task"] = "rest"
        elif any(kw in fname_no_ext for kw in ("finger", "tapping", "fingertap")):
            entities["task"] = "fingertapping"
        elif "walking" in fname_no_ext or "walk" in fname_no_ext:
            entities["task"] = "walking"
        elif any(kw in fname_no_ext for kw in ("motor", "tap")):
            entities["task"] = "motor"

    if fname_low.endswith(".snirf") or "nirs" in fname_low:
        modality_label, subdir = "nirs",          "nirs"
    elif any(k in fname_low for k in ("t1w", "t1")):
        modality_label, subdir = "T1w",           "anat"
    elif any(k in fname_low for k in ("t2w", "t2")):
        modality_label, subdir = "T2w",           "anat"
    elif any(k in fname_low for k in ("bold", "func")):
        modality_label, subdir = "bold",          "func"
    elif "dwi" in fname_low:
        modality_label, subdir = "dwi",           "dwi"
    else:
        modality_label, subdir = None,            "anat"

    if entities or modality_label:
        parts = []
        for key in ("ses", "task", "acq", "run"):
            if key in entities:
                parts.append(f"{key}-{entities[key]}")
        if modality_label:
            parts.append(modality_label)
        if parts:
            suffix = "_".join(parts)
            return {"suffix": suffix, "subdirectory": subdir,
                    "category": categorize_scan_type(suffix)}

    # ------------------------------------------------------------------
    # Priority 3: Heuristic path keywords
    # ------------------------------------------------------------------
    if any(kw in path_lower for kw in ("anat", "mprage", "t1w", "t1 ")):
        return {"suffix": "T1w",            "subdirectory": "anat", "category": "anatomical"}
    if any(kw in path_lower for kw in ("func", "bold")):
        m = re.search(r"task[_-]([A-Za-z0-9]+)", path_lower)
        suffix = f"task-{m.group(1)}_bold" if m else "task-rest_bold"
        return {"suffix": suffix, "subdirectory": "func", "category": "functional"}
    if "rest" in path_lower:
        return {"suffix": "task-rest_bold",  "subdirectory": "func", "category": "functional"}
    if any(kw in path_lower for kw in ("nirs", "fnirs", ".snirf")):
        return {"suffix": "nirs",            "subdirectory": "nirs", "category": "functional"}
    if "dwi" in path_lower:
        return {"suffix": "dwi",             "subdirectory": "dwi",  "category": "diffusion"}

    # ------------------------------------------------------------------
    # Priority 4: Extension fallback
    # ------------------------------------------------------------------
    if fname_low.endswith(".snirf"):
        return {"suffix": "nirs", "subdirectory": "nirs", "category": "functional"}
    if fname_low.endswith((".nii", ".nii.gz")):
        return {"suffix": "T1w", "subdirectory": "anat", "category": "anatomical"}

    return {"suffix": "unknown", "subdirectory": "anat", "category": "unknown"}


def infer_subdirectory_from_suffix(suffix: str) -> str:
    """Map a BIDS suffix string to its subdirectory name."""
    s = suffix.lower()
    if "t1w" in s or "t2w" in s:  return "anat"
    if "bold" in s:                return "func"
    if "nirs" in s:                return "nirs"
    if "dwi"  in s:                return "dwi"
    return "anat"


def categorize_scan_type(suffix: str) -> str:
    """Return a broad category string for a BIDS suffix."""
    s = suffix.lower()
    if "t1w" in s or "t2w" in s:         return "anatomical"
    if "bold" in s or "nirs" in s:       return "functional"
    if "dwi"  in s:                      return "diffusion"
    return "unknown"


# ============================================================================
# Universal filepath analyzer
# ============================================================================

def analyze_filepath_universal(
    filepath: str,
    assignment_rules: List[Dict],
    filename_rules: List[Dict],
    modality: str = "mri",
) -> Dict[str, Any]:
    """
    Determine the BIDS subject ID and output filename for a source file.

    Subject assignment priority:
    1. 'match' glob patterns in assignment_rules.
    2. 'original' substring match.
    3. 'prefix' filename-prefix match.
    4. Standard BIDS sub-XX pattern already in the path.
    5. Fallback: 'unknown'.
    """
    filename   = filepath.split("/")[-1]
    path_parts = filepath.split("/")
    subject_id: Optional[str] = None

    for rule in assignment_rules:
        for pat in rule.get("match", []):
            if _match_glob_pattern(filepath, pat):
                subject_id = rule.get("subject")
                break
        if subject_id:
            break

    if not subject_id:
        for rule in assignment_rules:
            orig = rule.get("original")
            if orig and orig.lower() in filepath.lower():
                subject_id = rule.get("subject")
                break

    if not subject_id:
        for rule in assignment_rules:
            pfx = rule.get("prefix")
            if pfx and filename.lower().startswith(pfx.lower()):
                subject_id = rule.get("subject")
                break

    if not subject_id:
        for part in path_parts:
            m = re.search(r"sub[_-]?(\w+)", part, re.IGNORECASE)
            if m:
                subject_id = m.group(1)
                break

    if not subject_id:
        subject_id = "unknown"

    # Strip accidental 'sub-' prefix from the bare ID
    if subject_id.startswith("sub-"):
        subject_id = subject_id[4:]

    scan_info    = infer_scan_type_from_filepath(filepath, filename_rules)
    ext          = ".snirf" if modality == "nirs" else ".nii.gz"
    bids_filename = f"sub-{subject_id}_{scan_info['suffix']}{ext}"

    return {
        "subject_id":       subject_id,
        "scan_type_suffix": scan_info["suffix"],
        "bids_filename":    bids_filename,
        "subdirectory":     scan_info["subdirectory"],
        "scan_category":    scan_info["category"],
        "original_filepath": filepath,
        "modality":         modality,
    }


# ============================================================================
# Main executor
# ============================================================================

def execute_bids_plan(
    input_root: Path,
    output_dir: Path,
    plan: Dict[str, Any],
    aux_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the BIDS conversion plan produced by the planner stage.

    Conversions performed:
      MRI  : DICOM → NIfTI (dcm2niix), JNIfTI → NIfTI
      fNIRS: .mat / .nirs → SNIRF
      Ready: .nii/.nii.gz (MRI) and .snirf (fNIRS) copied directly.

    Unprocessed files are copied verbatim to bids_compatible/derivatives/.
    """
    info("=== Executing BIDS Plan v10 ===")

    bids_root = Path(output_dir) / "bids_compatible"
    ensure_dir(bids_root)
    ensure_dir(bids_root / "derivatives")

    processed_sources: set = set()
    logs:     List[Dict] = []
    successes = failures = 0

    # ── Step 1: copy trio files ───────────────────────────────────────
    info("\n[1/5] Organizing trio files...")
    for trio_file in ("dataset_description.json", "README.md", "participants.tsv"):
        src = output_dir / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")

    # ── Step 2: process data files ────────────────────────────────────
    info("\n[2/5] Processing data files...")
    all_files_paths = list_all_files(input_root)
    all_files_str   = [
        str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths
    ]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    info(f"Total files: {len(all_files_str)}")

    assignment_rules: List[Dict] = plan.get("assignment_rules", [])
    mappings:         List[Dict] = plan.get("mappings", [])
    info(f"  Assignment rules: {len(assignment_rules)} subjects")

    for mapping in mappings:
        modality      = mapping.get("modality")
        patterns      = mapping.get("match", [])
        filename_rules = mapping.get("filename_rules", [])
        format_ready  = mapping.get("format_ready", False)
        convert_to    = mapping.get("convert_to", "none")
        exclude_pats  = mapping.get("exclude", [])

        info(f"\n  Processing {modality} files...")
        info(f"    Format ready: {format_ready}, Convert to: {convert_to}")

        # Match files (with exclusion)
        matched_files: List[str] = []
        for fp in all_files_str:
            if exclude_pats and any(_match_glob_pattern(fp, ex) for ex in exclude_pats):
                continue
            if any(_match_glob_pattern(fp, pat) for pat in patterns):
                matched_files.append(fp)

        if not matched_files:
            warn(f"    ⚠ No files matched for {modality}")
            continue
        info(f"    ✓ Matched: {len(matched_files)} files")

        # ── fNIRS conversion (direct .mat/.nirs → SNIRF) ──────────────
        if modality == "nirs" and not format_ready:
            info(f"    → fNIRS conversion required ({convert_to})")
            if convert_to != "snirf":
                warn(f"    ⚠ Unknown conversion target: {convert_to}")
                failures += len(matched_files)
                continue

            normalized_path = output_dir / "_staging" / "nirs_headers_normalized.json"
            if normalized_path.exists():
                # CSV → SNIRF via normalized headers
                info("    → Using normalized headers for CSV→SNIRF conversion")
                try:
                    normalized  = read_json(normalized_path)
                    snirf_files = write_snirf_from_normalized(
                        normalized=normalized,
                        input_root=input_root,
                        output_dir=bids_root / "_temp_snirf",
                    )
                    if snirf_files:
                        info(f"    ✓ Generated {len(snirf_files)} SNIRF files from CSV")
                        write_nirs_sidecars(
                            bids_root / "_temp_snirf",
                            {"TaskName": plan.get("task_name", "task"),
                             "SamplingFrequency": 10.0},
                        )
                        successes += len(snirf_files)
                    else:
                        warn("    ✗ CSV→SNIRF conversion produced no files")
                        failures += len(matched_files)
                except Exception as e:
                    warn(f"    ✗ CSV→SNIRF conversion failed: {e}")
                    failures += len(matched_files)
            else:
                # Direct .mat / .nirs → SNIRF
                info("    → Direct file conversion (.mat/.nirs → SNIRF)")
                converted_count = 0
                for fp_str in matched_files:
                    fp = path_str_to_path.get(fp_str)
                    if not fp:
                        continue
                    analysis     = analyze_filepath_universal(fp_str, assignment_rules,
                                                             filename_rules, modality="nirs")
                    dst          = (bids_root / f"sub-{analysis['subject_id']}"
                                    / analysis.get("subdirectory", "nirs")
                                    / analysis["bids_filename"])
                    ensure_dir(dst.parent)
                    ext = fp.suffix.lower()
                    converter = convert_mat_to_snirf if ext == ".mat" else (
                                convert_nirs_to_snirf if ext == ".nirs" else None)
                    if converter:
                        if converter(fp, dst, quiet=False):
                            converted_count += 1
                            successes += 1
                            processed_sources.add(fp_str)
                        else:
                            failures += 1
                    else:
                        warn(f"      ⚠ Unknown fNIRS format: {ext}")
                        failures += 1
                if converted_count:
                    info(f"    ✓ Converted {converted_count} fNIRS files to SNIRF")
                else:
                    warn("    ✗ No fNIRS files were converted")
            continue

        # ── MRI / format-ready: analyze → group → convert / copy ──────
        file_analyses = [
            analyze_filepath_universal(f, assignment_rules, filename_rules,
                                       modality=modality)
            for f in matched_files
        ]

        file_groups: Dict[str, Dict] = {}
        for analysis in file_analyses:
            subj        = analysis["subject_id"]
            scan_suffix = analysis["scan_type_suffix"]
            fp_str      = analysis["original_filepath"]
            is_dicom    = fp_str.lower().endswith(".dcm")

            if is_dicom:
                # Group DICOM files by subject + scan_suffix + normalized filename base.
                # The normalized base separates different body-part series
                # (e.g. VHFCT1mm-Hip vs VHFCT1mm-Head) that share the same suffix (T1w).
                fname_base = _normalize_filename(fp_str)
                group_key  = f"{subj}_{scan_suffix}_{fname_base}"
            else:
                # For non-DICOM files (fNIRS SNIRF, NIfTI), include the
                # filename base in the group key so that multiple files
                # with the same subject+suffix but different names
                # (e.g. different tasks) are kept as separate scan groups.
                fname_base_nir = _normalize_filename(fp_str)
                group_key = f"{subj}_{scan_suffix}_{fname_base_nir}"

            if group_key not in file_groups:
                if is_dicom:
                    fname_base = _normalize_filename(fp_str)

                    # BIDS filename strategy for DICOM series:
                    #
                    # Case A: LLM already provided an acq- label in scan_suffix
                    #         (e.g. scan_suffix = "acq-ankle_T1w" from filename_rules)
                    #         → use scan_suffix directly, do NOT add another acq-
                    #         Result: sub-1_acq-ankle_T1w.nii.gz  ✓
                    #
                    # Case B: LLM gave a generic suffix (e.g. scan_suffix = "T1w")
                    #         with no acq- entity.
                    #         → derive a short acq- label from the filename base
                    #         Result: sub-1_acq-ankle_T1w.nii.gz  ✓
                    #
                    # Previously, executor always added acq-{full_fname_base} regardless,
                    # producing double acq- entities like:
                    #   sub-1_acq-vhfct1mmankle_acq-ankle_T1w.nii.gz  ✗
                    if "acq-" in scan_suffix:
                        # Case A: LLM already set acq-, trust it
                        bids_fname = f"sub-{subj}_{scan_suffix}.nii.gz"
                    else:
                        # Case B: derive a clean, short label from the body-part word
                        acq_label  = _extract_acq_label(fname_base)
                        bids_fname = f"sub-{subj}_acq-{acq_label}_{scan_suffix}.nii.gz"

                    subdir = analysis["subdirectory"]
                else:
                    bids_fname = analysis["bids_filename"]
                    subdir     = analysis["subdirectory"]

                file_groups[group_key] = {
                    "subject_id":    subj,
                    "scan_suffix":   scan_suffix,
                    "bids_filename": bids_fname,
                    "subdirectory":  subdir,
                    "files":         [],
                    "modality":      modality,
                }
            file_groups[group_key]["files"].append(fp_str)

        info(f"    Grouped into {len(file_groups)} scan groups")

        # Deduplicate within each group
        for gdata in file_groups.values():
            if len(gdata["files"]) <= 1:
                continue
            norm_groups: Dict[str, List[str]] = defaultdict(list)
            for f in gdata["files"]:
                norm_groups[_normalize_filename(f)].append(f)
            deduped: List[str] = []
            for norm_files in norm_groups.values():
                parent_dirs = {"/".join(f.split("/")[:-1]) for f in norm_files}
                if len(parent_dirs) > 1:
                    deduped.append(_select_preferred_file(norm_files))
                else:
                    deduped.extend(norm_files)
            gdata["files"] = deduped

        # Subject summary
        subj_groups: Dict[str, int] = {}
        for gd in file_groups.values():
            subj_groups[gd["subject_id"]] = subj_groups.get(gd["subject_id"], 0) + 1
        info(f"    Subjects: {len(subj_groups)}")
        for sid in sorted(subj_groups, key=lambda x: int(x) if x.isdigit() else 0)[:15]:
            info(f"      sub-{sid}: {subj_groups[sid]} scan(s)")

        # Convert / copy each group
        info(f"    Processing {len(file_groups)} scan groups...")
        for gdata in file_groups.values():
            try:
                fp_str   = gdata["files"][0]
                fp       = path_str_to_path.get(fp_str)
                if not fp:
                    failures += 1
                    continue

                subj         = gdata["subject_id"]
                bids_filename = gdata["bids_filename"]
                subdirectory  = gdata["subdirectory"]
                file_ext      = ".nii.gz" if fp.name.lower().endswith(".nii.gz") \
                                else fp.suffix.lower()

                dst = bids_root / f"sub-{subj}" / subdirectory / bids_filename
                ensure_dir(dst.parent)

                done = False

                # JNIfTI → NIfTI
                if file_ext in (".jnii", ".bnii"):
                    if check_jnifti_support():
                        info(f"      → Converting JNIfTI: {fp.name}")
                        if convert_jnifti_to_nifti(fp, dst, quiet=True):
                            successes += 1; done = True
                            processed_sources.add(fp_str)
                        else:
                            warn("      ✗ JNIfTI conversion failed"); failures += 1
                    else:
                        warn("      ⚠ JNIfTI support unavailable (install nibabel)")
                        failures += 1

                # DICOM → NIfTI
                elif file_ext == ".dcm":
                    if check_dcm2niix_available():
                        info(f"      → Converting DICOM batch: {fp.name}")
                        all_dicoms = [
                            path_str_to_path[f] for f in gdata["files"]
                            if path_str_to_path.get(f)
                            and path_str_to_path[f].suffix.lower() == ".dcm"
                        ]
                        if all_dicoms:
                            if run_dcm2niix_batch(all_dicoms, dst, quiet=True):
                                info(f"      ✓ Converted {len(all_dicoms)} DICOM files")
                                successes += 1; done = True
                                for f in gdata["files"]:
                                    processed_sources.add(f)
                            else:
                                warn("      ✗ DICOM conversion failed"); failures += 1
                        else:
                            warn("      ⚠ No DICOM files in group"); failures += 1
                    else:
                        warn("      ⚠ dcm2niix unavailable (install dcm2niix)")
                        failures += 1

                # SNIRF — already BIDS-ready
                elif file_ext == ".snirf" and modality == "nirs":
                    info(f"      → Copying SNIRF: {fp.name}")
                    copy_file(fp, dst)
                    successes += 1; done = True
                    processed_sources.add(fp_str)

                # NIfTI — already BIDS-ready
                # FIX: removed the undefined _write_nifti_sidecar_if_needed() call
                # that was here and would raise NameError at runtime.
                elif file_ext in (".nii", ".nii.gz") and modality == "mri":
                    copy_file(fp, dst)
                    successes += 1; done = True
                    processed_sources.add(fp_str)

                else:
                    warn(f"      ⚠ Unsupported format for {modality}: {file_ext}")
                    failures += 1

                if done:
                    logs.append({
                        "source":      fp_str,
                        "destination": f"sub-{subj}/{subdirectory}/{bids_filename}",
                        "action":      "convert" if file_ext in
                                       (".dcm", ".jnii", ".bnii", ".mat", ".nirs")
                                       else "copy",
                        "modality":    modality,
                        "status":      "success",
                    })

            except Exception as e:
                warn(f"      ✗ Failed: {e}")
                failures += 1

    # ── Step 3: copy unprocessed files to derivatives/ ────────────────
    info("\n[3/5] Copying unprocessed files to derivatives/...")
    derivatives_root = bids_root / "derivatives"
    unprocessed = [f for f in all_files_str if f not in processed_sources]
    info(f"  Total: {len(all_files_str)}, processed: {len(processed_sources)}, "
         f"unprocessed: {len(unprocessed)}")
    copied_deriv = 0
    for fp_str in unprocessed:
        src = path_str_to_path.get(fp_str)
        if src and src.exists():
            try:
                copy_file(src, derivatives_root / fp_str)
                copied_deriv += 1
            except Exception as e:
                warn(f"  Could not copy to derivatives: {fp_str}: {e}")
    info(f"  ✓ Copied {copied_deriv} files to derivatives/")

    # ── Step 4: write logs and manifest ───────────────────────────────
    info("\n[4/5] Finalizing...")
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    manifest_files = sorted(
        str(p.relative_to(bids_root)).replace("\\", "/")
        for p in bids_root.rglob("*") if p.is_file()
    )
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files":       manifest_files,
        "tree":        _build_ascii_tree(bids_root),
    })

    # ── Step 5: summary ───────────────────────────────────────────────
    subject_dirs = list(bids_root.glob("sub-*"))
    info(f"\n[5/5] Summary")
    info("━" * 60)
    info("✓ BIDS Dataset Created")
    info(f"Location:         {bids_root}")
    info(f"Files processed:  {successes}")
    info(f"Failed:           {failures}")
    info(f"Subjects:         {len(subject_dirs)}")

    if subject_dirs:
        info("\nSubject directories:")
        for sd in sorted(subject_dirs)[:15]:
            nii   = len(list(sd.rglob("*.nii.gz")))
            snirf = len(list(sd.rglob("*.snirf")))
            total = nii + snirf
            if nii and snirf:
                info(f"  {sd.name}: {total} files ({nii} NIfTI, {snirf} SNIRF)")
            elif nii:
                info(f"  {sd.name}: {nii} NIfTI file(s)")
            elif snirf:
                info(f"  {sd.name}: {snirf} SNIRF file(s)")
        if len(subject_dirs) > 15:
            info(f"  ... and {len(subject_dirs) - 15} more")
    else:
        warn("⚠ WARNING: No subject directories created!")

    return {
        "total_mappings":        len(mappings),
        "successful_conversions": successes,
        "failed_conversions":    failures,
        "bids_root":             str(bids_root),
        "subject_count":         len(subject_dirs),
    }