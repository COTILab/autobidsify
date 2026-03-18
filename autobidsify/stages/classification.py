# classification.py
# Extension-based deterministic classification for mixed-modality datasets.

"""
Classification Module — Extension-Based Deterministic File Triage

This module classifies files into MRI, fNIRS, or unknown pools based solely on file extensions. No LLM is used.

Classification rules:
- MRI   extensions: .dcm, .nii, .nii.gz, .jnii, .bnii
- fNIRS extensions: .snirf, .nirs, .mat
- Unknown: all other files (auxiliary metadata, documents, tables, sidecars, etc.) — sent to the unknown pool by design, NOT because they are unrecognized errors.

Notes on .mat files:
- .mat files are treated as fNIRS (Homer3/MATLAB format) by convention.
- This is correct for standard fNIRS pipelines (Homer3, AtlasViewer, etc.).
- If your .mat files contain MRI array data or other non-fNIRS content, move them out of the dataset or review classification_plan.json manually.

Notes on the unknown pool:
- Auxiliary files (JSON sidecars, TSV tables, PDFs, READMEs, etc.) are intentionally placed in the unknown pool. This is expected behavior, not a classification failure. These files do not carry primary imaging data and are handled separately by the evidence and trio stages.

Workflow:
1. Read all_files list from evidence_bundle.json
2. Classify each file by extension → nirs_files / mri_files / unknown_files
3. Save classification_plan.json
4. Physically copy files to their respective pool directories

Output:
1. _staging/classification_plan.json
2. Pool directories defined by constants: NIRS_POOL, MRI_POOL, UNKNOWN_POOL
"""

from pathlib import Path
from typing import Dict, Any, List

from autobidsify.utils import ensure_dir, write_json, copy_file, info, warn, fatal, read_json
from autobidsify.constants import NIRS_POOL, MRI_POOL, UNKNOWN_POOL, STAGING_DIR

CLASSIFICATION_PLAN_FILENAME = "classification_plan.json"

MRI_EXTS  = {'.dcm', '.nii', '.nii.gz', '.jnii', '.bnii'}
NIRS_EXTS = {'.snirf', '.nirs', '.mat'}


def _detect_extension(relpath: str) -> str:
    """
    Return the normalized file extension for a relative path.

    Handles the compound extension .nii.gz explicitly before falling back
    to pathlib suffix extraction.

    Examples:
        'sub-01/anat/scan.nii.gz' → '.nii.gz'
        'VHMCT1mm-Hip (134).dcm'  → '.dcm'
        'data/signal.mat'         → '.mat'
    """
    if relpath.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return Path(relpath).suffix.lower()


def classify_and_stage(bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Classify files into NIRS/MRI/UNKNOWN pools by file extension.

    Deterministic — no LLM, no document analysis, no semantic reasoning.
    Classification is based entirely on file extension via MRI_EXTS / NIRS_EXTS.

    Args:
        bundle:  Evidence bundle dict (must contain 'root' and 'all_files').
        out_dir: Pipeline output directory.

    Returns:
        Classification plan dict (also written to _staging/classification_plan.json).
    """
    info("Classifying files by extension (deterministic)...")

    # Defensive checks on the evidence bundle
    root_str = bundle.get("root")
    if not root_str:
        fatal("Evidence bundle is missing required field: 'root'. "
              "Re-run the 'evidence' stage to regenerate.")
        return {}

    root = Path(root_str)
    if not root.exists():
        fatal(f"Evidence bundle root does not exist on disk: {root}\n"
              "Check that the original data path is still accessible.")
        return {}

    all_files: List[str] = bundle.get("all_files", [])
    if not all_files:
        warn("Evidence bundle 'all_files' is empty — classification plan will be empty. "
             "Check the input dataset and evidence stage output.")

    # Classify by extension
    nirs_files:    List[str] = []
    mri_files:     List[str] = []
    unknown_files: List[str] = []

    for relpath in all_files:
        ext = _detect_extension(relpath)
        if ext in MRI_EXTS:
            mri_files.append(relpath)
        elif ext in NIRS_EXTS:
            nirs_files.append(relpath)
        else:
            unknown_files.append(relpath)

    # Build and save classification plan
    plan: Dict[str, Any] = {
        "nirs_files":            nirs_files,
        "mri_files":             mri_files,
        "unknown_files":         unknown_files,
        "classification_method": "extension_based",
        "classification_rules": [
            f"MRI extensions:   {sorted(MRI_EXTS)}",
            f"fNIRS extensions: {sorted(NIRS_EXTS)}",
            ".mat is treated as fNIRS (Homer3/MATLAB convention); "
            "review manually if your .mat files are not fNIRS data.",
            "All other extensions go to the unknown pool by design "
            "(auxiliary metadata, sidecars, documents, etc.).",
        ],
        "counts": {
            "mri_files":     len(mri_files),
            "nirs_files":    len(nirs_files),
            "unknown_files": len(unknown_files),
            "all_files":     len(all_files),
        },
    }

    plan_path = Path(out_dir) / STAGING_DIR / CLASSIFICATION_PLAN_FILENAME
    write_json(plan_path, plan)

    info(f"✓ Classification plan saved: {plan_path}")
    info(f"  MRI files:     {len(mri_files)}")
    info(f"  fNIRS files:   {len(nirs_files)}")
    info(f"  Unknown files: {len(unknown_files)} "
         f"(auxiliary/metadata files — expected)")

    # Copy files to pool directories
    _stage_files_to_pools(root, plan, out_dir)

    return plan


def _stage_files_to_pools(root: Path, plan: Dict[str, Any], out_dir: Path) -> None:
    """
    Copy classified files to their respective pool directories.

    Preserves the original relative directory structure inside each pool.
    Logs a warning for any source files that are missing or not regular files
    so that silent data loss is visible.

    Args:
        root:    Absolute path to the data root (validated Path object).
        plan:    Classification plan dict.
        out_dir: Pipeline output directory.
    """
    nirs_pool    = Path(out_dir) / NIRS_POOL
    mri_pool     = Path(out_dir) / MRI_POOL
    unknown_pool = Path(out_dir) / UNKNOWN_POOL

    ensure_dir(nirs_pool)
    ensure_dir(mri_pool)
    ensure_dir(unknown_pool)

    def _copy_list(file_list: List[str], pool: Path, label: str) -> None:
        """Copy files to pool, reporting any that could not be found."""
        copied:  int       = 0
        missing: List[str] = []

        for relpath in file_list:
            src = root / relpath
            if src.is_file():
                dst = pool / relpath
                ensure_dir(dst.parent)
                copy_file(src, dst)
                copied += 1
            else:
                missing.append(relpath)

        if copied > 0:
            info(f"  ✓ Staged {copied} files to {label} pool")

        if missing:
            warn(f"  ⚠ {len(missing)} file(s) not found during {label} staging:")
            for mp in missing[:5]:
                warn(f"      {mp}")
            if len(missing) > 5:
                warn(f"      ... and {len(missing) - 5} more")
                warn("      Check the evidence bundle and on-disk dataset paths.")

    _copy_list(plan.get("nirs_files",    []), nirs_pool,    "NIRS")
    _copy_list(plan.get("mri_files",     []), mri_pool,     "MRI")
    _copy_list(plan.get("unknown_files", []), unknown_pool, "UNKNOWN")

    unknown_count = len(plan.get("unknown_files", []))
    if unknown_count > 0:
        info(f"  Note: {unknown_count} file(s) in unknown pool are auxiliary "
             f"files (metadata, sidecars, docs) — this is expected behavior.")


def classify_files(output_dir: Path) -> None:
    """
    Top-level classification entry point called by the CLI ('autobidsify classify').

    Reads the evidence bundle from _staging/evidence_bundle.json,
    runs extension-based classification, and copies files to pool
    directories. No LLM or model parameter is required.

    Args:
        output_dir: Pipeline output directory.
                    Must contain _staging/evidence_bundle.json
                    (generated by the 'evidence' stage).
    """
    bundle_path = Path(output_dir) / STAGING_DIR / "evidence_bundle.json"

    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}\n"
              "Run 'autobidsify evidence' first.")
        return

    bundle = read_json(bundle_path)
    plan   = classify_and_stage(bundle, output_dir)

    info(f"\nClassification summary:")
    info(f"  MRI files:     {len(plan.get('mri_files',     []))}")
    info(f"  fNIRS files:   {len(plan.get('nirs_files',    []))}")
    info(f"  Unknown files: {len(plan.get('unknown_files', []))} "
         f"(auxiliary files, expected)")