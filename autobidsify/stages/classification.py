# classification.py
# LLM-based classification for mixed modality datasets.

"""
The classification module is used for intelligent sorting of mixed-modal data.

Features:
1. LLM Classification: Classifies documents based on content, filename, and metadata.
2. Three Output Categories: NIRS files, MRI files, and indeterminate files.
3. Physical Triage: Copies files to separate pool directories.
4. Evidence Tracing: Records classification criteria and confidence levels.

Workflow:
1. Send the evidence bundle to LLM (containing complete document content).
2. LLM analyzes the descriptions in the document (e.g., "fNIRS optodes", "3T MRI scanner").
3. Generate classification_plan.json based on the analysis results.
4. Physically copy files to nirs_pool and mri_pool.
5. Indeterminate files are placed in the unknown_pool and questions are generated.

Classification Strategy:
- Prioritize document content (modalities mentioned in protocol.pdf).
- Secondary use file extensions and paths.
- Finally, use metadata features (e.g., array dimensions).
- Mark uncertain files as unknown and require user intervention.

Output Files:
- classification_plan.json: Classification results and criteria.
- _staging/nirs_pool/: NIRS file pool
- _staging/mri_pool/: MRI file pool
- _staging/unknown/: Unknown files
"""

from pathlib import Path
from typing import Dict, Any, List
import json
from autobidsify.utils import ensure_dir, write_json, copy_file, info, warn, fatal, read_json
from autobidsify.constants import CLASSIFICATION_PLAN, NIRS_POOL, MRI_POOL, UNKNOWN_POOL

def classify_and_stage(model: str, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Classify files into NIRS/MRI/UNKNOWN pools by file extension.
    
    Deterministic classification — no LLM required.
    
    MRI:   .dcm, .nii, .nii.gz, .jnii, .bnii
    fNIRS: .snirf, .nirs, .mat
    Other: unknown pool
    """
    info("Classifying files by extension (deterministic)...")

    MRI_EXTS   = {'.dcm', '.nii', '.nii.gz', '.jnii', '.bnii'}
    NIRS_EXTS  = {'.snirf', '.nirs', '.mat'}

    all_files = bundle.get("all_files", [])

    nirs_files    = []
    mri_files     = []
    unknown_files = []

    for relpath in all_files:
        name_lower = relpath.lower()

        if name_lower.endswith('.nii.gz'):
            ext = '.nii.gz'
        else:
            ext = Path(relpath).suffix.lower()

        if ext in MRI_EXTS:
            mri_files.append(relpath)
        elif ext in NIRS_EXTS:
            nirs_files.append(relpath)
        else:
            unknown_files.append(relpath)

    plan = {
        "nirs_files":    nirs_files,
        "mri_files":     mri_files,
        "unknown_files": unknown_files,
        "classification_method": "extension_based",
        "classification_rationale": {
            "confidence_level": "high",
            "key_evidence_from_documents": [
                f"MRI extensions: {sorted(MRI_EXTS)}",
                f"fNIRS extensions: {sorted(NIRS_EXTS)}",
                f".mat always treated as fNIRS (Homer3/MATLAB fNIRS format)"
            ]
        }
    }

    # Save classification plan
    plan_path = Path(out_dir) / "_staging" / "classification_plan.json"
    write_json(plan_path, plan)
    info(f"✓ Classification plan saved: {plan_path}")
    info(f"  MRI files:   {len(mri_files)}")
    info(f"  fNIRS files: {len(nirs_files)}")
    info(f"  Unknown:     {len(unknown_files)}")

    # Stage files into pools
    _stage_files_to_pools(bundle, plan, out_dir)

    return plan

def _stage_files_to_pools(bundle: Dict[str, Any], plan: Dict[str, Any], out_dir: Path):
    """
    Physically copy files to their respective pools.
    
    Creates directory structure in pools matching original structure.
    """
    root = Path(bundle["root"])
    
    # Create pool directories
    nirs_pool = Path(out_dir) / NIRS_POOL
    mri_pool = Path(out_dir) / MRI_POOL
    unknown_pool = Path(out_dir) / UNKNOWN_POOL
    
    ensure_dir(nirs_pool)
    ensure_dir(mri_pool)
    ensure_dir(unknown_pool)
    
    def stage_file_list(file_list: List[str], pool: Path, label: str):
        """Copy files from list to pool."""
        count = 0
        for relpath in file_list:
            src = root / relpath
            if src.is_file():
                dst = pool / relpath
                ensure_dir(dst.parent)
                copy_file(src, dst)
                count += 1
        
        if count > 0:
            info(f"✓ Staged {count} files to {label} pool")
    
    # Stage each category
    stage_file_list(plan.get("nirs_files", []), nirs_pool, "NIRS")
    stage_file_list(plan.get("mri_files", []), mri_pool, "MRI")
    stage_file_list(plan.get("unknown_files", []), unknown_pool, "UNKNOWN")
    
    # Warn about unknown files
    unknown_count = len(plan.get("unknown_files", []))
    if unknown_count > 0:
        warn(f"{unknown_count} files could not be classified")
        warn(f"Review classification_plan.json and assign manually")

# def classify_files(model: str, output_dir: Path) -> None:
def classify_files(output_dir: Path) -> None:
    """
    High-level classification function called by CLI.
    
    Args:
        model: LLM model name
        output_dir: Output directory (contains _staging/evidence_bundle.json)
    """
    # Load evidence bundle
    bundle_path = Path(output_dir) / "_staging" / "evidence_bundle.json"
    
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        fatal("Run 'evidence' step first")
        return
    
    bundle = read_json(bundle_path)
    
    # Run classification
    plan = classify_and_stage(model, bundle, output_dir)
    
    # Display summary
    nirs_count = len(plan.get("nirs_files", []))
    mri_count = len(plan.get("mri_files", []))
    unknown_count = len(plan.get("unknown_files", []))
    
    info(f"\nClassification summary:")
    info(f"  NIRS files: {nirs_count}")
    info(f"  MRI files: {mri_count}")
    info(f"  Unknown files: {unknown_count}")
