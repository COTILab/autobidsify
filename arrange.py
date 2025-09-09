# -*- coding: utf-8 -*-
"""
Arrange files into a BIDS-like tree focusing on SNIRF, NIfTI, and DICOM.
Rules:
- Keep trio at root (copy or leave as-is if already present)
- SNIRF(.snirf) -> sub-XX/nirs/sub-XX_task-<task>_nirs.snirf  (task defaults to 'rest')
- NIfTI(.nii/.nii.gz):
    * if 4D -> func/  sub-XX_task-<task>_bold.nii.gz
    * if 3D -> anat/  sub-XX_T1w.nii.gz (fallback) or T2w if name/header suggests
- DICOM(.dcm) -> sourcedata/dicom/<original_relpath>
- All other files -> derivatives/_misc/<original_relpath>   (preserve, do not lose)
- Subject assignment: strictly requires --nsubjects
    * If participants.tsv exists at root, read participant_id list to build subjects
    * Otherwise, create subjects '01..N'
    * Then assign discovered SNIRF/NIfTI files to subjects in round-robin order
      (simple, deterministic; you can later replace with clustering/LLM strategies)
Output:
- conversion_log.json (mapping)
- manifest.yaml (full tree and file list) folded into BIDSManifest.yaml
"""
from pathlib import Path
from typing import List, Dict, Tuple
import csv, os
from utils import ensure_dir, copy_file, write_text, write_json, green, yellow
import yaml

try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

TRIO_CANON = {"readme.md", "participants.tsv", "dataset_description.json"}

def _read_participants_ids(bids_root: Path) -> List[str]:
    """Read participants.tsv at root if present; return list of participant_id without 'sub-' prefix."""
    p = bids_root / "participants.tsv"
    if not p.exists():
        return []
    ids = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            r = csv.DictReader(f, delimiter="\t")
            # If no header or missing column, fallback to first column name matching participant_id-ish
            field = None
            fields = [c.strip().lower() for c in r.fieldnames or []]
            if "participant_id" in fields:
                field = r.fieldnames[fields.index("participant_id")]
            elif fields:
                # try to guess
                for cand in r.fieldnames:
                    if cand.lower().startswith("participant"):
                        field = cand; break
            if not field:
                return []
            for row in r:
                vid = (row.get(field) or "").strip()
                if not vid:
                    continue
                # Normalize: accept "sub-01" or "01"
                if vid.lower().startswith("sub-"):
                    vid = vid[4:]
                ids.append(vid.zfill(2))
    except Exception:
        return []
    # de-duplicate preserving order
    seen = set(); out=[]
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _list_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]

def _is_nifti(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")

def _is_snirf(p: Path) -> bool:
    return p.suffix.lower() == ".snirf"

def _is_dicom(p: Path) -> bool:
    return p.suffix.lower() == ".dcm"

def _guess_is_bold(p: Path) -> bool:
    """Use nibabel (if available) or filename hints to guess functional 4D."""
    if _HAS_NIB and _is_nifti(p):
        try:
            img = nib.load(str(p))
            if len(img.shape) >= 4 and (img.shape[3] or 0) > 1:
                return True
        except Exception:
            pass
    # filename fallback
    nm = p.name.lower()
    if "bold" in nm or "func" in nm or "rest" in nm:
        return True
    return False

def _guess_anat_type(p: Path) -> str:
    nm = p.name.lower()
    if "t2" in nm:
        return "T2w"
    return "T1w"

def _build_subjects(n_subjects: int, bids_root: Path) -> List[str]:
    # Prefer participants.tsv if exists and has ids
    ids_from_parts = _read_participants_ids(bids_root)
    if ids_from_parts:
        return ids_from_parts
    return [str(i+1).zfill(2) for i in range(int(n_subjects))]

def _copy_trio_if_present(prepared_root: Path, bids_root: Path):
    """Copy trio from prepared root to BIDS root with canonical lowercase names."""
    for name in TRIO_CANON:
        for cand in [name, name.upper(), name.capitalize()]:
            src = Path(prepared_root) / cand
            if src.exists() and src.is_file():
                dst = Path(bids_root) / name
                copy_file(src, dst)
                break

def _assign_round_robin(files: List[Path], subjects: List[str]) -> Dict[str, str]:
    """
    Return mapping: rel -> subject (round-robin by sorted path).
    """
    mapping = {}
    files_sorted = sorted(files, key=lambda p: str(p).lower())
    if not subjects:
        return mapping
    i = 0
    for p in files_sorted:
        mapping[str(p)] = subjects[i % len(subjects)]
        i += 1
    return mapping

def _build_tree_text(root: Path) -> str:
    lines = [root.name + "/"]
    def walk(d: Path, prefix: str=""):
        items = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, p in enumerate(items):
            last = (i == len(items)-1)
            elbow = "└── " if last else "├── "
            lines.append(prefix + elbow + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir():
                walk(p, prefix + ("    " if last else "│   "))
    walk(root)
    return "\n".join(lines)

def arrange_to_bids(prepared_root: Path, out_dir: Path, n_subjects: int, default_task: str = "rest"):
    """
    Core arrange routine:
    - Copy trio if already present
    - Build subject list
    - Move/copy SNIRF, NIfTI, DICOM, others according to the policy
    """
    prepared_root = Path(prepared_root)
    bids_root = Path(out_dir)
    ensure_dir(bids_root)

    # 1) Copy trio from input root if present (do not overwrite if already exists)
    _copy_trio_if_present(prepared_root, bids_root)

    # 2) Subject list
    subjects = _build_subjects(n_subjects, bids_root)

    # 3) Collect files by modality
    all_files = _list_files(prepared_root)
    rel = lambda p: str(Path(p).relative_to(prepared_root)).replace("\\", "/")
    snirf_files = [p for p in all_files if _is_snirf(p)]
    nifti_files = [p for p in all_files if _is_nifti(p)]
    dicom_files = [p for p in all_files if _is_dicom(p)]
    others = [p for p in all_files if p not in snirf_files+nifti_files+dicom_files and p.name.lower() not in TRIO_CANON]

    # 4) Round-robin assignment for SNIRF and NIfTI
    snirf_assign = _assign_round_robin([rel(p) for p in snirf_files], subjects)
    nifti_assign = _assign_round_robin([rel(p) for p in nifti_files], subjects)

    # 5) Place SNIRF
    planned: Dict[str, str] = {}
    for p in snirf_files:
        rp = rel(p); subj = snirf_assign.get(rp)
        if not subj: continue
        out_rel = f"sub-{subj}/nirs/sub-{subj}_task-{default_task}_nirs.snirf"
        dst = bids_root / out_rel
        copy_file(prepared_root / rp, dst)
        planned[rp] = out_rel

    # 6) Place NIfTI
    for p in nifti_files:
        rp = rel(p); subj = nifti_assign.get(rp)
        if not subj: continue
        if _guess_is_bold(p):
            out_rel = f"sub-{subj}/func/sub-{subj}_task-{default_task}_bold{'.nii.gz' if rp.lower().endswith('.nii.gz') else '.nii'}"
        else:
            wt = _guess_anat_type(p)
            out_rel = f"sub-{subj}/anat/sub-{subj}_{wt}{'.nii.gz' if rp.lower().endswith('.nii.gz') else '.nii'}"
        dst = bids_root / out_rel
        copy_file(prepared_root / rp, dst)
        planned[rp] = out_rel

    # 7) Place DICOM → sourcedata/dicom/keep_relative_path
    for p in dicom_files:
        rp = rel(p)
        dst = bids_root / "sourcedata" / "dicom" / rp
        copy_file(prepared_root / rp, dst)
        planned[rp] = str(Path('sourcedata/dicom') / rp)

    # 8) Others → derivatives/_misc/keep_relative_path
    for p in others:
        rp = rel(p)
        dst = bids_root / "derivatives" / "_misc" / rp
        copy_file(prepared_root / rp, dst)
        planned[rp] = str(Path('derivatives/_misc') / rp)

    # 9) Write logs and manifest
    log = {
        "notes": "Round-robin subject assignment. You can later replace with clustering/LLM.",
        "subjects": [f"sub-{s}" for s in subjects],
        "mapped_count": len(planned),
        "planned": planned
    }
    write_json(bids_root / "conversion_log.json", log)

    # Manifest YAML with tree
    files_out = []
    for p in bids_root.rglob("*"):
        if p.is_file():
            files_out.append(str(p.relative_to(bids_root)).replace("\\", "/"))
    files_out.sort()
    manifest = {
        "file_count": len(files_out),
        "files": files_out,
        "tree_text": _build_tree_text(bids_root),
    }
    write_text(bids_root / "BIDSManifest.yaml", yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    print(green(f"[arrange] placed {len(planned)} files into BIDS-like tree."))
    print(yellow("Note: DICOM kept in sourcedata/dicom/. Others preserved in derivatives/_misc/."))

