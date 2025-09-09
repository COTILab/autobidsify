# -*- coding: utf-8 -*-
"""
Scan the prepared input tree and build an 'evidence bundle' for the LLM:
- Full file list and extension counts
- Light-weight content samples (text/CSV/JSON)
- SNIRF/NIfTI/DICOM headers (metadata only; no pixels/voxels)
- Trio presence and contents if already provided at root
- User description text and required number of subjects
"""
from pathlib import Path
from typing import Dict, List, Any
import os, csv, json, hashlib

from utils import read_text, write_text

# Optional heavy deps (fail-soft)
try:
    import h5py  # for SNIRF
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

try:
    import nibabel as nib  # for NIfTI
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

try:
    import pydicom  # for DICOM
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

TEXT_EXT = {".txt", ".md", ".rtf", ".html", ".htm", ".log"}
TABLE_EXT = {".csv", ".tsv", ".json", ".yaml", ".yml"}
NIFTI_EXT = {".nii", ".nii.gz"}
SNIRF_EXT = {".snirf"}
DICOM_EXT = {".dcm"}  # note: some dicoms have no extension; we avoid deep sniffing for speed

TRIO_CANON = {"readme.md", "participants.tsv", "dataset_description.json"}

def _sha1_head(p: Path, max_bytes: int = 4096) -> str:
    h = hashlib.sha1()
    try:
        with p.open("rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return ""

def _sample_text(p: Path, max_chars: int = 2000) -> str:
    return read_text(p, max_chars=max_chars)

def _sample_csv_tsv(p: Path, max_rows: int = 5) -> Dict[str, Any]:
    info = {"path": str(p), "head": [], "dialect": "csv"}
    try:
        is_tsv = p.suffix.lower() == ".tsv"
        dialect = csv.excel_tab if is_tsv else csv.excel
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f, dialect=dialect)
            for i, row in enumerate(r):
                info["head"].append(row)
                if i >= max_rows: break
        info["dialect"] = "tsv" if is_tsv else "csv"
    except Exception:
        pass
    return info

def _read_json_yaml_head(p: Path, max_chars: int = 4000) -> Dict[str, Any]:
    # Just show prefix text; avoid parsing large files for robustness
    return {"path": str(p), "text": read_text(p, max_chars=max_chars)}

def _nifti_header(p: Path) -> Dict[str, Any]:
    meta = {"path": str(p), "ok": False}
    try:
        if not _HAS_NIB:
            meta["reason"] = "nibabel not installed"
            return meta
        img = nib.load(str(p))
        hdr = img.header
        meta.update({
            "shape": tuple(int(x) for x in img.shape),
            "zooms": tuple(float(x) for x in getattr(img, "header", {}).get_zooms()[:len(img.shape)]),
            "datatype": int(hdr.get_data_dtype().num),
            "descrip": str(hdr.get("descrip", b"")).strip(),
        })
        meta["ok"] = True
    except Exception as e:
        meta["reason"] = f"{type(e).__name__}: {e}"
    return meta

def _snirf_header(p: Path) -> Dict[str, Any]:
    meta = {"path": str(p), "ok": False}
    try:
        if not _HAS_H5PY:
            meta["reason"] = "h5py not installed"
            return meta
        with h5py.File(str(p), "r") as f:
            # Read a few common fields if present
            meta["nirs_meta"] = {}
            for key in ["metaDataTags", "nirs", "probe"]:
                if key in f:
                    meta["nirs_meta"][key] = list(f[key].keys()) if hasattr(f[key], "keys") else str(type(f[key]))
            # Try typical subject ID locations
            sid = None
            try:
                sid = f["nirs"]["metaDataTags"]["SubjectID"][()]
                if isinstance(sid, bytes): sid = sid.decode("utf-8", "ignore")
            except Exception:
                pass
            if sid: meta["subjectID"] = str(sid)
            meta["ok"] = True
    except Exception as e:
        meta["reason"] = f"{type(e).__name__}: {e}"
    return meta

def _dicom_tags(p: Path) -> Dict[str, Any]:
    meta = {"path": str(p), "ok": False}
    try:
        if not _HAS_PYDICOM:
            meta["reason"] = "pydicom not installed"
            return meta
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        # Redact obvious PHI fields; include protocol/sequence hints
        safe = {}
        want = [
            ("PatientID", "PatientID"),
            ("StudyDescription", "StudyDescription"),
            ("SeriesDescription", "SeriesDescription"),
            ("ProtocolName", "ProtocolName"),
            ("Manufacturer", "Manufacturer"),
            ("Modality", "Modality"),
            ("EchoTime", "EchoTime"),
            ("RepetitionTime", "RepetitionTime"),
            ("FlipAngle", "FlipAngle"),
        ]
        for tag_name, key in want:
            val = getattr(ds, tag_name, None)
            if val is not None:
                safe[key] = str(val)
        meta["tags"] = safe
        meta["ok"] = True
    except Exception as e:
        meta["reason"] = f"{type(e).__name__}: {e}"
    return meta

def scan_tree(prepared_root: Path, user_text: str, n_subjects: int, sample_limit_per_ext: int = 6) -> Dict[str, Any]:
    root = Path(prepared_root)
    all_files = [p for p in root.rglob("*") if p.is_file()]

    counts = {}
    by_ext: Dict[str, List[Path]] = {}
    for p in all_files:
        ext = p.suffix.lower()
        if p.name.lower().endswith(".nii.gz"):
            ext = ".nii.gz"
        counts[ext] = counts.get(ext, 0) + 1
        by_ext.setdefault(ext, []).append(p)

    # Trio presence at ROOT (case-insensitive)
    trio_found = {name: (root / name).exists() for name in TRIO_CANON}

    samples: Dict[str, Any] = {
        "text": [], "tables": [], "json_like": [],
        "snirf_headers": [], "nifti_headers": [], "dicom_tags": []
    }

    # Collect light samples per extension
    for ext, files in by_ext.items():
        for p in files[:sample_limit_per_ext]:
            low = ext.lower()
            if low in TEXT_EXT:
                samples["text"].append({"path": str(p.relative_to(root)), "snippet": _sample_text(p, 1500)})
            elif low in {".csv", ".tsv"}:
                info = _sample_csv_tsv(p)
                info["path"] = str(p.relative_to(root))
                samples["tables"].append(info)
            elif low in {".json", ".yaml", ".yml", ".html", ".htm"}:
                j = _read_json_yaml_head(p, max_chars=2000)
                j["path"] = str(p.relative_to(root))
                samples["json_like"].append(j)
            elif low in SNIRF_EXT:
                h = _snirf_header(p)
                h["relpath"] = str(p.relative_to(root))
                samples["snirf_headers"].append(h)
            elif low in NIFTI_EXT:
                h = _nifti_header(p)
                h["relpath"] = str(p.relative_to(root))
                samples["nifti_headers"].append(h)
            elif low in DICOM_EXT:
                h = _dicom_tags(p)
                h["relpath"] = str(p.relative_to(root))
                samples["dicom_tags"].append(h)
            else:
                # nothing for other binaries; we still keep counts
                pass

    # Read trio contents if present at root
    trio_texts = {}
    for name in TRIO_CANON:
        p = root / name
        if p.exists() and p.is_file():
            trio_texts[name] = read_text(p, max_chars=10000)

    bundle = {
        "root": str(root),
        "all_files": [str(p.relative_to(root)).replace("\\", "/") for p in all_files],
        "counts": counts,
        "samples": samples,
        "trio_found": trio_found,
        "trio_texts": trio_texts,
        "user_hints": {"n_subjects": int(n_subjects)},
        "user_text": str(user_text or "").strip(),
    }
    return bundle

def save_evidence_bundle(bundle: dict, out_path: Path):
    write_text(out_path, json.dumps(bundle, indent=2, ensure_ascii=False))

