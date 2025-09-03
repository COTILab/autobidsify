# -*- coding: utf-8 -*-
"""
Lightweight scanner that walks the prepared input tree, samples representative files,
and extracts just-enough metadata for LLM mapping. It does not convert data.
"""
from pathlib import Path
import hashlib
import json
import csv
import zipfile

TEXT_EXT = {".txt", ".md", ".rtf", ".html", ".htm"}
TABLE_EXT = {".csv", ".tsv", "xlsx"}
DOC_EXT = {".pdf", ".docx", ".pptx"}
EEG_EXT = {".edf", ".bdf", ".vhdr", ".vmrk", ".eeg", ".set", ".fdt"}
MRI_EXT = {".nii", ".dcm"}  # .nii.gz handled specially
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz"}

# NIRS can appear in multiple vendor/proprietary formats
# We include .nirs (vendor), .snirf (BIDS-preferred), .mat (common export), and plain tables.
NIRS_EXT = {".snirf", ".nirs", ".mat"}  # extend later if needed

# Keywords to heuristically classify ambiguous files (like CSV/TSV/TXT/MAT) as NIRS
NIRS_NAME_HINTS = ("nirs", "fnirs", "nirx", "homER", "snirf")

MRI_EXT = {".nii", ".dcm"}  # .nii.gz handled specially
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz"}

USER_TRIO = {"README.md", "participants.tsv", "dataset_description.json"}

def file_hash_head(p: Path, max_bytes: int = 4096) -> str:
    """Return SHA1 of the first max_bytes of the file to fingerprint quickly."""
    h = hashlib.sha1()
    try:
        with p.open("rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return ""

def sample_text_head(p: Path, max_chars: int = 600) -> str:
    """Return the first few characters of a text-like file for summarization."""
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""

def sample_table_head(p: Path, max_rows: int = 3):
    """Return header and first few rows from CSV/TSV to infer columns."""
    try:
        dialect = csv.excel_tab if p.suffix.lower()==".tsv" else csv.excel
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, dialect=dialect)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= max_rows: break
            return {"rows": rows}
    except Exception:
        return {"rows": []}

def detect_kind(p: Path) -> str:
    """Coarse type based on extension; add heuristics so NIRS is recognized even if CSV/MAT."""
    s = p.suffix.lower()
    name_lower = p.name.lower()

    # Trio at root
    if p.name.lower() in USER_TRIO:
        return "user_trio"

    # MRI
    if p.name.lower().endswith(".nii.gz") or s in MRI_EXT:
        return "mri"

    # NIRS by explicit extension
    if s in NIRS_EXT:
        return "nirs"

    # Heuristic: CSV/TSV/TXT/MAT with NIRS-ish names → classify as NIRS
    if s in {".csv", ".tsv", ".txt", ".mat", "xlsx"}:
        # check filename and parent folder names
        parent_chain = [p.parent.name.lower()]
        # also include grandparent one level up as a weak hint
        if p.parent.parent and p.parent.parent != p.parent:
            parent_chain.append(p.parent.parent.name.lower())
        if any(h in name_lower for h in NIRS_NAME_HINTS) or any(
            any(h in x for h in NIRS_NAME_HINTS) for x in parent_chain
        ):
            return "nirs"

    # EEG
    if s in EEG_EXT:
        return "eeg"

    # Tables & text (fallback)
    if s in TABLE_EXT:
        return "table"
    if s in TEXT_EXT:
        return "text"
    if s in DOC_EXT:
        return "doc"
    if s in ARCHIVE_EXT:
        return "archive"
    return "other"

def list_archive(zip_path: Path, max_entries: int = 30):
    """List a few entries inside a .zip archive without fully extracting."""
    meta = {"type":"zip","entries":[]}
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for i, info in enumerate(z.infolist()[:max_entries]):
                meta["entries"].append({"name": info.filename, "size": info.file_size})
    except Exception:
        pass
    return meta

def scan_tree(root: Path, sample_per_ext: int = 5):
    """Walk the tree and collect a compact 'evidence bundle' for the LLM."""
    root = Path(root)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = {}
    for p in all_files:
        key = p.suffix.lower()
        if p.name.lower().endswith(".nii.gz"):
            key = ".nii.gz"
        by_ext.setdefault(key, []).append(p)

    # Representative samples per extension
    samples = []
    for ext, lst in by_ext.items():
        for p in lst[:sample_per_ext]:
            entry = {
                "relpath": str(p.relative_to(root)).replace("\\","/"),
                "size": p.stat().st_size,
                "suffix": ext,
                "kind": detect_kind(p),
                "sha1_head": file_hash_head(p),
            }
            if entry["kind"] == "text":
                entry["text_head"] = sample_text_head(p)
            if entry["kind"] == "table":
                entry["table_head"] = sample_table_head(p)
            if entry["kind"] == "archive" and p.suffix.lower()==".zip":
                entry["archive_listing"] = list_archive(p)
            samples.append(entry)

    # User trio presence (at root)
    trio_found = {name: (root / name).exists() for name in USER_TRIO}

    bundle = {
        "root": str(root),
        "counts": {ext: len(lst) for ext, lst in by_ext.items()},
        "samples": samples,
        "trio_found": trio_found,
    }
    return bundle

def save_evidence_bundle(bundle: dict, out_path: Path):
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

