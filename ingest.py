# -*- coding: utf-8 -*-
"""
Ingestion helpers:
- Accept either a folder or a .zip archive as --in
- If it's a .zip, extract to a staging directory under --out (so we never touch the original)
- Return the prepared input root (folder) and a list of original archives (for copying to derivatives later)
"""
from pathlib import Path
import zipfile
from typing import Tuple, List
from utils import ensure_dir, copy_file

def prepare_input_tree(in_path: Path, out_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Prepare the input data for scanning/execution.
    - If 'in_path' is a directory, return it directly (no copy).
    - If 'in_path' is a .zip, extract to <out_dir>/_staging/extracted/ and return that folder.
    - Return a tuple: (prepared_root_folder, [original_archive_paths])
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    staging = out_dir / "_staging"
    ensure_dir(staging)
    if in_path.is_dir():
        return in_path, []
    # Handle .zip archive
    if in_path.is_file() and in_path.suffix.lower() == ".zip":
        extract_root = staging / "extracted"
        ensure_dir(extract_root)
        # Extract the entire archive for simplicity; in very large datasets you may implement partial extraction.
        with zipfile.ZipFile(in_path, "r") as z:
            z.extractall(extract_root)
        return extract_root, [in_path]
    # Unknown input: treat as directory error
    raise ValueError(f"Unsupported --in path: {in_path} (expect a folder or a .zip archive)")

def copy_original_archives_to_derivatives(archives: List[Path], out_dir: Path):
    """
    Copy the original archives into derivatives/orig/archives for provenance.
    """
    bucket = Path(out_dir) / "derivatives" / "orig" / "archives"
    ensure_dir(bucket)
    for a in archives:
        dst = bucket / a.name
        copy_file(a, dst)

