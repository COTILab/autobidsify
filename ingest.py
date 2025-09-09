# -*- coding: utf-8 -*-
"""
Ingestion layer:
- Accept a directory or a .zip as input.
- If .zip, extract it into <out>/_staging/extracted and return that folder.
- Keep the original archive path (for provenance copy to derivatives/orig/archives).
"""
from pathlib import Path
import zipfile
from typing import Tuple, List
from utils import ensure_dir, copy_file

def prepare_input_tree(in_path: Path, out_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Return (prepared_root, [original_archives]).
    - If in_path is a directory: use it as-is.
    - If in_path is a .zip: extract into out/_staging/extracted and return the extracted root.
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    staging = out_dir / "_staging"
    ensure_dir(staging)

    if in_path.is_dir():
        return in_path, []

    if in_path.is_file() and in_path.suffix.lower() == ".zip":
        extract_root = staging / "extracted"
        ensure_dir(extract_root)
        with zipfile.ZipFile(in_path, "r") as z:
            z.extractall(extract_root)
        return extract_root, [in_path]

    raise ValueError(f"Unsupported --in path: {in_path} (expect a folder or a .zip archive)")

def copy_original_archives_to_derivatives(archives: List[Path], out_dir: Path):
    """Copy original input archives into derivatives/orig/archives for provenance."""
    bucket = Path(out_dir) / "derivatives" / "orig" / "archives"
    ensure_dir(bucket)
    for a in archives:
        dst = bucket / a.name
        copy_file(a, dst)

