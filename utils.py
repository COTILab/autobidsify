# -*- coding: utf-8 -*-
"""
General helpers for file operations and terminal output.
"""
from pathlib import Path
import shutil

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    """Copy file with dirs ensured; overwrite if exists."""
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def move_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))

# Simple colored output
def color(s, code): return f"\033[{code}m{s}\033[0m"
def green(s): return color(s, "32")
def yellow(s): return color(s, "33")
def red(s): return color(s, "31")

