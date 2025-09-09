# -*- coding: utf-8 -*-
"""
Generic utilities: filesystem helpers, safe text I/O, simple colored prints.
"""
from pathlib import Path
import shutil
import json
import sys

def ensure_dir(path: Path):
    """Create directory tree if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    """Copy a file preserving metadata; ensure destination directory exists."""
    ensure_dir(Path(dst).parent)
    shutil.copy2(str(src), str(dst))

def move_file(src: Path, dst: Path):
    """Move/rename a file safely; ensure destination directory exists."""
    ensure_dir(Path(dst).parent)
    shutil.move(str(src), str(dst))

def read_text(path: Path, max_chars: int = None) -> str:
    """Read text with UTF-8 and errors ignored; optionally truncate."""
    try:
        s = Path(path).read_text(encoding="utf-8", errors="ignore")
        if max_chars is not None:
            return s[:max_chars]
        return s
    except Exception:
        return ""

def write_text(path: Path, text: str):
    """Write UTF-8 text to a file, creating parents if needed."""
    ensure_dir(Path(path).parent)
    Path(path).write_text(text, encoding="utf-8")

def write_json(path: Path, obj: dict):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# Simple colored output (stdout only when TTY)
def _color(s, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{s}\033[0m"
    return s

def green(s):  return _color(s, "32")
def yellow(s): return _color(s, "33")
def red(s):    return _color(s, "31")

