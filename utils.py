# utils.py
# Utility functions for file operations, JSON/YAML I/O, and logging.

"""
中文说明：
通用工具函数模块，提供系统基础功能：

核心功能：
1. 文件操作：读写JSON/YAML/文本文件
2. 目录管理：创建目录、复制文件/目录树
3. 日志输出：带颜色的控制台输出
4. 哈希计算：文件指纹（用于一致性检查）
5. 文件扫描：递归列出所有文件

设计原则：
- 统一的错误处理
- UTF-8编码确保跨平台兼容
- 自动创建父目录
- 清晰的日志输出
"""

import json
import yaml
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import sys

# ANSI color codes for terminal output
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'
COLOR_RESET = '\033[0m'

def fatal(msg: str) -> None:
    """Print fatal error and exit."""
    print(f"{COLOR_RED}[FATAL] {msg}{COLOR_RESET}", file=sys.stderr)
    sys.exit(1)

def warn(msg: str) -> None:
    """Print warning message."""
    print(f"{COLOR_YELLOW}[WARNING] {msg}{COLOR_RESET}", file=sys.stderr)

def info(msg: str) -> None:
    """Print info message."""
    print(f"{COLOR_GREEN}[INFO] {msg}{COLOR_RESET}")

def debug(msg: str) -> None:
    """Print debug message."""
    print(f"{COLOR_BLUE}[DEBUG] {msg}{COLOR_RESET}")

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def write_json(path: Path, data: Any) -> None:
    """Write data to JSON file with UTF-8 encoding."""
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json(path: Path) -> Any:
    """Read JSON file with UTF-8 encoding."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_yaml(path: Path, data: Any) -> None:
    """Write data to YAML file with UTF-8 encoding."""
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def read_yaml(path: Path) -> Any:
    """Read YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_text(path: Path, text: str) -> None:
    """Write text to file with UTF-8 encoding."""
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_text(path: Path) -> str:
    """Read text file with UTF-8 encoding."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def copy_file(src: Path, dst: Path) -> None:
    """Copy file from src to dst, creating parent directories."""
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)

def copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree."""
    shutil.copytree(src, dst, dirs_exist_ok=True)

def list_all_files(root: Path) -> List[Path]:
    """
    Recursively list all files under root.
    
    Skips hidden files (starting with '.').
    """
    files = []
    for p in Path(root).rglob("*"):
        if p.is_file() and not p.name.startswith('.'):
            files.append(p)
    return files

def sha1_head(path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate SHA1 hash of file's first chunk.
    
    Used for quick file fingerprinting without reading entire file.
    """
    h = hashlib.sha1()
    try:
        with open(path, 'rb') as f:
            h.update(f.read(chunk_size))
        return h.hexdigest()[:16]  # Return first 16 chars
    except Exception:
        return "error"

def sha256_full(data: str) -> str:
    """Calculate SHA256 hash of string data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()
