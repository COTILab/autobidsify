# -*- coding: utf-8 -*-
"""
BIDSMap (YAML rules) helpers and a tiny expression engine for templating.
"""
from pathlib import Path
import yaml
import re

# --------------------------
# I/O helpers
# --------------------------
def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def load_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def default_bidsmap_stub() -> str:
    """A minimal fallback BIDSMap if LLM is unavailable."""
    data = {
        "bids_version": "1.10.0",
        "policy": {
            "keep_user_trio": True,
            "route_residuals_to_derivatives": True,
            "derivatives_buckets": {
                "orig": ["**/*"],
                "docs": ["**/*.pdf","**/*.html","**/*.htm","**/*.md"],
                "intermediate": ["**/*.log","**/*.tmp","**/*.bak"],
                "_misc": ["**/*"]
            }
        },
        "entities": {
            "subject": {"pad": 2, "sanitize": True},
            "session": {"omit_if_single": True},
            "run": {"omit_if_single": True}
        },
        "mappings": [],
        "participants": {
            "respect_user_file": True,
            "fallback_generate_if_missing": False
        },
        "questions": ["Please confirm task labels and dataset license if applicable."]
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

# --------------------------
# Minimal expression engine
# --------------------------
def _apply_pipe(value: str, pipe: str) -> str:
    """Apply a single pipe (regex, pad2, sanitize, or:default, lower, upper)."""
    if pipe.startswith("regex:"):
        # Accept regex:'pattern',N or regex:"pattern",N
        m = re.match(r"regex:(?:'|\")(.+?)(?:'|\"),([0-9]+)$", pipe)
        if m:
            pat = m.group(1)
            gi = int(m.group(2))
            m2 = re.search(pat, value or "")
            return m2.group(gi) if m2 else ""
    elif pipe == "pad2":
        return (value or "").zfill(2)
    elif pipe == "sanitize":
        return re.sub(r"[^A-Za-z0-9]+", "", value or "")
    elif pipe.startswith("or:"):
        return value if value else pipe.split(":",1)[1]
    elif pipe == "lower":
        return (value or "").lower()
    elif pipe == "upper":
        return (value or "").upper()
    return value

def eval_expr(expr: str, context: dict) -> str:
    """
    Evaluate an expression like: {filename|regex:'sub-(\\d+)',1|pad2|or:'01'}
    Recognized context fields: filename, parentname, relpath.
    """
    body = expr.strip()[1:-1].strip()
    parts = [p.strip() for p in body.split("|")]
    if not parts: return ""
    source = parts[0]
    val = context.get(source, "")
    for pipe in parts[1:]:
        val = _apply_pipe(val, pipe)
    return val

def fill_template(template: str, context: dict) -> str:
    """Replace {expr} blocks in a template using eval_expr."""
    def repl(m):
        return eval_expr(m.group(0), context) or ""
    return re.sub(r"\{[^{}]+\}", repl, template)

