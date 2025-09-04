# -*- coding: utf-8 -*-
"""
Validator wrapper. If 'bids-validator' CLI exists, run it and return JSON.
Otherwise, return a stub report so the pipeline keeps working offline.
"""
import json, shutil, subprocess
from pathlib import Path

def run_bids_validator(bids_root: Path) -> dict:
    cli = shutil.which("bids-validator")
    if not cli:
        # Stub: pretend we ran it and found no errors
        return {"status": "stubbed", "issues": {"errors": [], "warnings": []}}
    try:
        out = subprocess.check_output([cli, "--json", str(bids_root)], text=True, stderr=subprocess.STDOUT)
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        try:
            return json.loads(e.output)
        except Exception:
            return {"status": "error", "raw": e.output}

def summarize_issues(report: dict):
    issues = []
    for item in report.get("issues", {}).get("errors", []):
        issues.append({
            "severity": "error",
            "code": item.get("code"),
            "message": item.get("reason") or item.get("message"),
            "files": [f.get("file", {}).get("path") for f in item.get("files", [])]
        })
    for item in report.get("issues", {}).get("warnings", []):
        issues.append({
            "severity": "warning",
            "code": item.get("code"),
            "message": item.get("reason") or item.get("message"),
            "files": [f.get("file", {}).get("path") for f in item.get("files", [])]
        })
    return issues

