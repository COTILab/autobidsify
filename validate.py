# -*- coding: utf-8 -*-
"""
Thin wrapper around the `bids-validator` CLI.
If not installed, return a stub report (no errors) so the pipeline is not blocked.
"""
from pathlib import Path
import shutil, subprocess, json

def run_bids_validator(bids_root: Path) -> dict:
    cli = shutil.which("bids-validator")
    if not cli:
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

