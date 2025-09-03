# -*- coding: utf-8 -*-
"""
Executor that applies a BIDSMap to an input tree:
- Matches files by glob patterns
- Builds target BIDS paths via small templating engine
- Copies or "converts" (stubbed) files
- Routes all residual files into derivatives, except the user trio which remain at BIDS root
"""
from pathlib import Path
import yaml
import fnmatch
import json
from typing import Dict, List, Set
from rules import fill_template
from utils import ensure_dir, copy_file, yellow, green

USER_TRIO = {"readme.md", "participants.tsv", "dataset_description.json"}

def list_all_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]

def plan_mappings(in_dir: Path, bidsmap: dict) -> Dict[str, str]:
    """
    Build a mapping plan: input relpath -> target BIDS relpath (or None if not BIDS).
    The executor only plans BIDS targets here; routing to derivatives is handled later.
    """
    planned: Dict[str, str] = {}
    files = list_all_files(in_dir)
    for p in files:
        rel = str(p.relative_to(in_dir)).replace("\\","/")
        if p.name.lower() in USER_TRIO:
            continue
        for m in bidsmap.get("mappings", []):
            for rule in m.get("match", []):
                pattern = rule.get("pattern", "**/*")
                if not fnmatch.fnmatch(rel, pattern):
                    continue
                out_tmpl = rule.get("bids_out")
                if not out_tmpl:
                    continue
                ctx = {
                    "filename": p.name,
                    "parentname": p.parent.name,
                    "relpath": rel
                }
                target = fill_template(out_tmpl, ctx).strip("/")
                if target:
                    planned[rel] = target
                    break
            else:
                continue
            break
    return planned

def copy_user_trio(in_dir: Path, out_dir: Path):
    """Copy user-provided trio to BIDS root if present (do not overwrite if already exists)."""
    for name in USER_TRIO:
        src = in_dir / name
        if src.exists():
            dst = out_dir / name
            ensure_dir(dst.parent)
            if not dst.exists():
                copy_file(src, dst)

def route_residuals(in_dir: Path, out_dir: Path, planned: Dict[str, str], bidsmap: dict):
    """
    Place all files NOT planned for BIDS (and not the user trio) into derivatives/.
    Create buckets under derivatives (orig/docs/intermediate/_misc) based on simple globs.
    """
    policy = bidsmap.get("policy", {})
    if not bool(policy.get("route_residuals_to_derivatives", True)):
        return
    buckets = policy.get("derivatives_buckets", {})
    derivatives_root = out_dir / "derivatives"
    ensure_dir(derivatives_root)

    files = list_all_files(in_dir)
    mapped_set: Set[str] = set(planned.keys())
    for src in files:
        rel = str(src.relative_to(in_dir)).replace("\\","/")
        if src.name.lower() in USER_TRIO or rel in mapped_set:
            continue
        # find first matching bucket
        bucket = "_misc"
        for bname, globs in buckets.items():
            if any(fnmatch.fnmatch(rel, g) for g in globs):
                bucket = bname
                break
        dst = derivatives_root / bucket / rel
        ensure_dir(dst.parent)
        copy_file(src, dst)

def execute_bidsmap(in_dir: Path, out_dir: Path, rules_text: str, dry_run: bool = False):
    """
    Orchestrate the execution:
    - Build mapping plan
    - Copy user trio
    - Copy/convert mapped files into BIDS tree
    - Route all residual files into derivatives/
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    bidsmap = yaml.safe_load(rules_text)

    # 1) Build the mapping plan
    plan = plan_mappings(in_dir, bidsmap)
    print(yellow(f"[plan] {len(plan)} files will be placed into BIDS proper."))

    # 2) Copy trio
    copy_user_trio(in_dir, out_dir)

    # 3) Apply the plan (copy/rename; real converters can be plugged later)
    for rel, target_rel in plan.items():
        src = in_dir / rel
        dst = out_dir / target_rel
        ensure_dir(dst.parent)
        if not dry_run:
            copy_file(src, dst)

    # 4) Route residual files into derivatives
    route_residuals(in_dir, out_dir, plan, bidsmap)

    # 5) Write a conversion log
    log = {
        "mapped_count": len(plan),
        "planned": plan,
        "notes": "Demo executor; plug real conversions (e.g., DICOM->NIfTI via dcm2niix) as needed."
    }
    (out_dir / "conversion_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(green("[done] BIDS tree assembled; residual files are routed to derivatives/."))

