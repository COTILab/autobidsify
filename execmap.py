# -*- coding: utf-8 -*-
"""
Executor that applies a BIDSMap to an input tree using LLM-provided subject assignment rules.
- Case-insensitive glob matching for both mapping patterns and subject assignment.
- Context variables for templating: filename, parentname, relpath, subject, ext
"""
from pathlib import Path
import yaml, fnmatch, json
from typing import Dict, List, Set, Optional
from rules import fill_template
from utils import ensure_dir, copy_file, yellow, green

USER_TRIO = {"readme.md", "participants.tsv", "dataset_description.json"}

def list_all_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]

def _ext_of(p: Path) -> str:
    """Return normalized extension including dot; handle .nii.gz specially."""
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return p.suffix.lower()  # includes leading dot or empty string

def _subject_from_assignment(rel: str, bidsmap: dict) -> Optional[str]:
    """Resolve subject label for a given relpath using assignment_rules (case-insensitive)."""
    for rule in bidsmap.get("assignment_rules", []):
        subj = str(rule.get("subject", "")).zfill(2)
        for pat in rule.get("match", []):
            if fnmatch.fnmatch(rel.lower(), str(pat).lower()):
                return subj
    return None

def plan_mappings(in_dir: Path, bidsmap: dict) -> Dict[str, str]:
    """Build mapping plan: input relpath -> target BIDS relpath (or None if not BIDS)."""
    planned: Dict[str, str] = {}
    files = list_all_files(in_dir)
    for p in files:
        rel = str(p.relative_to(in_dir)).replace("\\","/")
        if p.name.lower() in USER_TRIO:
            continue
        # subject from LLM assignment (no regex)
        subject = _subject_from_assignment(rel.lower(), bidsmap) or ""
        for m in bidsmap.get("mappings", []):
            for rule in m.get("match", []):
                pattern = str(rule.get("pattern", "**/*"))
                if not fnmatch.fnmatch(rel.lower(), pattern.lower()):
                    continue
                out_tmpl = rule.get("bids_out")
                if not out_tmpl:
                    continue
                ctx = {
                    "filename": p.name,
                    "parentname": p.parent.name,
                    "relpath": rel,
                    "subject": subject,         # may be empty if not assigned
                    "ext": _ext_of(p),          # keep original extension
                }
                target = fill_template(out_tmpl, ctx).strip("/")
                # If template expects {subject} but we don't have one, skip to avoid invalid BIDS
                if ("{subject}" in out_tmpl) and (not subject):
                    continue
                if target:
                    planned[rel] = target
                    break
            else:
                continue
            break
    return planned

def copy_user_trio(in_dir: Path, out_dir: Path):
    """Copy trio files ignoring case; enforce lowercase names in output (BIDS style)."""
    for f in in_dir.iterdir():
        if f.is_file() and f.name.lower() in USER_TRIO:
            dst = out_dir / f.name.lower()
            ensure_dir(dst.parent)
            if not dst.exists():
                copy_file(f, dst)

def route_residuals(in_dir: Path, out_dir: Path, planned: Dict[str, str], bidsmap: dict):
    """Route all non-planned files (except trio) into derivatives/ buckets."""
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
        bucket = "_misc"
        for bname, globs in buckets.items():
            if any(fnmatch.fnmatch(rel.lower(), str(g).lower()) for g in globs):
                bucket = bname
                break
        dst = derivatives_root / bucket / rel
        ensure_dir(dst.parent)
        copy_file(src, dst)

def _generate_tree_text(root: Path) -> str:
    root = Path(root)
    lines: List[str] = [root.name + "/"]
    def walk(d: Path, prefix: str = ""):
        entries = sorted([p for p in d.iterdir()], key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, p in enumerate(entries):
            is_last = (i == len(entries) - 1)
            elbow = "└── " if is_last else "├── "
            lines.append(prefix + elbow + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir():
                extension = "    " if is_last else "│   "
                walk(p, prefix + extension)
    walk(root)
    return "\n".join(lines)

def _build_manifest(out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    files = []
    for p in out_dir.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(out_dir)).replace("\\", "/"))
    files.sort()
    return {"file_count": len(files), "files": files, "tree_text": _generate_tree_text(out_dir)}

def execute_bidsmap(in_dir: Path, out_dir: Path, rules_text: str, dry_run: bool = False):
    """Orchestrate: plan -> copy trio -> place BIDS files -> route residuals -> write logs + manifest."""
    in_dir = Path(in_dir); out_dir = Path(out_dir)
    bidsmap = yaml.safe_load(rules_text)

    plan = plan_mappings(in_dir, bidsmap)
    print(yellow(f"[plan] {len(plan)} files will be placed into BIDS proper."))

    copy_user_trio(in_dir, out_dir)

    for rel, target_rel in plan.items():
        src = in_dir / rel
        dst = out_dir / target_rel
        ensure_dir(dst.parent)
        if not dry_run:
            copy_file(src, dst)

    route_residuals(in_dir, out_dir, plan, bidsmap)

    log = {"mapped_count": len(plan), "planned": plan,
           "notes": "Assignment based on LLM-provided 'assignment_rules' (no regex in executor)."}
    (out_dir / "conversion_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # Manifest YAML
    manifest = _build_manifest(out_dir)
    y = yaml.safe_load(rules_text) or {}
    y["manifest"] = manifest
    (out_dir / "BIDSMap_with_manifest.yaml").write_text(
        yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(green("[manifest] BIDSMap_with_manifest.yaml written."))
    print(green("[done] BIDS tree assembled; residual files are routed to derivatives/."))

