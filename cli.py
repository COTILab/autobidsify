#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry point for the auto_bidsify pipeline.

Key update:
- New flag --nsubjects to let the user specify the total number of subjects.
  This hint is stored in evidence_bundle.json and consumed by the LLM to
  create assignment_rules (subject -> file patterns) without relying on regex.
"""
import argparse, json
from pathlib import Path
from ingest import prepare_input_tree, copy_original_archives_to_derivatives
from scan import scan_tree, save_evidence_bundle
from rules import save_text, load_text
from llm_openai import generate_bidsmap_with_openai_or_stub
from execmap import execute_bidsmap
from validate import run_bids_validator, summarize_issues
from utils import green, yellow, red, ensure_dir

def cmd_scan(args):
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)
    prepared_root, archives = prepare_input_tree(in_path, out_dir)
    # Pass nsubjects hint into the evidence bundle
    nsubjects = args.nsubjects if getattr(args, "nsubjects", None) else None
    bundle = scan_tree(prepared_root, n_subjects=nsubjects)
    bundle_path = out_dir / "evidence_bundle.json"
    save_evidence_bundle(bundle, bundle_path)
    info = {"prepared_root": str(prepared_root), "original_archives": [str(a) for a in archives]}
    (out_dir / "pipeline_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(green(f"Evidence bundle saved: {bundle_path}"))
    if archives:
        print(yellow(f"Detected {len(archives)} archive(s). They will be copied into derivatives/orig/archives in exec step."))

def cmd_map(args):
    evidence_path = Path(args.evidence).resolve()
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)
    evidence = json.loads(load_text(evidence_path))
    # If --nsubjects is provided at map-time, override/add it in evidence
    if getattr(args, "nsubjects", None):
        evidence.setdefault("user_hints", {})["n_subjects"] = int(args.nsubjects)
    rules_text = generate_bidsmap_with_openai_or_stub(
        evidence=evidence,
        llm_backend=args.llm,
        model=args.model,
        prefer_user_trio=True
    )
    bidsmap_path = out_dir / "BIDSMap.yaml"
    save_text(bidsmap_path, rules_text)
    print(green(f"BIDSMap saved: {bidsmap_path}"))

def cmd_exec(args):
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)
    info_path = out_dir / "pipeline_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        prepared_root = Path(info["prepared_root"])
        archives = [Path(x) for x in info.get("original_archives", [])]
    else:
        prepared_root, archives = prepare_input_tree(in_path, out_dir)
    rules_path = Path(args.rules).resolve()
    rules_text = load_text(rules_path)
    execute_bidsmap(prepared_root, out_dir, rules_text, dry_run=args.dry_run)
    if archives:
        copy_original_archives_to_derivatives(archives, out_dir)
        print(yellow("Original archive(s) copied into derivatives/orig/archives."))
    print(green("Execution completed."))

def cmd_validate(args):
    out_dir = Path(args.output).resolve()
    report = run_bids_validator(out_dir)
    issues = summarize_issues(report)
    print(yellow("Validator raw report (truncated fields hidden):"))
    import json as _json
    print(_json.dumps(report, indent=2)[:12000])
    if any(i['severity']=="error" for i in issues):
        print(red("Validation found errors."))
    else:
        print(green("Validation passed without errors."))

def cmd_full(args):
    cmd_scan(args)
    evidence_path = Path(args.output) / "evidence_bundle.json"
    margs = argparse.Namespace(
        evidence=str(evidence_path),
        output=args.output,
        llm=args.llm,
        model=args.model,
        nsubjects=args.nsubjects
    )
    cmd_map(margs)
    rules_path = Path(args.output) / "BIDSMap.yaml"
    eargs = argparse.Namespace(input=args.input, output=args.output, rules=str(rules_path), dry_run=False)
    cmd_exec(eargs)
    cmd_validate(args)

def main():
    ap = argparse.ArgumentParser(prog="auto_bidsify_v3")
    sub = ap.add_subparsers(dest="cmd")
    # Manual required for Python 3.6 compatibility
    # (Argparse required=True for subparsers is 3.7+)
    def _require_cmd(a):
        if a.cmd is None:
            ap.print_help(); exit(1)

    ap_scan = sub.add_parser("scan", help="Scan input tree (or zip) and build evidence bundle")
    ap_scan.add_argument("--in", dest="input", required=True)
    ap_scan.add_argument("--out", dest="output", required=True)
    ap_scan.add_argument("--nsubjects", type=int, help="Total number of subjects to expect (hint for LLM)")
    ap_scan.set_defaults(func=cmd_scan)

    ap_map = sub.add_parser("map", help="Generate BIDSMap (YAML) via ChatGPT or stub")
    ap_map.add_argument("--evidence", required=True)
    ap_map.add_argument("--out", dest="output", required=True)
    ap_map.add_argument("--llm", choices=["openai","stub"], default="openai")
    ap_map.add_argument("--model", default="gpt-5-mini")
    ap_map.add_argument("--nsubjects", type=int, help="Override/define subject count for LLM")
    ap_map.set_defaults(func=cmd_map)

    ap_exec = sub.add_parser("exec", help="Execute BIDSMap and route residual files to derivatives")
    ap_exec.add_argument("--in", dest="input", required=True)
    ap_exec.add_argument("--out", dest="output", required=True)
    ap_exec.add_argument("--rules", required=True)
    ap_exec.add_argument("--dry-run", action="store_true")
    ap_exec.set_defaults(func=cmd_exec)

    ap_val = sub.add_parser("validate", help="Run bids-validator if available (stub otherwise)")
    ap_val.add_argument("--out", dest="output", required=True)
    ap_val.set_defaults(func=cmd_validate)

    ap_full = sub.add_parser("full", help="Run full pipeline: scan -> map -> exec -> validate")
    ap_full.add_argument("--in", dest="input", required=True)
    ap_full.add_argument("--out", dest="output", required=True)
    ap_full.add_argument("--llm", choices=["openai","stub"], default="openai")
    ap_full.add_argument("--model", default="gpt-5-mini")
    ap_full.add_argument("--nsubjects", type=int, help="Total number of subjects to expect (hint for LLM)")
    ap_full.set_defaults(func=cmd_full)

    args = ap.parse_args()
    _require_cmd(args)
    args.func(args)

if __name__ == "__main__":
    main()

