# -*- coding: utf-8 -*-
"""
Command-line entry point for the new auto_bidsify pipeline.

Hard requirements:
- User must provide a descriptive text (via --user-text or --user-text-file)
- User must provide --nsubjects (integer > 0)

Steps:
- scan: build evidence bundle (reads trio contents if exist)
- trio: generate missing dataset_description.json / README.md / participants.tsv via LLM or stub
- arrange: place files into a BIDS-like tree (SNIRF/NIfTI/DICOM focus; others preserved)
- validate: run bids-validator if available
- full: run all steps in order
"""
import argparse, json
from pathlib import Path
from ingest import prepare_input_tree, copy_original_archives_to_derivatives
from scan import scan_tree, save_evidence_bundle
from llm_openai import generate_trio_if_missing
from arrange import arrange_to_bids
from validate import run_bids_validator, summarize_issues
from utils import ensure_dir, read_text, write_text, green, yellow, red

def _read_user_text(args) -> str:
    if args.user_text_file:
        p = Path(args.user_text_file)
        if not p.exists():
            raise SystemExit(f"--user-text-file not found: {p}")
        return read_text(p, max_chars=20000)
    if args.user_text:
        return args.user_text.strip()
    raise SystemExit("--user-text or --user-text-file is required.")

def cmd_scan(args):
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    ensure_dir(out_dir)

    # Prepare input
    prepared_root, archives = prepare_input_tree(in_path, out_dir)
    # Read user text and nsubjects (hard requirements)
    user_text = _read_user_text(args)
    if args.nsubjects is None or int(args.nsubjects) <= 0:
        raise SystemExit("--nsubjects must be a positive integer.")

    bundle = scan_tree(prepared_root, user_text=user_text, n_subjects=int(args.nsubjects))
    bundle_path = out_dir / "evidence_bundle.json"
    save_evidence_bundle(bundle, bundle_path)

    # Persist pipeline info
    info = {"prepared_root": str(prepared_root), "original_archives": [str(a) for a in archives]}
    write_text(out_dir / "pipeline_info.json", json.dumps(info, indent=2, ensure_ascii=False))

    print(green(f"[scan] evidence bundle saved: {bundle_path}"))
    if archives:
        print(yellow(f"[scan] detected {len(archives)} archive(s). They will be copied later to derivatives/orig/archives."))

def cmd_trio(args):
    out_dir = Path(args.output).resolve()
    bundle_path = out_dir / "evidence_bundle.json"
    if not bundle_path.exists():
        raise SystemExit("evidence_bundle.json not found; run 'scan' first.")
    evidence = json.loads(read_text(bundle_path, max_chars=None))
    warnings = generate_trio_if_missing(evidence, out_dir=str(out_dir), model=args.model)
    if warnings:
        print(yellow("[trio] warnings:"))
        for w in warnings:
            print(yellow(f" - {w}"))
    else:
        print(green("[trio] trio files existed or were generated successfully."))

def cmd_arrange(args):
    out_dir = Path(args.output).resolve()
    info_path = out_dir / "pipeline_info.json"
    if not info_path.exists():
        raise SystemExit("pipeline_info.json not found; run 'scan' first.")
    info = json.loads(read_text(info_path, max_chars=None))
    prepared_root = Path(info["prepared_root"])
    nsubjects = args.nsubjects
    if nsubjects is None or int(nsubjects) <= 0:
        raise SystemExit("--nsubjects must be provided again for arrange step.")

    arrange_to_bids(prepared_root=prepared_root, out_dir=out_dir, n_subjects=int(nsubjects), default_task=args.task)

    # Copy original archive(s) into derivatives/orig/archives
    archives = [Path(x) for x in info.get("original_archives", [])]
    if archives:
        copy_original_archives_to_derivatives(archives, out_dir)
        print(yellow("[arrange] original archive(s) copied to derivatives/orig/archives."))

def cmd_validate(args):
    out_dir = Path(args.output).resolve()
    report = run_bids_validator(out_dir)
    issues = summarize_issues(report)
    print(yellow("[validate] raw report:"))
    print(json.dumps(report, indent=2)[:12000])
    if any(i["severity"] == "error" for i in issues):
        print(red("[validate] Validation found errors."))
    else:
        print(green("[validate] Validation passed without errors (or stubbed)."))

def cmd_full(args):
    # 1) scan
    cmd_scan(args)
    # 2) trio (LLM/stub) to generate missing trio files
    margs = argparse.Namespace(output=args.output, model=args.model)
    cmd_trio(margs)
    # 3) arrange
    aargs = argparse.Namespace(output=args.output, nsubjects=args.nsubjects, task=args.task)
    cmd_arrange(aargs)
    # 4) validate
    cmd_validate(args)

def main():
    ap = argparse.ArgumentParser(prog="auto_bidsify_new")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common flags
    ap.add_argument("--user-text", help="Free-form dataset description (string).", default=None)
    ap.add_argument("--user-text-file", help="Path to a text file containing dataset description.", default=None)
    ap.add_argument("--nsubjects", type=int, help="Total number of subjects (required).", default=None)

    # scan
    ap_scan = sub.add_parser("scan", help="Scan input and build evidence bundle.")
    ap_scan.add_argument("--in", dest="input", required=True)
    ap_scan.add_argument("--out", dest="output", required=True)
    ap_scan.set_defaults(func=cmd_scan)

    # trio
    ap_trio = sub.add_parser("trio", help="Generate missing dataset_description/README/participants (LLM or stub).")
    ap_trio.add_argument("--out", dest="output", required=True)
    ap_trio.add_argument("--model", default="gpt-5-mini")
    ap_trio.set_defaults(func=cmd_trio)

    # arrange
    ap_arr = sub.add_parser("arrange", help="Place files into a BIDS-like structure.")
    ap_arr.add_argument("--out", dest="output", required=True)
    ap_arr.add_argument("--nsubjects", type=int, required=True)
    ap_arr.add_argument("--task", default="rest", help="Default task label for fNIRS/fMRI.")
    ap_arr.set_defaults(func=cmd_arrange)

    # validate
    ap_val = sub.add_parser("validate", help="Run bids-validator if available (stub otherwise).")
    ap_val.add_argument("--out", dest="output", required=True)
    ap_val.set_defaults(func=cmd_validate)

    # full
    ap_full = sub.add_parser("full", help="Run full pipeline: scan -> trio -> arrange -> validate")
    ap_full.add_argument("--in", dest="input", required=True)
    ap_full.add_argument("--out", dest="output", required=True)
    ap_full.add_argument("--model", default="gpt-5-mini")
    ap_full.add_argument("--task", default="rest")
    ap_full.set_defaults(func=cmd_full)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

