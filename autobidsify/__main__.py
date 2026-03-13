#!/usr/bin/env python3
"""
Command-line interface for BIDS Pipeline
NEW: Support for both OpenAI and Qwen (Ollama) models
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autobidsify.stages.ingest import ingest_data
from autobidsify.stages.evidence import build_evidence_bundle
from autobidsify.stages.classification import classify_files
from autobidsify.stages.trio import (
    trio_generate_all,
    generate_dataset_description,
    generate_readme,
    generate_participants
)
from autobidsify.converters.planner import build_bids_plan
from autobidsify.converters.executor import execute_bids_plan
from autobidsify.converters.validators import validate_bids_compatible
from autobidsify.utils import info, warn, fatal, read_json, read_yaml
from autobidsify.constants import QWEN_RECOMMENDED_MODELS


def is_qwen_model(model: str) -> bool:
    """Check if model is a Qwen model."""
    return model.startswith('qwen')


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o1/o3/gpt-5 series)."""
    return (
        model.startswith("o1") or 
        model.startswith("o3") or
        model.startswith("gpt-5")
    )


def validate_model(model: str) -> None:
    """Validate and display model information."""
    if is_qwen_model(model):
        info(f"Using Qwen model (via Ollama): {model}")
        info(f"  Make sure Ollama is running: ollama serve")
        info(f"  Make sure model is pulled: ollama pull {model}")
    elif is_reasoning_model(model):
        info(f"Using OpenAI reasoning model: {model}")
    else:
        info(f"Using OpenAI model: {model}")


def setup_parser():
    """Setup command-line argument parser with Qwen support."""
    parser = argparse.ArgumentParser(
        description="BIDS Standardization Pipeline v10 (OpenAI + Qwen support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Using OpenAI (default)
  python cli.py full --input data/ --output bids_out --model gpt-4o
  
  # Using Qwen (via Ollama) - Recommended for coding tasks
  python cli.py full --input data/ --output bids_out --model qwen2.5-coder:7b
  
  # Using Qwen - General purpose
  python cli.py full --input data/ --output bids_out --model qwen2.5:14b
  
  # Using Qwen - Lightweight
  python cli.py full --input data/ --output bids_out --model qwen2.5:7b
  
  # With custom ID strategy
  python cli.py full --input data/ --output bids_out \\
    --model qwen2.5-coder:14b --id-strategy numeric

Supported Qwen Models (via Ollama):
  General:
    - qwen2.5:7b         (Balanced, 4.7GB)
    - qwen2.5:14b        (Better performance)
    - qwen2.5:32b        (Strong performance)
    - qwen3:8b           (Latest generation)
  
  Coding (Recommended for BIDS pipeline):
    - qwen2.5-coder:7b   (Code-optimized, recommended)
    - qwen2.5-coder:14b  (Better code understanding)
    - qwen2.5-coder:32b  (Near GPT-4o performance)

Setup Ollama:
  1. Install: https://ollama.com/download
  2. Start: ollama serve
  3. Pull model: ollama pull qwen2.5-coder:7b
  4. Install Python lib: pip install ollama
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--input', type=str, required=True,
                            help='Input data path (file or directory)')
    full_parser.add_argument('--output', type=str, required=True,
                            help='Output directory for BIDS dataset')
    full_parser.add_argument('--nsubjects', type=int, default=None,
                            help='Number of subjects (optional, will auto-detect if not provided)')
    full_parser.add_argument('--modality', choices=['mri', 'nirs', 'mixed'],
                            help='Data modality (optional, will detect if not provided)')
    full_parser.add_argument('--describe', type=str,
                            help='Additional description or notes about the dataset')
    full_parser.add_argument('--model', type=str, default='gpt-4o',
                            help='LLM model: OpenAI (gpt-4o, gpt-4o-mini) or Qwen (qwen2.5-coder:7b, qwen2.5:14b, etc.)')
    full_parser.add_argument('--id-strategy', type=str,
                            choices=['auto', 'numeric', 'semantic'],
                            default='auto',
                            help='Subject ID strategy (auto/numeric/semantic)')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data')
    ingest_parser.add_argument('--input', type=str, required=True,
                              help='Input data path (file or directory)')
    ingest_parser.add_argument('--output', type=str, required=True,
                              help='Output directory')
    
    # Evidence command
    evidence_parser = subparsers.add_parser('evidence', help='Build evidence bundle')
    evidence_parser.add_argument('--output', type=str, required=True,
                                help='Output directory')
    evidence_parser.add_argument('--nsubjects', type=int, default=None,
                                help='Number of subjects (optional, will auto-detect)')
    evidence_parser.add_argument('--modality', choices=['mri', 'nirs', 'mixed'],
                                help='Data modality')
    evidence_parser.add_argument('--describe', type=str,
                                help='Additional description')

    # Classification command
    classify_parser = subparsers.add_parser('classify', help='Classify files')
    classify_parser.add_argument('--output', type=str, required=True,
                                help='Output directory')
    
    # Trio command
    trio_parser = subparsers.add_parser('trio', help='Generate trio files')
    trio_parser.add_argument('--output', type=str, required=True,
                            help='Output directory')
    trio_parser.add_argument('--file', type=str,
                            choices=['dataset_description', 'readme', 'participants', 'all'],
                            default='all',
                            help='Which trio file(s) to generate')
    trio_parser.add_argument('--model', type=str, default='gpt-4o',
                            help='LLM model (OpenAI or Qwen)')
    
    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Generate BIDS plan')
    plan_parser.add_argument('--output', type=str, required=True,
                            help='Output directory')
    plan_parser.add_argument('--model', type=str, default='gpt-4o',
                            help='LLM model (OpenAI or Qwen)')
    plan_parser.add_argument('--id-strategy', type=str,
                            choices=['auto', 'numeric', 'semantic'],
                            default='auto',
                            help='Subject ID strategy')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute conversions')
    execute_parser.add_argument('--output', type=str, required=True,
                                help='Output directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate BIDS dataset')
    validate_parser.add_argument('--output', type=str, required=True,
                                help='Output directory')
    
    return parser


def run_full_pipeline(args):
    """Run complete BIDS conversion pipeline."""
    info("=== Starting Full Pipeline v10 ===")
    validate_model(args.model)
    info(f"Subject ID strategy: {args.id_strategy}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    needs_classification = (args.modality == 'mixed' or args.modality is None)
    
    # Stage 1: Ingest
    info("\n[1/7] Ingesting data...")
    ingest_data(args.input, output_dir)
    
    # Stage 2: Evidence
    info("\n[2/7] Building evidence bundle...")
    user_hints = {
        "n_subjects": args.nsubjects,
        "modality_hint": args.modality,
        "user_text": args.describe or ""
    }
    build_evidence_bundle(output_dir, user_hints)
    
    # Stage 3: Classification (if needed)
    if needs_classification:
        info("\n[3/7] Classifying files (mixed modality detected)...")
        classify_files(output_dir)
    else:
        info(f"\n[3/7] Skipping classification (single modality: {args.modality})...")
        info("  ✓ No classification needed for single-modality datasets")
    
    # Stage 4: Trio
    info("\n[4/7] Generating BIDS trio files...")
    bundle = read_json(output_dir / "_staging" / "evidence_bundle.json")
    
    count_source = bundle.get("subject_detection", {}).get("count_source")
    if count_source == "user_provided":
        info(f"✓ Using user-provided subject count: {args.nsubjects}")
    
    dd_result = generate_dataset_description(args.model, bundle, output_dir)
    readme_result = generate_readme(args.model, bundle, output_dir)
    parts_result = generate_participants(args.model, bundle, output_dir)
    
    all_warnings = []
    all_warnings.extend(dd_result.get("warnings", []))
    all_warnings.extend(readme_result.get("warnings", []))
    all_warnings.extend(parts_result.get("warnings", []))
    
    if all_warnings:
        warn("\nWarnings:")
        for w in all_warnings:
            warn(f"  {w}")
    
    # Stage 5: Plan
    info("\n[5/7] Generating BIDS plan...")
    trio_status = {
        "dataset_description": (output_dir / "dataset_description.json").exists(),
        "readme": (output_dir / "README.md").exists(),
        "participants": (output_dir / "participants.tsv").exists()
    }
    planning_inputs = {"evidence_bundle": bundle, "trio_status": trio_status}
    
    plan_result = build_bids_plan(args.model, planning_inputs, output_dir, id_strategy=args.id_strategy)
    
    if plan_result.get("status") == "blocked":
        fatal("\n⚠ BLOCKING QUESTIONS DETECTED:")
        for q in plan_result.get("questions", []):
            if q.get("severity") == "block":
                fatal(f"  • {q.get('message')}")
        fatal("\nPlease resolve these issues and re-run the plan command")
        return
    
    if not (output_dir / "participants.tsv").exists():
        warn("WARNING: participants.tsv was not created by Plan stage")
    
    # Stage 6: Execute
    info("\n[6/7] Executing conversions...")
    
    ingest_info = read_json(output_dir / "_staging" / "ingest_info.json")
    actual_data_path = Path(ingest_info.get("actual_data_path", 
                                            output_dir / "_staging" / "extracted"))
    
    plan_dict = read_yaml(output_dir / "_staging" / "BIDSPlan.yaml")
    execute_bids_plan(actual_data_path, output_dir, plan_dict, {})
    
    # Stage 7: Validate
    info("\n[7/7] Validating BIDS dataset...")
    validate_bids_compatible(output_dir)
    
    info("\n=== Pipeline Complete ===")
    info(f"Output: {output_dir / 'bids_compatible'}")


def run_ingest(args):
    """Run ingest stage."""
    info("=== Running Ingest ===")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_data(args.input, output_dir)
    info("✓ Ingest complete")


def run_evidence(args):
    """Run evidence bundle generation."""
    info("=== Building Evidence Bundle ===")
    output_dir = Path(args.output)
    user_hints = {
        "n_subjects": args.nsubjects,
        "modality_hint": args.modality,
        "user_text": args.describe or ""
    }
    build_evidence_bundle(output_dir, user_hints)
    info("✓ Evidence bundle complete")


def run_classify(args):
    """Run file classification."""
    info("=== Classifying Files ===")
    output_dir = Path(args.output)
    classify_files(output_dir)
    info("✓ Classification complete")


def run_trio(args):
    """Run trio file generation."""
    info("=== Generating BIDS Trio ===")
    validate_model(args.model)
    output_dir = Path(args.output)
    
    bundle_path = output_dir / "_staging" / "evidence_bundle.json"
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        return
    
    bundle = read_json(bundle_path)
    
    if args.file == 'dataset_description':
        result = generate_dataset_description(args.model, bundle, output_dir)
    elif args.file == 'readme':
        result = generate_readme(args.model, bundle, output_dir)
    elif args.file == 'participants':
        result = generate_participants(args.model, bundle, output_dir)
    else:
        result = trio_generate_all(args.model, bundle, output_dir)
    
    if result.get("warnings"):
        warn("\nWarnings:")
        for w in result["warnings"]:
            warn(f"  {w}")
    
    info("✓ Trio generation complete")


def run_plan(args):
    """Run BIDS plan generation."""
    info("=== Generating BIDS Plan ===")
    validate_model(args.model)
    info(f"Subject ID strategy: {args.id_strategy}")
    
    output_dir = Path(args.output)
    
    bundle_path = output_dir / "_staging" / "evidence_bundle.json"
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        return
    
    bundle = read_json(bundle_path)
    
    trio_status = {
        "dataset_description": (output_dir / "dataset_description.json").exists(),
        "readme": (output_dir / "README.md").exists(),
        "participants": (output_dir / "participants.tsv").exists()
    }
    
    planning_inputs = {"evidence_bundle": bundle, "trio_status": trio_status}
    
    result = build_bids_plan(args.model, planning_inputs, output_dir, id_strategy=args.id_strategy)
    
    if result.get("status") == "ok":
        info("✓ BIDS plan generation complete")
    elif result.get("status") == "blocked":
        warn("\n⚠ BLOCKING QUESTIONS DETECTED:")
        for q in result.get("questions", []):
            if q.get("severity") == "block":
                warn(f"  • {q.get('message')}")
        warn("\nPlease resolve these issues and re-run this command")
    else:
        warn("BIDS plan generation encountered errors")


def run_execute(args):
    """Run BIDS plan execution."""
    info("=== Executing BIDS Plan ===")
    output_dir = Path(args.output)
    
    plan_path = output_dir / "_staging" / "BIDSPlan.yaml"
    if not plan_path.exists():
        fatal(f"BIDS plan not found: {plan_path}")
        return
    
    plan_dict = read_yaml(plan_path)
    
    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    if not ingest_info_path.exists():
        fatal(f"Ingest info not found: {ingest_info_path}")
        return
    
    ingest_info = read_json(ingest_info_path)
    actual_data_path = Path(ingest_info.get("actual_data_path", 
                                            output_dir / "_staging" / "extracted"))
    
    aux_inputs = {}
    
    result = execute_bids_plan(actual_data_path, output_dir, plan_dict, aux_inputs)
    
    info("✓ Execution complete")
    info(f"  BIDS dataset: {result.get('bids_root')}")


def run_validate(args):
    """Run BIDS validation."""
    info("=== Validating BIDS Dataset ===")
    output_dir = Path(args.output)
    
    result = validate_bids_compatible(output_dir)
    
    if result.get("status") == "complete":
        info("")
        info("✓ Validation complete")
    else:
        warn("Validation encountered errors")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'full':
            run_full_pipeline(args)
        elif args.command == 'ingest':
            run_ingest(args)
        elif args.command == 'evidence':
            run_evidence(args)
        elif args.command == 'classify':
            run_classify(args)
        elif args.command == 'trio':
            run_trio(args)
        elif args.command == 'plan':
            run_plan(args)
        elif args.command == 'execute':
            run_execute(args)
        elif args.command == 'validate':
            run_validate(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        warn("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        fatal(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()