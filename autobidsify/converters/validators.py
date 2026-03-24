# validators.py
# BIDS dataset validation — three-tier approach

"""
Validation Module

Validates the generated BIDS dataset using a three-tier approach:

  Tier 1 — Python bids_validator package (pip install bids-validator)
            File-by-file path compliance check using BIDSValidator.is_bids().
            Checks whether each file's relative path conforms to the BIDS
            filename convention. No Node.js required.
            Limitation: validates filenames only, not dataset-level structure.

  Tier 2 — npm bids-validator CLI (npm install -g bids-validator)
            Full dataset-level structural validation. Checks dataset_description.json,
            participants.tsv, sidecar JSON completeness, etc.
            Used automatically if installed and Tier 1 is available.

  Tier 3 — Internal fallback (_internal_bids_validation)
            Basic checks for required files and fields. Pure Python, no external
            tools needed. Used when neither Tier 1 nor Tier 2 is available.

Installation:
    pip install bids-validator          # Tier 1 (included in autobidsify dependencies)
    npm install -g bids-validator       # Tier 2 (optional, for full validation)
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import shutil
import subprocess

from autobidsify.utils import warn, info


# ============================================================================
# Tier 1: Python bids_validator — file-path compliance check
# ============================================================================

def _run_python_bids_validator(bids_root: Path) -> Dict[str, Any]:
    """
    Validate BIDS file paths using the bids_validator Python package.

    Uses BIDSValidator.is_bids() to check each file's relative path against
    the BIDS filename convention. Paths must be relative to the dataset root
    and prefixed with '/' as required by the API.

    This tier checks filename conformance only — it does not validate
    dataset-level structure (dataset_description.json contents, etc.).

    Args:
        bids_root: Root of the BIDS dataset (bids_compatible/)

    Returns:
        Dict with keys: 'valid_files', 'invalid_files', 'summary', 'validator'
    """
    try:
        from bids_validator import BIDSValidator
    except ImportError:
        return {"available": False}

    validator = BIDSValidator()

    valid_files:   List[str] = []
    invalid_files: List[str] = []
    skipped_files: List[str] = []  # derivatives/, _staging/, hidden files

    # Collect all files under bids_root
    for abs_path in sorted(bids_root.rglob("*")):
        if not abs_path.is_file():
            continue

        rel = abs_path.relative_to(bids_root)
        rel_str = str(rel).replace("\\", "/")

        # Skip derivatives and other non-subject directories
        # (bids_validator.is_bids() returns False for these by design)
        first_part = rel_str.split("/")[0]
        if first_part in ("derivatives", "sourcedata", "code", "stimuli"):
            skipped_files.append(rel_str)
            continue

        # API requires leading '/' and relative path from dataset root
        bids_path = "/" + rel_str

        try:
            if validator.is_bids(bids_path):
                valid_files.append(rel_str)
            else:
                invalid_files.append(rel_str)
        except Exception:
            # Malformed path — treat as invalid but don't crash
            invalid_files.append(rel_str)

    total_checked = len(valid_files) + len(invalid_files)
    pass_rate = (
        round(len(valid_files) / total_checked * 100, 1)
        if total_checked > 0 else 0.0
    )

    # Display results
    info(f"  Python bids_validator: checked {total_checked} files")
    if invalid_files:
        info(f"  ⚠ {len(invalid_files)} file(s) with non-BIDS-compliant paths:")
        for f in invalid_files[:10]:
            info(f"      {f}")
        if len(invalid_files) > 10:
            info(f"      ... and {len(invalid_files) - 10} more")
    else:
        info(f"  ✓ All {len(valid_files)} checked files have BIDS-compliant paths")

    info(f"  Skipped (derivatives/non-subject): {len(skipped_files)} files")

    return {
        "available":     True,
        "valid_files":   valid_files,
        "invalid_files": invalid_files,
        "skipped_files": skipped_files,
        "summary": {
            "total_checked":    total_checked,
            "valid_count":      len(valid_files),
            "invalid_count":    len(invalid_files),
            "skipped_count":    len(skipped_files),
            "pass_rate_pct":    pass_rate,
        },
        "validator": "bids_validator_python",
    }


# ============================================================================
# Tier 2: npm bids-validator CLI — full dataset validation
# ============================================================================

def _run_npm_bids_validator(bids_root: Path) -> Dict[str, Any]:
    """
    Run the npm bids-validator CLI for full dataset-level validation.

    Args:
        bids_root: Root of the BIDS dataset

    Returns:
        Validation report dict, or {'available': False} if not installed.
    """
    validator_path = shutil.which("bids-validator")
    if not validator_path:
        return {"available": False}

    info("  Running npm bids-validator (full structural validation)...")

    try:
        result = subprocess.run(
            [validator_path, "--json", str(bids_root)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            warn(f"  Could not parse npm bids-validator output: {e}")
            return {"available": True, "parse_error": str(e)}

        issues       = report.get("issues", {})
        errors       = issues.get("errors", [])
        warnings_lst = issues.get("warnings", [])

        if errors:
            warn(f"  Found {len(errors)} BIDS error(s):")
            for i, err in enumerate(errors, 1):
                warn(f"    {i}. {err.get('code', 'ERROR')}: {err.get('reason', 'Unknown')}")
        else:
            info("  ✓ No BIDS errors found")

        if warnings_lst:
            info(f"  ⚠ {len(warnings_lst)} warning(s):")
            for i, w in enumerate(warnings_lst, 1):
                info(f"      {i}. {w.get('code', 'WARN')}: {w.get('reason', 'Unknown')}")
        else:
            info("  ✓ No warnings")

        report["available"] = True
        return report

    except subprocess.TimeoutExpired:
        warn("  npm bids-validator timed out")
        return {"available": True, "error": "timeout"}
    except Exception as e:
        warn(f"  npm bids-validator failed: {e}")
        return {"available": True, "error": str(e)}


# ============================================================================
# Tier 3: Internal fallback — basic structural checks
# ============================================================================

def _internal_bids_validation(bids_root: Path) -> Dict[str, Any]:
    """
    Internal fallback BIDS validation using pure Python.

    Checks for required files and required fields inside
    dataset_description.json. Used when neither the Python package
    nor the npm CLI is available.

    Args:
        bids_root: Root of the BIDS dataset

    Returns:
        Validation report dict with 'issues', 'summary', 'validator' keys.
    """
    info("  Running internal validation (basic structural checks)...")

    errors:       List[Dict] = []
    warnings_lst: List[Dict] = []

    # --- dataset_description.json ----------------------------------------
    dd_path = bids_root / "dataset_description.json"
    if not dd_path.exists():
        errors.append({
            "code":   "MISSING_DATASET_DESCRIPTION",
            "reason": "dataset_description.json is required",
        })
    else:
        try:
            with open(dd_path, encoding="utf-8") as f:
                dd = json.load(f)
            if not dd.get("Name"):
                errors.append({
                    "code":   "MISSING_NAME",
                    "reason": "'Name' field is required in dataset_description.json",
                })
            if not dd.get("BIDSVersion"):
                warnings_lst.append({
                    "code":   "MISSING_BIDS_VERSION",
                    "reason": "BIDSVersion should be specified",
                })
            if not dd.get("License"):
                errors.append({
                    "code":   "MISSING_LICENSE",
                    "reason": "'License' field is required in dataset_description.json",
                })
        except json.JSONDecodeError:
            errors.append({
                "code":   "INVALID_JSON",
                "reason": "dataset_description.json contains invalid JSON",
            })

    # --- README -----------------------------------------------------------
    readme_variants = ["README.md", "readme.md", "README.txt", "README"]
    if not any((bids_root / v).exists() for v in readme_variants):
        warnings_lst.append({
            "code":   "MISSING_README",
            "reason": "README file is recommended",
        })

    # --- participants.tsv -------------------------------------------------
    if not (bids_root / "participants.tsv").exists():
        warnings_lst.append({
            "code":   "MISSING_PARTICIPANTS",
            "reason": "participants.tsv is recommended",
        })

    # --- Subject directories ----------------------------------------------
    subject_dirs = list(bids_root.glob("sub-*"))
    if not subject_dirs:
        errors.append({
            "code":   "NO_SUBJECTS",
            "reason": "No sub-* directories found in bids_compatible/",
        })

    # --- Display ----------------------------------------------------------
    if errors:
        warn(f"  Found {len(errors)} error(s):")
        for err in errors:
            warn(f"    • {err['code']}: {err['reason']}")
    else:
        info("  ✓ No critical errors")

    if warnings_lst:
        info(f"  ⚠ {len(warnings_lst)} warning(s):")
        for w in warnings_lst:
            info(f"      • {w['code']}: {w['reason']}")
    else:
        info("  ✓ No warnings")

    total_files = len(list(bids_root.rglob("*")))

    return {
        "available": True,
        "issues": {
            "errors":   errors,
            "warnings": warnings_lst,
        },
        "summary": {
            "totalFiles":   total_files,
            "subjectCount": len(subject_dirs),
        },
        "validator": "internal",
    }


# ============================================================================
# Unified validator — runs all available tiers
# ============================================================================

def run_bids_validator(bids_root: Path) -> Dict[str, Any]:
    """
    Run BIDS validation using all available tiers.

    Tier 1 (Python bids_validator) and Tier 2 (npm bids-validator) are
    independent and complementary — Tier 1 checks filenames per file,
    Tier 2 checks full dataset structure. Both are run when available.
    If neither is available, Tier 3 (internal) is used as fallback.

    Args:
        bids_root: Root of the BIDS-compatible dataset directory.

    Returns:
        Dict with keys:
          'python_validator'  — Tier 1 result (or {'available': False})
          'npm_validator'     — Tier 2 result (or {'available': False})
          'internal'          — Tier 3 result (always present as fallback)
    """
    report: Dict[str, Any] = {}
    any_tier_ran = False

    # --- Tier 1: Python bids_validator ------------------------------------
    info("  [Tier 1] Python bids_validator (file-path compliance)...")
    py_result = _run_python_bids_validator(bids_root)
    report["python_validator"] = py_result
    if py_result.get("available"):
        report["npm_validator"] = {"available": False, "skipped": "Tier 1 succeeded"}
        report["internal"]      = {"available": False, "skipped": "Tier 1 succeeded"}
        return report
    info("  [Tier 1] bids_validator Python package not available.")
    info("           Install with: pip install bids-validator")

    # --- Tier 2: npm bids-validator CLI (only if Tier 1 unavailable) -----
    info("  [Tier 2] npm bids-validator (full structural validation)...")
    npm_result = _run_npm_bids_validator(bids_root)
    report["npm_validator"] = npm_result
    if npm_result.get("available"):
        report["internal"] = {"available": False, "skipped": "Tier 2 succeeded"}
        return report
    info("  [Tier 2] npm bids-validator not found.")
    info("           Install with: npm install -g bids-validator")

    # --- Tier 3: Internal fallback (only if Tier 1 and Tier 2 both unavailable) ---
    info("  [Tier 3] Internal validation (required-file checks)...")
    report["internal"] = _internal_bids_validation(bids_root)
    info("  For more thorough validation:")
    info("    pip install bids-validator        # filename compliance")
    info("    npm install -g bids-validator     # full structural check")

    return report


# ============================================================================
# Public entry point
# ============================================================================

def validate_bids_compatible(output_dir: Path) -> Dict[str, Any]:
    """
    Validate the bids_compatible/ directory produced by the execute stage.

    Args:
        output_dir: Pipeline output directory (parent of bids_compatible/).

    Returns:
        Dict with 'status', 'bids_directory', and 'bids_report' keys.
    """
    bids_dir = output_dir / "bids_compatible"

    if not bids_dir.exists():
        warn(f"bids_compatible directory not found: {bids_dir}")
        warn("Please run 'autobidsify execute' first.")
        return {
            "status":  "error",
            "message": "bids_compatible not found",
        }

    info(f"Validating: {bids_dir}")
    info("")

    bids_report = run_bids_validator(bids_dir)

    info("")

    subject_count = len(list(bids_dir.glob("sub-*")))
    total_files   = len(list(bids_dir.rglob("*")))

    info("Dataset summary:")
    info(f"  Subjects:    {subject_count}")
    info(f"  Total files: {total_files}")
    info("")

    return {
        "status":         "complete",
        "bids_directory": str(bids_dir),
        "bids_report":    bids_report,
    }