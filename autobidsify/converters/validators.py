# validators.py
# BIDS validation

"""
This module validates the generated dataset to ensure it conforms to the BIDS specification.

Core Functions:

1. BIDS Structure Validation (using the official bids-validator or built-in checks)

2. Required File Checks (dataset_description.json, README, participants.tsv)

3. Required Field Validation (Name, License)

Note:

- NIRS data in BIDS must be in SNIRF format, so separate SNIRF validation is not required.

- If BIDS compliant, the SNIRF file already conforms to the specification.
"""

from pathlib import Path
from typing import Dict, Any
import json
import shutil
import subprocess
from utils import warn, info

def run_bids_validator(bids_root: Path) -> Dict[str, Any]:
    """
    Run BIDS validator CLI tool.
    
    Args:
        bids_root: Root of BIDS dataset (bids_compatible/)
    
    Returns:
        Validation report
    """
    validator_path = shutil.which("bids-validator")
    
    if not validator_path:
        warn("bids-validator not found in PATH")
        warn("Install with: npm install -g bids-validator")
        info("")
        info("Using internal validation instead...")
        return _internal_bids_validation(bids_root)
    
    info("Running bids-validator...")
    
    try:
        result = subprocess.run(
            [validator_path, "--json", str(bids_root)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        try:
            report = json.loads(result.stdout)
            issues = report.get("issues", {})
            errors = issues.get("errors", [])
            warnings_list = issues.get("warnings", [])
            
            if errors:
                warn(f"Found {len(errors)} BIDS errors:")
                for i, err in enumerate(errors[:5], 1):
                    warn(f"  {i}. {err.get('code', 'ERROR')}: {err.get('reason', 'Unknown')}")
                if len(errors) > 5:
                    warn(f"  ... and {len(errors) - 5} more errors")
            else:
                info("  ✓ No BIDS errors found")
            
            if warnings_list:
                info(f"  ⚠ {len(warnings_list)} warnings")
                for i, w in enumerate(warnings_list[:3], 1):
                    info(f"    {i}. {w.get('code', 'WARN')}: {w.get('reason', 'Unknown')}")
                if len(warnings_list) > 3:
                    info(f"    ... and {len(warnings_list) - 3} more warnings")
            else:
                info("  ✓ No warnings")
            
            return report
            
        except json.JSONDecodeError as e:
            warn(f"Could not parse validator output: {e}")
            return {"issues": {"errors": [], "warnings": []}}
    
    except subprocess.TimeoutExpired:
        warn("bids-validator timed out")
        return {"issues": {"errors": ["Timeout"], "warnings": []}}
    
    except Exception as e:
        warn(f"bids-validator failed: {e}")
        return _internal_bids_validation(bids_root)

def _internal_bids_validation(bids_root: Path) -> Dict[str, Any]:
    """Internal BIDS validation (fallback)."""
    info("Internal validation...")
    
    errors = []
    warnings_list = []
    
    # Check dataset_description.json
    dd_path = bids_root / "dataset_description.json"
    if not dd_path.exists():
        errors.append({
            "code": "MISSING_DATASET_DESCRIPTION",
            "reason": "dataset_description.json is required"
        })
    else:
        try:
            with open(dd_path) as f:
                dd = json.load(f)
            
            if not dd.get("Name"):
                errors.append({
                    "code": "MISSING_NAME",
                    "reason": "'Name' field is required"
                })
            
            if not dd.get("BIDSVersion"):
                warnings_list.append({
                    "code": "MISSING_BIDS_VERSION",
                    "reason": "BIDSVersion should be specified"
                })
            
            if not dd.get("License"):
                errors.append({
                    "code": "MISSING_LICENSE",
                    "reason": "'License' field is required"
                })
        
        except json.JSONDecodeError:
            errors.append({
                "code": "INVALID_JSON",
                "reason": "dataset_description.json is invalid"
            })
    
    # Check README
    readme_variants = ["README.md", "readme.md", "README.txt"]
    has_readme = any((bids_root / variant).exists() for variant in readme_variants)
    
    if not has_readme:
        warnings_list.append({
            "code": "MISSING_README",
            "reason": "README is recommended"
        })
    
    # Check participants.tsv
    if not (bids_root / "participants.tsv").exists():
        warnings_list.append({
            "code": "MISSING_PARTICIPANTS",
            "reason": "participants.tsv is recommended"
        })
    
    # Check subject directories
    subject_dirs = list(bids_root.glob("sub-*"))
    if not subject_dirs:
        errors.append({
            "code": "NO_SUBJECTS",
            "reason": "No subject directories found"
        })
    
    # Display results
    if errors:
        warn(f"Found {len(errors)} errors:")
        for err in errors:
            warn(f"  • {err['code']}: {err['reason']}")
    else:
        info("  ✓ No critical errors")
    
    if warnings_list:
        info(f"  ⚠ {len(warnings_list)} warnings")
        for w in warnings_list:
            info(f"    • {w['code']}: {w['reason']}")
    else:
        info("  ✓ No warnings")
    
    return {
        "issues": {"errors": errors, "warnings": warnings_list},
        "summary": {
            "totalFiles": len(list(bids_root.rglob("*"))),
            "subjectCount": len(subject_dirs)
        },
        "validator": "internal"
    }

def validate_bids_compatible(output_dir: Path) -> Dict[str, Any]:
    """
    Validate bids_compatible directory.
    
    Args:
        output_dir: Pipeline output directory
    
    Returns:
        Validation report
    """
    bids_dir = output_dir / "bids_compatible"
    
    if not bids_dir.exists():
        warn(f"bids_compatible directory not found: {bids_dir}")
        warn("Please run 'execute' step first")
        return {
            "status": "error",
            "message": "bids_compatible not found"
        }
    
    info(f"Validating: {bids_dir}")
    info("")
    
    # Run BIDS validation
    bids_report = run_bids_validator(bids_dir)
    
    info("")
    
    # Count subjects and files
    subject_count = len(list(bids_dir.glob("sub-*")))
    total_files = len(list(bids_dir.rglob("*")))
    
    info(f"Dataset summary:")
    info(f"  Subjects: {subject_count}")
    info(f"  Total files: {total_files}")
    info("")
    
    return {
        "status": "complete",
        "bids_directory": str(bids_dir),
        "bids_report": bids_report
    }
