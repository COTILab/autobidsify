# trio.py
# BIDS Trio generation with BIDS validation and smart updates

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from utils import write_json, write_text, read_json, warn, info, fatal, debug
from constants import (
    TRIO_README, TRIO_PARTICIPANTS, TRIO_DATASET_DESC,
    LICENSE_WHITELIST, SEVERITY_WARN, SEVERITY_INFO
)
from llm import llm_trio_dataset_description, llm_trio_readme, llm_trio_participants

DEBUG_MODE = True

def check_trio_status(out_dir: Path) -> Dict[str, Any]:
    status = {
        "dataset_description": {"exists": False, "path": None, "data": None},
        "readme": {"exists": False, "path": None, "variant": None},
        "participants": {"exists": False, "path": None}
    }
    
    dd_path = out_dir / TRIO_DATASET_DESC
    if dd_path.exists():
        status["dataset_description"]["exists"] = True
        status["dataset_description"]["path"] = dd_path
        try:
            status["dataset_description"]["data"] = read_json(dd_path)
        except:
            pass
    
    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    for item in out_dir.iterdir():
        if item.is_file() and item.name.lower() in readme_variants:
            status["readme"]["exists"] = True
            status["readme"]["path"] = item
            status["readme"]["variant"] = item.name
            break
    
    parts_path = out_dir / TRIO_PARTICIPANTS
    if parts_path.exists():
        status["participants"]["exists"] = True
        status["participants"]["path"] = parts_path
    
    return status

def normalize_license_locally(license_str: str) -> Optional[str]:
    if not license_str:
        return None
    
    normalized = re.sub(r'[-\s]+', '', license_str.upper())
    
    mappings = {
        'PDDL': ['PDDL', 'PDDL10', 'PUBLICDOMAINDEDICATIONLICENSE'],
        'CC0': ['CC0', 'CC010', 'CREATIVECOMMONSZERO'],
        'PD': ['PD', 'PUBLICDOMAIN'],
        'CC-BY-4.0': ['CCBY40', 'CCBY4', 'CREATIVECOMMONSATTRIBUTION4'],
        'CC-BY-SA-4.0': ['CCBYSA40', 'CCBYSA4', 'CREATIVECOMMONSATTRIBUTIONSHAREALIKE4'],
        'BSD-3-Clause': ['BSD3CLAUSE', 'BSD3', 'BSDNEW', 'BSDREVISED'],
        'BSD-2-Clause': ['BSD2CLAUSE', 'BSD2', 'BSDORIGINAL', 'BSDOLD'],
        'MIT': ['MIT', 'MITLICENSE'],
        'GPL-2.0': ['GPL20', 'GPL2', 'GNUGPL2'],
        'GPL-2.0+': ['GPL20+', 'GPL2+', 'GPL2ORLATER'],
        'GPL-3.0': ['GPL30', 'GPL3', 'GNUGPL3'],
        'GPL-3.0+': ['GPL30+', 'GPL3+', 'GPL3ORLATER'],
        'LGPL-3.0+': ['LGPL30+', 'LGPL3+'],
        'MPL': ['MPL', 'MPL20', 'MOZILLAPUBLICLICENSE'],
        'CDDL-1.0': ['CDDL', 'CDDL10'],
        'GFDL-1.3': ['GFDL', 'GFDL13'],
        'CC-BY-NC-4.0': ['CCBYNC40', 'CCBYNC4'],
        'CC-BY-NC-SA-4.0': ['CCBYNCSA40', 'CCBYNCSA4'],
        'CC-BY-NC-ND-4.0': ['CCBYNCND40', 'CCBYNCND4']
    }
    
    for standard, variants in mappings.items():
        if normalized in variants:
            return standard
    
    return 'Non-Standard'

def _parse_llm_json_response(response_text: str, step_name: str, show_preview: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse LLM JSON response with multiple fallback strategies.
    
    Args:
        response_text: Raw LLM response
        step_name: Step name for logging
        show_preview: Whether to show preview on failure (False for large responses)
    """
    if not response_text or not response_text.strip():
        warn(f"{step_name}: LLM returned empty response")
        return None
    
    text = response_text.strip()
    
    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    elif text.startswith("```"):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    
    # Try direct parse
    try:
        obj = json.loads(text)
        if DEBUG_MODE:
            debug(f"{step_name}: ✓ JSON parsed successfully")
        return obj
    except json.JSONDecodeError as e:
        if DEBUG_MODE:
            debug(f"{step_name}: Direct parse failed: {e}")
        
        # Try raw_decode for extra text
        if "Extra data" in str(e) or "Expecting" in str(e):
            try:
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(text)
                if DEBUG_MODE:
                    debug(f"{step_name}: ✓ Extracted JSON using raw_decode")
                return obj
            except Exception as e2:
                if DEBUG_MODE:
                    debug(f"{step_name}: raw_decode failed: {e2}")
        
        # Try regex extraction
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                obj = json.loads(match.group(0))
                if DEBUG_MODE:
                    debug(f"{step_name}: ✓ Extracted JSON using regex")
                return obj
        except:
            pass
        
        # All strategies failed
        warn(f"{step_name}: Failed to parse JSON: {e}")
        if show_preview:
            warn(f"Response preview: {text[:500]}...")
        else:
            warn(f"Response length: {len(text)} chars (preview suppressed)")
        return None

def _is_markdown_content(text: str) -> bool:
    text = text.strip()
    return any([
        text.startswith('#'),
        text.startswith('##'),
        '# ' in text[:100],
        '\n## ' in text[:200],
        text.startswith('**'),
        '- ' in text[:100],
        '\n- ' in text[:200],
    ])

def _validate_dataset_description(dd: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    
    if not dd.get("Name"):
        issues.append("Missing required field: Name")
    if not dd.get("BIDSVersion"):
        issues.append("Missing required field: BIDSVersion")
    if not dd.get("License"):
        issues.append("Missing required field: License")
    elif dd.get("License") not in LICENSE_WHITELIST:
        issues.append(f"License '{dd.get('License')}' not in BIDS whitelist (will be auto-normalized)")
    
    if "Authors" in dd and not isinstance(dd["Authors"], list):
        issues.append(f"Authors must be an array, found: {type(dd['Authors']).__name__}")
    if "Funding" in dd and not isinstance(dd["Funding"], list):
        issues.append(f"Funding must be an array, found: {type(dd['Funding']).__name__}")
    if "EthicsApprovals" in dd and not isinstance(dd["EthicsApprovals"], list):
        issues.append(f"EthicsApprovals must be an array, found: {type(dd['EthicsApprovals']).__name__}")
    
    if dd.get("License") == "Non-Standard" and not dd.get("DataLicense"):
        issues.append("License='Non-Standard' requires DataLicense field")
    
    empty_fields = [k for k, v in dd.items() if v == "" or v == []]
    if empty_fields:
        issues.append(f"Empty fields (will be removed): {', '.join(empty_fields)}")
    
    is_valid = len([i for i in issues if "Missing required" in i or "must be an array" in i]) == 0
    return is_valid, issues

def _fix_field_types(dd: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    fixed = dd.copy()
    fixes = []
    
    # Fix Authors
    if "Authors" in fixed:
        if isinstance(fixed["Authors"], str):
            if fixed["Authors"].strip():
                fixed["Authors"] = [fixed["Authors"]]
                fixes.append("Converted Authors from string to array")
            else:
                del fixed["Authors"]
        elif isinstance(fixed["Authors"], list) and len(fixed["Authors"]) == 0:
            del fixed["Authors"]
    
    # Fix Funding
    if "Funding" in fixed:
        if isinstance(fixed["Funding"], str):
            if fixed["Funding"].strip():
                fixed["Funding"] = [fixed["Funding"]]
                fixes.append("Converted Funding from string to array")
            else:
                del fixed["Funding"]
        elif isinstance(fixed["Funding"], list) and len(fixed["Funding"]) == 0:
            del fixed["Funding"]
    
    # Fix EthicsApprovals
    if "EthicsApprovals" in fixed:
        if isinstance(fixed["EthicsApprovals"], str):
            if fixed["EthicsApprovals"].strip():
                fixed["EthicsApprovals"] = [fixed["EthicsApprovals"]]
                fixes.append("Converted EthicsApprovals from string to array")
            else:
                del fixed["EthicsApprovals"]
        elif isinstance(fixed["EthicsApprovals"], list) and len(fixed["EthicsApprovals"]) == 0:
            del fixed["EthicsApprovals"]
    
    # Remove empty strings
    keys_to_remove = [k for k, v in fixed.items() 
                      if v == "" and k not in ["Name", "BIDSVersion", "DatasetType", "License"]]
    for key in keys_to_remove:
        del fixed[key]
    
    return fixed, fixes

def _merge_dataset_description(existing: Dict[str, Any], new: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    merged = existing.copy()
    notes = []
    
    for key, new_value in new.items():
        if key == "BIDSVersion":
            if merged.get(key) != "1.10.0":
                merged[key] = "1.10.0"
                notes.append("Updated BIDSVersion to 1.10.0")
            continue
        
        existing_value = existing.get(key)
        
        if not existing_value and new_value:
            if isinstance(new_value, (list, str)):
                if new_value:
                    merged[key] = new_value
                    notes.append(f"Added field '{key}'")
            else:
                merged[key] = new_value
                notes.append(f"Added field '{key}'")
        elif existing_value and key == "License":
            if existing_value not in LICENSE_WHITELIST:
                normalized = normalize_license_locally(existing_value)
                if normalized and normalized in LICENSE_WHITELIST:
                    merged[key] = normalized
                    notes.append(f"Normalized License: '{existing_value}' -> '{normalized}'")
    
    return merged, notes

def generate_dataset_description(model: str, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    info("=== Generating dataset_description.json ===")
    
    dd_path = out_dir / TRIO_DATASET_DESC
    warnings = []
    
    # Load existing
    existing_dd = None
    if dd_path.exists():
        try:
            existing_dd = read_json(dd_path)
            info(f"Found existing file: {dd_path}")
            
            # Validate and fix
            is_valid, validation_issues = _validate_dataset_description(existing_dd)
            if not is_valid or validation_issues:
                info("Validating existing file format:")
                for issue in validation_issues:
                    info(f"  ⚠ {issue}")
                
                fixed_dd, fixes = _fix_field_types(existing_dd)
                if fixes:
                    info("Auto-fixing field types:")
                    for fix in fixes:
                        info(f"  ✓ {fix}")
                    existing_dd = fixed_dd
            else:
                info("✓ Existing file format is valid")
        except Exception as e:
            warn(f"Could not read existing file: {e}")
            existing_dd = None
    
    # Call LLM
    payload = json.dumps({
        "user_hints": bundle.get("user_hints", {}),
        "documents": bundle.get("documents", []),
        "counts_by_ext": bundle.get("counts_by_ext", {}),
        "existing": existing_dd
    }, ensure_ascii=False)
    
    result = None
    llm_dd = None
    
    try:
        response_text = llm_trio_dataset_description(model, payload)
        
        if DEBUG_MODE:
            debug(f"LLM response length: {len(response_text)} chars")
        
        result = _parse_llm_json_response(response_text, "dataset_description")
        
        if result:
            llm_dd = result.get("dataset_description", {})
            if DEBUG_MODE:
                debug(f"LLM returned fields: {list(llm_dd.keys())}")
        else:
            warn("Failed to parse LLM response")
    except Exception as e:
        warn(f"LLM call failed: {e}")
    
    # Merge data
    final_dd = None
    update_notes = []
    
    if llm_dd and existing_dd:
        if llm_dd != existing_dd:
            info("Merging LLM data with existing file...")
            final_dd, update_notes = _merge_dataset_description(existing_dd, llm_dd)
            if update_notes:
                info("Updates from LLM:")
                for note in update_notes:
                    info(f"  • {note}")
            else:
                info("No new fields from LLM")
        else:
            info("LLM returned same data as existing")
            final_dd = existing_dd.copy()
    elif llm_dd and not existing_dd:
        info("Creating new file from LLM data")
        final_dd = llm_dd
    elif not llm_dd and existing_dd:
        warn("LLM failed to extract data, using existing file")
        final_dd = existing_dd.copy()
    else:
        fatal("No data available")
        return {"warnings": [], "questions": []}
    
    # Build final structure (only non-empty fields)
    required_structure = {
        "Name": final_dd.get("Name", ""),
        "BIDSVersion": "1.10.0",
        "DatasetType": final_dd.get("DatasetType", "raw"),
        "License": final_dd.get("License", "")
    }
    
    if final_dd.get("HEDVersion"):
        required_structure["HEDVersion"] = final_dd.get("HEDVersion")
    
    # Authors: Must be array
    if final_dd.get("Authors"):
        authors = final_dd.get("Authors")
        if isinstance(authors, str) and authors.strip():
            required_structure["Authors"] = [authors]
            info("✓ Converted Authors from string to array")
        elif isinstance(authors, list) and len(authors) > 0:
            required_structure["Authors"] = authors
    
    if final_dd.get("Acknowledgements"):
        required_structure["Acknowledgements"] = final_dd.get("Acknowledgements")
    if final_dd.get("HowToAcknowledge"):
        required_structure["HowToAcknowledge"] = final_dd.get("HowToAcknowledge")
    
    # Funding: Must be array
    if final_dd.get("Funding"):
        funding = final_dd.get("Funding")
        if isinstance(funding, str) and funding.strip():
            required_structure["Funding"] = [funding]
            info("✓ Converted Funding from string to array")
        elif isinstance(funding, list) and len(funding) > 0:
            required_structure["Funding"] = funding
    
    # EthicsApprovals: Must be array
    if final_dd.get("EthicsApprovals"):
        ethics = final_dd.get("EthicsApprovals")
        if isinstance(ethics, str) and ethics.strip():
            required_structure["EthicsApprovals"] = [ethics]
            info("✓ Converted EthicsApprovals from string to array")
        elif isinstance(ethics, list) and len(ethics) > 0:
            required_structure["EthicsApprovals"] = ethics
    
    if final_dd.get("ReferencesAndLinks") and len(final_dd.get("ReferencesAndLinks")) > 0:
        required_structure["ReferencesAndLinks"] = final_dd.get("ReferencesAndLinks")
    if final_dd.get("DatasetDOI"):
        required_structure["DatasetDOI"] = final_dd.get("DatasetDOI")
    if final_dd.get("GeneratedBy") and len(final_dd.get("GeneratedBy")) > 0:
        required_structure["GeneratedBy"] = final_dd.get("GeneratedBy")
    if final_dd.get("SourceDatasets") and len(final_dd.get("SourceDatasets")) > 0:
        required_structure["SourceDatasets"] = final_dd.get("SourceDatasets")
    
    if final_dd.get("License") == "Non-Standard" and final_dd.get("DataLicense"):
        required_structure["DataLicense"] = final_dd.get("DataLicense")
    
    # Validate required fields
    if not required_structure["Name"]:
        warnings.append("WARNING: Missing 'Name' field (REQUIRED)")
    
    # License normalization
    if not required_structure["License"]:
        warnings.append(f"WARNING: Missing 'License' field (REQUIRED)")
    elif required_structure["License"] not in LICENSE_WHITELIST:
        original_license = required_structure["License"]
        normalized = normalize_license_locally(original_license)
        if normalized and normalized in LICENSE_WHITELIST:
            info(f"✓ License normalized: '{original_license}' -> '{normalized}'")
            required_structure["License"] = normalized
        else:
            warnings.append(f"WARNING: License '{original_license}' not recognized")
    
    if required_structure.get("License") == "Non-Standard" and not required_structure.get("DataLicense"):
        warnings.append("WARNING: License='Non-Standard' requires 'DataLicense' field")
    
    # Write file
    write_json(dd_path, required_structure)
    info(f"✓ {'Updated' if existing_dd else 'Created'}: {dd_path}")
    
    if result and "extraction_log" in result:
        info("Metadata extraction sources:")
        for field, source in result["extraction_log"].items():
            info(f"  {field}: {source}")
    
    if result and "normalization_notes" in result:
        for note in result["normalization_notes"]:
            info(note)
    
    if result and "warnings" in result:
        for w in result["warnings"]:
            warnings.append(w)
    
    return {"warnings": warnings, "questions": result.get("questions", []) if result else []}

def generate_readme(model: str, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Generate README.md using LLM."""
    info("=== Generating README.md ===")
    
    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    existing_readme = None
    
    for item in out_dir.iterdir():
        if item.is_file() and item.name.lower() in readme_variants:
            existing_readme = item
            break
    
    if existing_readme:
        info(f"✓ Found existing: {existing_readme.name}")
        return {"warnings": [], "questions": []}
    
    payload = json.dumps({
        "documents": bundle.get("documents", []),
        "user_hints": bundle.get("user_hints", {}),
        "existing_readme": None
    }, ensure_ascii=False)
    
    try:
        response_text = llm_trio_readme(model, payload)
        
        if _is_markdown_content(response_text):
            info("✓ LLM returned direct Markdown content")
            result = {"readme_content": response_text.strip()}
        else:
            result = _parse_llm_json_response(response_text, "README", show_preview=True)
            if result is None:
                info(f"Could not parse LLM response, using default README")
                result = {"readme_content": "# Dataset\n\nNeuroimaging dataset.\n"}
            else:
                if DEBUG_MODE:
                    debug(f"Successfully parsed README JSON response")
    except Exception as e:
        warn(f"README generation failed: {e}")
        result = {"readme_content": "# Dataset\n\nNeuroimaging dataset.\n"}
    
    readme_content = result.get("readme_content", "# Dataset\n\nNeuroimaging dataset.\n")
    write_text(out_dir / TRIO_README, readme_content)
    info(f"✓ Created: {TRIO_README}")
    
    if "extraction_log" in result:
        info("README extraction sources:")
        for field, source in result["extraction_log"].items():
            info(f"  {field}: {source}")
    
    return {"warnings": [], "questions": []}

def generate_participants(model: str, bundle: Dict[str, Any], out_dir: Path, force_simple: bool = False) -> Dict[str, Any]:
    """
    Generate participants.tsv - SIMPLIFIED VERSION.
    
    Args:
        force_simple: If True, generate simple sequential IDs even for large datasets
    
    New strategy:
    - Only generate if data is simple and small (≤ 100 subjects)
    - For complex or large datasets, defer to Plan stage
    - Plan stage has better subject detection logic
    """
    info("=== Generating participants.tsv ===")
    
    parts_path = out_dir / TRIO_PARTICIPANTS
    
    if parts_path.exists():
        info(f"✓ Found existing: {parts_path}")
        return {"warnings": [], "questions": []}
    
    n_subjects = bundle.get("user_hints", {}).get("n_subjects", 1)
    
    # DECISION: Defer to Plan stage for complex cases (unless forced)
    if not force_simple:
        if n_subjects > 100:
            info(f"Large dataset ({n_subjects} subjects) - deferring to Plan stage")
            info("  Plan stage will generate participants.tsv with proper subject detection")
            return {"warnings": [], "questions": [], "deferred": True}
        
        # Check if file structure is complex
        all_files = bundle.get("all_files", [])
        if len(all_files) > 500:
            info(f"Complex file structure ({len(all_files)} files) - deferring to Plan stage")
            return {"warnings": [], "questions": [], "deferred": True}
    
    # Simple case or forced: generate basic participants.tsv
    if force_simple:
        info(f"Generating basic participants.tsv (forced mode)")
    else:
        info(f"Simple dataset ({n_subjects} subjects) - generating basic participants.tsv")
    
    lines = ["participant_id\n"]
    for i in range(1, n_subjects + 1):
        lines.append(f"sub-{i:02d}\n")
    parts_content = "".join(lines)
    
    write_text(parts_path, parts_content)
    info(f"✓ Created: {parts_path} (basic format)")
    
    if not force_simple:
        info(f"  Note: Plan stage may update this file with additional columns")
    else:
        warn(f"  WARNING: Using sequential IDs (sub-01 to sub-{n_subjects:02d})")
        warn(f"  This may not match actual file structure. Consider running 'plan' instead.")
    
    return {"warnings": [], "questions": []}

def trio_generate_all(model: str, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    info("=== Generating BIDS Trio (all files) ===")
    
    status = check_trio_status(out_dir)
    
    info("Trio file status:")
    info(f"  dataset_description.json: {'EXISTS' if status['dataset_description']['exists'] else 'MISSING'}")
    info(f"  README.md: {'EXISTS' if status['readme']['exists'] else 'MISSING'}")
    info(f"  participants.tsv: {'EXISTS' if status['participants']['exists'] else 'MISSING'}")
    info("")
    
    all_warnings = []
    all_questions = []
    
    dd_result = generate_dataset_description(model, bundle, out_dir)
    all_warnings.extend(dd_result.get("warnings", []))
    all_questions.extend(dd_result.get("questions", []))
    
    readme_result = generate_readme(model, bundle, out_dir)
    all_warnings.extend(readme_result.get("warnings", []))
    all_questions.extend(readme_result.get("questions", []))
    
    parts_result = generate_participants(model, bundle, out_dir)
    all_warnings.extend(parts_result.get("warnings", []))
    all_questions.extend(parts_result.get("questions", []))
    
    # Track if participants.tsv was deferred
    parts_deferred = parts_result.get("deferred", False)
    
    # IMPORTANT: Summary of what was actually generated
    info("")
    info("=== Trio Generation Summary ===")
    
    # Check what exists now
    dd_exists = (out_dir / TRIO_DATASET_DESC).exists()
    readme_exists = (out_dir / TRIO_README).exists()
    parts_exists = (out_dir / TRIO_PARTICIPANTS).exists()
    
    generated_count = sum([dd_exists, readme_exists, parts_exists])
    deferred_count = 1 if parts_deferred else 0
    
    if dd_exists:
        info("✓ dataset_description.json - Generated")
    else:
        warn("✗ dataset_description.json - NOT generated")
    
    if readme_exists:
        info("✓ README.md - Generated")
    else:
        warn("✗ README.md - NOT generated")
    
    if parts_exists:
        info("✓ participants.tsv - Generated")
    elif parts_deferred:
        info("○ participants.tsv - Deferred to Plan stage")
        info("  → Reason: Complex dataset requires file structure analysis")
        info("  → Next step: Run 'plan' command to complete trio generation")
    else:
        warn("✗ participants.tsv - NOT generated")
    
    info("")
    info(f"Status: {generated_count}/3 generated, {deferred_count}/3 deferred")
    
    if parts_deferred:
        info("")
        info("To complete BIDS trio generation, run:")
        info(f"  python cli.py plan --output {out_dir} --model {model}")
    
    return {"warnings": all_warnings, "questions": all_questions}
