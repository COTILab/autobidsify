# evidence.py
# Evidence bundle builder with automatic subject detection

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from utils import list_all_files, write_json, sha1_head, warn, info, fatal, copy_file, read_json
from constants import MAX_TEXT_SIZE, MAX_PDF_SIZE, MAX_DOCX_SIZE, MAX_PDF_PAGES
import csv
import re

TEXT_EXT = {".txt", ".md", ".rst", ".html", ".htm", ".log"}
TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls"}
DOC_EXT = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".odt"}
MRI_EXT = {".nii", ".dcm"}
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz"}
NIRS_EXT = {".snirf", ".nirs", ".mat", ".h5", ".hdf5"}
ARRAY_EXT = {".mat", ".h5", ".hdf5", ".npy", ".npz"}
TRIO_NAMES = {"readme.md", "participants.tsv", "dataset_description.json"}

def _is_trio_file(name: str) -> bool:
    return name.lower() in TRIO_NAMES

def _extract_text_content(path: Path) -> Optional[str]:
    try:
        size = path.stat().st_size
        if size > MAX_TEXT_SIZE:
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                return f.read(MAX_TEXT_SIZE) + f"\n\n[TRUNCATED: {size} bytes]"
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[ERROR: {e}]"

def _extract_pdf_content(path: Path) -> Optional[str]:
    size = path.stat().st_size
    if size > MAX_PDF_SIZE:
        return f"[PDF too large: {size} bytes]"
    
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text_parts = []
            max_pages = min(len(pdf.pages), MAX_PDF_PAGES)
            for page_num in range(max_pages):
                text = pdf.pages[page_num].extract_text()
                if text:
                    text_parts.append(text)
            result = "\n\n".join(text_parts)
            if max_pages < len(pdf.pages):
                result += f"\n\n[TRUNCATED: {max_pages}/{len(pdf.pages)} pages]"
            return result
    except ImportError:
        pass
    except Exception as e:
        warn(f"pdfplumber failed: {e}")
    
    try:
        import PyPDF2
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            max_pages = min(len(pdf_reader.pages), MAX_PDF_PAGES)
            text_parts = [pdf_reader.pages[i].extract_text() for i in range(max_pages)]
            return "\n\n".join(text_parts)
    except:
        pass
    
    return "[ERROR: No PDF library available]"

def _extract_docx_content(path: Path) -> Optional[str]:
    try:
        import docx
        doc = docx.Document(path)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(text_parts)[:100000]
    except:
        return "[ERROR: python-docx not installed]"

def _extract_document_content(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXT:
        return _extract_text_content(path)
    elif suffix == ".pdf":
        return _extract_pdf_content(path)
    elif suffix == ".docx":
        return _extract_docx_content(path)
    return None

def _table_head(path: Path, max_rows: int = 5) -> Dict[str, Any]:
    head = {"rows": []}
    suf = path.suffix.lower()
    if suf not in {".csv", ".tsv"}:
        return head
    
    dialect = csv.excel_tab if suf == ".tsv" else csv.excel
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f, dialect=dialect)
            for i, row in enumerate(r):
                head["rows"].append(row)
                if i >= max_rows:
                    break
    except:
        pass
    return head

def detect_kind(p: Path) -> str:
    s = p.suffix.lower()
    name = p.name.lower()
    
    if _is_trio_file(name):
        return "user_trio"
    if name.endswith(".nii.gz") or s in MRI_EXT:
        return "mri"
    if s in NIRS_EXT:
        return "nirs"
    if s in TABLE_EXT:
        return "table"
    if s in ARRAY_EXT:
        return "array"
    if s in TEXT_EXT:
        return "text_doc"
    if s in DOC_EXT:
        return "document"
    if s in ARCHIVE_EXT:
        return "archive"
    return "other"

def _promote_trio_files(data_root: Path, output_dir: Path) -> Dict[str, List[str]]:
    """Detect and promote existing trio files to output root."""
    promoted = {"dataset_description": [], "readme": [], "participants": []}
    
    # dataset_description.json
    dd_candidates = list(data_root.glob("**/dataset_description.json"))
    if dd_candidates:
        source = dd_candidates[0]
        dest = output_dir / "dataset_description.json"
        if not dest.exists():
            copy_file(source, dest)
            promoted["dataset_description"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing dataset_description.json")
    
    # README variants
    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    for variant in readme_variants:
        candidates = list(data_root.glob(f"**/{variant}"))
        candidates.extend(list(data_root.glob(f"**/{variant.upper()}")))
        if candidates:
            source = candidates[0]
            dest = output_dir / "README.md"
            if not dest.exists():
                copy_file(source, dest)
                promoted["readme"].append(str(source.relative_to(data_root)))
                info(f"✓ Promoted existing {source.name} → README.md")
            break
    
    # participants.tsv
    parts_candidates = list(data_root.glob("**/participants.tsv"))
    if parts_candidates:
        source = parts_candidates[0]
        dest = output_dir / "participants.tsv"
        if not dest.exists():
            copy_file(source, dest)
            promoted["participants"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing participants.tsv")
    
    return promoted

def _auto_detect_subject_count(all_files: List[str], documents: List[Dict]) -> Dict[str, Any]:
    """
    Automatically detect subject count using multiple strategies.
    
    Returns:
        {
            "method": "deterministic|document_mention|estimation|failed",
            "detected_count": int or None,
            "confidence": "high|medium|low|none",
            "evidence": str,
            "patterns_found": List[str],
            "needs_user_input": bool
        }
    """
    
    # Strategy 1: Deterministic pattern matching
    subject_patterns = [
        (r'sub-(\d+)', "BIDS standard (sub-XX)"),
        (r'sub(\d+)', "BIDS variant (subXX)"),
        (r'subject[_-]?(\d+)', "Subject prefix"),
        (r's(\d+)', "Short form (sXX)"),
        (r'participant[_-]?(\d+)', "Participant prefix"),
        (r'([A-Za-z]+)_sub(\d+)', "Site-prefixed (Site_subXX)"),
        (r'^(\d{2,4})(?:/|\\|_)', "Numeric directories (001/, 025/)"),
    ]
    
    subject_ids: Set[str] = set()
    matched_patterns = []
    
    for filepath in all_files:
        # Check each directory level
        parts = filepath.split('/')
        for part in parts[:3]:  # Check first 3 levels
            for pattern, description in subject_patterns:
                match = re.search(pattern, part, re.IGNORECASE)
                if match:
                    # Extract the numeric part
                    if len(match.groups()) >= 2:
                        subject_id = match.group(2)  # For patterns like Site_subXX
                    else:
                        subject_id = match.group(1)
                    
                    subject_ids.add(subject_id)
                    if description not in matched_patterns:
                        matched_patterns.append(description)
                    break
    
    deterministic_count = len(subject_ids)
    
    # Check confidence based on file count
    avg_files_per_subject = len(all_files) / deterministic_count if deterministic_count > 0 else 0
    
    if deterministic_count > 0 and avg_files_per_subject >= 2:
        # High confidence: found subjects, reasonable file distribution
        return {
            "method": "deterministic",
            "detected_count": deterministic_count,
            "confidence": "high",
            "evidence": f"Detected {deterministic_count} unique subject IDs using patterns: {', '.join(matched_patterns)}",
            "patterns_found": matched_patterns,
            "needs_user_input": False,
            "avg_files_per_subject": round(avg_files_per_subject, 1)
        }
    
    # Strategy 2: Document mention analysis
    doc_mentions = []
    for doc in documents:
        content = doc.get("content", "").lower()
        
        # Look for explicit mentions
        patterns = [
            (r'(\d+)\s+subjects?', "X subjects"),
            (r'(\d+)\s+participants?', "X participants"),
            (r'n\s*=\s*(\d+)', "n=X"),
            (r'sample\s+size[:\s]+(\d+)', "sample size: X"),
            (r'total\s+of\s+(\d+)\s+(?:subjects?|participants?)', "total of X"),
        ]
        
        for pattern, desc in patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    try:
                        count = int(match)
                        if 1 <= count <= 10000:  # Reasonable range
                            doc_mentions.append({
                                "count": count,
                                "source": doc.get("filename", "unknown"),
                                "pattern": desc
                            })
                    except:
                        pass
    
    if doc_mentions:
        # Use the most common mention
        from collections import Counter
        counts = [m["count"] for m in doc_mentions]
        most_common = Counter(counts).most_common(1)[0][0]
        
        evidence_sources = [m["source"] for m in doc_mentions if m["count"] == most_common]
        
        return {
            "method": "document_mention",
            "detected_count": most_common,
            "confidence": "medium",
            "evidence": f"Found mention of {most_common} subjects in documents: {', '.join(set(evidence_sources))}",
            "patterns_found": ["Document text analysis"],
            "needs_user_input": False,
            "document_mentions": doc_mentions
        }
    
    # Strategy 3: Heuristic estimation (low confidence)
    if deterministic_count > 0 and avg_files_per_subject < 2:
        # Detected some subjects but suspicious distribution
        return {
            "method": "estimation",
            "detected_count": deterministic_count,
            "confidence": "low",
            "evidence": f"Detected {deterministic_count} possible subjects, but distribution is unusual (avg {avg_files_per_subject:.1f} files/subject)",
            "patterns_found": matched_patterns,
            "needs_user_input": True,
            "warning": "Low file count per subject - please verify the subject count"
        }
    
    # Strategy 4: Failed detection
    return {
        "method": "failed",
        "detected_count": None,
        "confidence": "none",
        "evidence": "Unable to automatically detect subject count from file structure or documents",
        "patterns_found": [],
        "needs_user_input": True,
        "suggestion": "Please provide subject count using --nsubjects parameter"
    }

def _build_evidence_bundle_internal(data_root: Path, user_n_subjects: Optional[int], 
                                     modality_hint: str, user_text: str,
                                     sample_per_ext: int = 5) -> Dict[str, Any]:
    """
    Build evidence bundle from actual data location.
    
    Args:
        data_root: Actual data directory (from ingest_info)
        user_n_subjects: User-provided subject count (can be None)
    """
    root = Path(data_root)
    files = list_all_files(root)
    info(f"Scanning {len(files)} files in {root}")

    by_ext: Dict[str, List[Path]] = {}
    for p in files:
        key = ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
        by_ext.setdefault(key, []).append(p)

    samples: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    
    for ext, lst in by_ext.items():
        for p in lst[:sample_per_ext]:
            entry = {
                "relpath": str(p.relative_to(root)).replace("\\","/"),
                "size": p.stat().st_size,
                "suffix": ext,
                "kind": detect_kind(p),
                "sha1_head": sha1_head(p)
            }
            
            kind = entry["kind"]
            
            if kind in {"text_doc", "document"}:
                # info(f"Extracting document content: {p.name}")  # Commented out - too verbose
                content = _extract_document_content(p)
                if content:
                    documents.append({
                        "relpath": entry["relpath"],
                        "filename": p.name,
                        "type": ext,
                        "size": entry["size"],
                        "content": content,
                        "purpose": "experimental_protocol_or_metadata"
                    })
                    entry["has_full_content"] = True
                    entry["content_length"] = len(content)
            elif kind == "table":
                entry["table_head"] = _table_head(p)
            
            samples.append(entry)

    # Determine final subject count
    final_count = None
    count_source = None
    subject_detection = None
    
    all_file_paths = [str(p.relative_to(root)).replace("\\","/") for p in files]
    
    if user_n_subjects is not None:
        # User provided count - skip all auto-detection
        final_count = user_n_subjects
        count_source = "user_provided"
        info(f"\nUsing user-provided subject count: {final_count}")
        info("  Skipping auto-detection")
        
        # Create minimal detection record
        subject_detection = {
            "method": "user_provided",
            "detected_count": None,
            "confidence": "user_override",
            "evidence": f"User explicitly provided count: {final_count}",
            "patterns_found": [],
            "needs_user_input": False,
            "skipped_auto_detection": True
        }
    else:
        # No user input - run auto-detection
        info("\nAuto-detecting subject count...")
        subject_detection = _auto_detect_subject_count(all_file_paths, documents)
        
        # Display detection results
        if subject_detection["confidence"] == "high":
            info(f"✓ {subject_detection['evidence']}")
            info(f"  Average files per subject: {subject_detection.get('avg_files_per_subject', 'N/A')}")
        elif subject_detection["confidence"] == "medium":
            info(f"◐ {subject_detection['evidence']}")
        elif subject_detection["confidence"] == "low":
            warn(f"⚠ {subject_detection['evidence']}")
            if subject_detection.get("warning"):
                warn(f"  {subject_detection['warning']}")
        else:
            warn(f"✗ {subject_detection['evidence']}")
            if subject_detection.get("suggestion"):
                warn(f"  {subject_detection['suggestion']}")
        
        # Use auto-detected count
        if subject_detection["detected_count"] is not None:
            final_count = subject_detection["detected_count"]
            count_source = subject_detection["method"]
            info(f"Using auto-detected subject count: {final_count}")
        else:
            warn("⚠ Subject count not determined!")
            warn("  This may cause issues in later stages")
            warn("  Consider re-running with --nsubjects parameter")
            final_count = 1  # Fallback
            count_source = "fallback"

    bundle = {
        "root": str(root),
        "counts_by_ext": {ext: len(lst) for ext, lst in by_ext.items()},
        "samples": samples,
        "documents": documents,
        "all_files": all_file_paths,
        "trio_found": {name: (root / name).exists() for name in TRIO_NAMES},
        "user_hints": {
            "n_subjects": final_count,
            "modality_hint": str(modality_hint) if modality_hint else "",
            "user_text": str(user_text) if user_text else ""
        },
        "subject_detection": {
            **subject_detection,
            "final_count": final_count,
            "count_source": count_source
        },
        "document_summary": {
            "total_documents": len(documents),
            "document_types": list(set(d["type"] for d in documents)) if documents else [],
            "total_text_length": sum(len(d["content"]) for d in documents)
        }
    }
    
    info(f"Extracted {len(documents)} documents")
    if documents:
        info(f"Total text: {bundle['document_summary']['total_text_length']} characters")
    
    return bundle

def build_evidence_bundle(output_dir: Path, user_hints: Dict[str, Any]) -> None:
    """
    Build evidence bundle using ingest_info to locate data.
    
    Auto-detects subject count if not provided.
    """
    output_dir = Path(output_dir)
    
    # Read ingest_info to get actual data path
    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    
    if not ingest_info_path.exists():
        fatal(f"Ingest info not found: {ingest_info_path}")
        fatal("Run 'ingest' step first")
        return
    
    ingest_info = read_json(ingest_info_path)
    
    # Get actual data path
    actual_data_path = ingest_info.get("actual_data_path")
    if not actual_data_path:
        # Fallback to old behavior
        actual_data_path = ingest_info.get("staging_dir")
    
    if not actual_data_path:
        fatal("Cannot determine data location from ingest_info")
        return
    
    data_root = Path(actual_data_path)
    
    if not data_root.exists():
        fatal(f"Data directory not found: {data_root}")
        return
    
    info(f"Using data from: {data_root}")
    
    # Promote trio files
    info("Checking for existing trio files in input data...")
    promoted = _promote_trio_files(data_root, output_dir)
    
    total = sum(len(f) for f in promoted.values())
    if total > 0:
        info(f"Promoted {total} trio file(s) from input data")
    else:
        info("No existing trio files found in input data")
    
    # Build evidence bundle with auto-detection
    bundle = _build_evidence_bundle_internal(
        data_root=data_root,
        user_n_subjects=user_hints.get("n_subjects"),  # Can be None
        modality_hint=user_hints.get("modality_hint", ""),
        user_text=user_hints.get("user_text", "")
    )
    
    bundle["trio_promoted"] = promoted
    bundle["data_source"] = {
        "type": ingest_info.get("input_type"),
        "original_path": ingest_info.get("input_path"),
        "actual_path": str(data_root)
    }
    
    write_json(output_dir / "_staging" / "evidence_bundle.json", bundle)
    info(f"✓ Evidence bundle saved")
    
    # Display final summary
    info("\n=== Evidence Bundle Summary ===")
    info(f"Total files: {len(bundle['all_files'])}")
    info(f"File types: {len(bundle['counts_by_ext'])}")
    info(f"Documents extracted: {len(bundle['documents'])}")
    
    subject_det = bundle["subject_detection"]
    info(f"Subject count: {subject_det['final_count']} (source: {subject_det['count_source']})")
    
    if not subject_det.get("skipped_auto_detection", False):
        info(f"Detection confidence: {subject_det['confidence']}")
        
        if subject_det.get("needs_user_input"):
            warn("\n⚠ RECOMMENDATION: Verify subject count and re-run if needed")
            warn("  Use: --nsubjects <count> to override")
