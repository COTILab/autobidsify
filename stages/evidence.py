# evidence.py v2
# Evidence bundle builder with universal analysis engine

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple  # CRITICAL: Added Tuple!
from collections import defaultdict
from utils import list_all_files, write_json, sha1_head, warn, info, fatal, copy_file, read_json
from constants import MAX_TEXT_SIZE, MAX_PDF_SIZE, MAX_DOCX_SIZE, MAX_PDF_PAGES
from universal_core import FileStructureAnalyzer
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
    promoted = {"dataset_description": [], "readme": [], "participants": []}
    
    dd_candidates = list(data_root.glob("**/dataset_description.json"))
    if dd_candidates:
        source = dd_candidates[0]
        dest = output_dir / "dataset_description.json"
        if not dest.exists():
            copy_file(source, dest)
            promoted["dataset_description"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing dataset_description.json")
    
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
    
    parts_candidates = list(data_root.glob("**/participants.tsv"))
    if parts_candidates:
        source = parts_candidates[0]
        dest = output_dir / "participants.tsv"
        if not dest.exists():
            copy_file(source, dest)
            promoted["participants"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing participants.tsv")
    
    return promoted

def _intelligent_file_sampling(files_by_ext: Dict[str, List[Path]], 
                                target_samples_per_ext: int = 5) -> Tuple[List[Path], Dict]:
    samples = []
    pattern_summary = {}
    
    for ext, file_list in files_by_ext.items():
        pattern_groups = defaultdict(list)
        
        for filepath in file_list:
            filename = filepath.name
            pattern = re.sub(r'\s*\(\d+\)', '', filename)
            pattern = re.sub(r'\d+', 'N', pattern)
            pattern_groups[pattern].append(filepath)
        
        n_patterns = len(pattern_groups)
        samples_per_pattern = max(1, target_samples_per_ext // n_patterns) if n_patterns > 0 else target_samples_per_ext
        
        ext_samples = []
        pattern_info = []
        
        for pattern, files in sorted(pattern_groups.items()):
            n_samples = min(len(files), samples_per_pattern)
            group_samples = files[:n_samples]
            ext_samples.extend(group_samples)
            
            pattern_info.append({
                "pattern": pattern,
                "total_files": len(files),
                "sampled": n_samples,
                "example_files": [f.name for f in group_samples[:2]]
            })
        
        if len(ext_samples) < target_samples_per_ext:
            remaining = target_samples_per_ext - len(ext_samples)
            large_groups = sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            for pattern, files in large_groups:
                if remaining <= 0:
                    break
                
                already_sampled = [f for f in ext_samples if f in files]
                available = [f for f in files if f not in already_sampled]
                
                n_additional = min(remaining, len(available))
                if n_additional > 0:
                    ext_samples.extend(available[:n_additional])
                    remaining -= n_additional
                    
                    for p_info in pattern_info:
                        if p_info["pattern"] == pattern:
                            p_info["sampled"] += n_additional
                            break
        
        samples.extend(ext_samples)
        
        pattern_summary[ext] = {
            "total_patterns": n_patterns,
            "total_files": len(file_list),
            "sampled_files": len(ext_samples),
            "patterns": pattern_info
        }
    
    return samples, pattern_summary

def _build_evidence_bundle_internal(data_root: Path, user_n_subjects: Optional[int], 
                                     modality_hint: str, user_text: str,
                                     sample_per_ext: int = 5) -> Dict[str, Any]:
    root = Path(data_root)
    files = list_all_files(root)
    info(f"Scanning {len(files)} files in {root}")

    all_file_paths = [str(p.relative_to(root)).replace("\\","/") for p in files]
    
    info("Analyzing file structure with universal engine...")
    analyzer = FileStructureAnalyzer(all_file_paths)
    
    dir_structure = analyzer.analyze_directory_structure()
    info(f"  Directory structure: {dir_structure['max_depth']} levels, template: {dir_structure['structure_template']}")
    info(f"  Unique directories: {dir_structure['total_unique_dirs']}")
    
    subject_detection_result = analyzer.detect_subject_identifiers(user_n_subjects)
    
    if subject_detection_result["best_candidate"]:
        best = subject_detection_result["best_candidate"]
        info(f"  Best subject pattern: {best['pattern_display']}")
        info(f"  Detected: {best['count']} subjects (confidence: {subject_detection_result['confidence']})")
        info(f"  Avg files/subject: {best['avg_files_per_subject']:.1f}")
    else:
        warn("  ⚠ No subject pattern detected")
    
    duplicates = analyzer.detect_duplicate_filenames()
    if duplicates:
        info(f"  Found {len(duplicates)} duplicate filenames across different paths")
        for fname, paths in list(duplicates.items())[:2]:
            info(f"    '{fname}' appears in {len(paths)} locations")
    
    tree_summary = analyzer.build_directory_tree_summary(max_subjects=50)
    info(f"  Structure summary: {tree_summary['sampled_subjects']}/{tree_summary['total_subjects_detected']} subjects")
    
    by_ext: Dict[str, List[Path]] = {}
    for p in files:
        key = ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
        by_ext.setdefault(key, []).append(p)
    
    info("\nSampling files for document extraction...")
    sampled_files, pattern_summary = _intelligent_file_sampling(by_ext, sample_per_ext)
    
    info("Sampling summary:")
    for ext, summary in pattern_summary.items():
        info(f"  {ext}: {summary['total_patterns']} patterns, {summary['sampled_files']}/{summary['total_files']} files")
    
    samples: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    
    for p in sampled_files:
        ext = ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
        
        entry = {
            "relpath": str(p.relative_to(root)).replace("\\","/"),
            "size": p.stat().st_size,
            "suffix": ext,
            "kind": detect_kind(p),
            "sha1_head": sha1_head(p)
        }
        
        kind = entry["kind"]
        
        if kind in {"text_doc", "document"}:
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
    
    if user_n_subjects is not None:
        final_count = user_n_subjects
        count_source = "user_provided"
        info(f"\nUsing user-provided subject count: {final_count}")
    elif subject_detection_result["best_candidate"]:
        final_count = subject_detection_result["best_candidate"]["count"]
        count_source = "auto_detected"
        info(f"\nUsing auto-detected subject count: {final_count}")
    else:
        final_count = 1
        count_source = "fallback"
        warn("\n⚠ Could not detect subject count, using fallback: 1")
    
    bundle = {
        "root": str(root),
        "counts_by_ext": {ext: len(lst) for ext, lst in by_ext.items()},
        "samples": samples,
        "documents": documents,
        "all_files": all_file_paths,
        "trio_found": {name: (root / name).exists() for name in TRIO_NAMES},
        
        "structure_analysis": {
            "directory_structure": dir_structure,
            "subject_detection": subject_detection_result,
            "duplicate_files": {k: v for k, v in list(duplicates.items())[:20]},
            "tree_summary_for_llm": tree_summary,
            "analyzer_confidence": subject_detection_result["confidence"]
        },
        
        "user_hints": {
            "n_subjects": final_count,
            "modality_hint": str(modality_hint) if modality_hint else "",
            "user_text": str(user_text) if user_text else ""
        },
        
        "subject_detection": {
            "method": "universal_analyzer",
            "detected_count": subject_detection_result["best_candidate"]["count"] if subject_detection_result["best_candidate"] else None,
            "confidence": subject_detection_result["confidence"],
            "final_count": final_count,
            "count_source": count_source,
            "best_pattern": subject_detection_result["best_candidate"]["pattern_display"] if subject_detection_result["best_candidate"] else None
        },
        
        "document_summary": {
            "total_documents": len(documents),
            "document_types": list(set(d["type"] for d in documents)),
            "total_text_length": sum(len(d["content"]) for d in documents)
        },
        
        "sampling_strategy": {
            "method": "pattern_based",
            "target_per_ext": sample_per_ext,
            "total_patterns_detected": sum(s["total_patterns"] for s in pattern_summary.values()),
            "pattern_summary": pattern_summary
        }
    }
    
    info(f"\nExtracted {len(documents)} documents")
    if documents:
        info(f"Total document text: {bundle['document_summary']['total_text_length']:,} characters")
    
    info("\n=== Universal Analysis Summary ===")
    info(f"Subject detection: {bundle['subject_detection']['confidence']} confidence")
    if subject_detection_result["best_candidate"]:
        info(f"  Pattern: {subject_detection_result['best_candidate']['pattern_display']}")
        info(f"  Count: {subject_detection_result['best_candidate']['count']}")
    info(f"Duplicate handling: {len(duplicates)} duplicate filenames detected")
    info(f"Sampling: {sum(s['total_patterns'] for s in pattern_summary.values())} unique patterns")
    
    return bundle

def build_evidence_bundle(output_dir: Path, user_hints: Dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    
    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    
    if not ingest_info_path.exists():
        fatal(f"Ingest info not found: {ingest_info_path}")
        fatal("Run 'ingest' step first")
        return
    
    ingest_info = read_json(ingest_info_path)
    
    actual_data_path = ingest_info.get("actual_data_path")
    if not actual_data_path:
        actual_data_path = ingest_info.get("staging_dir")
    
    if not actual_data_path:
        fatal("Cannot determine data location from ingest_info")
        return
    
    data_root = Path(actual_data_path)
    
    if not data_root.exists():
        fatal(f"Data directory not found: {data_root}")
        return
    
    info(f"Using data from: {data_root}")
    
    info("\nChecking for existing trio files...")
    promoted = _promote_trio_files(data_root, output_dir)
    
    total = sum(len(f) for f in promoted.values())
    if total > 0:
        info(f"Promoted {total} trio file(s)")
    else:
        info("No existing trio files found")
    
    bundle = _build_evidence_bundle_internal(
        data_root=data_root,
        user_n_subjects=user_hints.get("n_subjects"),
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
    info(f"\n✓ Evidence bundle saved")
    
    info("\n=== Evidence Bundle Summary ===")
    info(f"Total files: {len(bundle['all_files'])}")
    info(f"File types: {len(bundle['counts_by_ext'])}")
    info(f"Subject count: {bundle['subject_detection']['final_count']} (source: {bundle['subject_detection']['count_source']})")
    info(f"Detection confidence: {bundle['subject_detection']['confidence']}")
