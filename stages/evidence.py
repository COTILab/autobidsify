# evidence.py - COMPLETE VERSION with participant metadata evidence collection
# MODIFIED: Added JNIfTI detection support

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from utils import list_all_files, write_json, sha1_head, warn, info, fatal, copy_file, read_json
from constants import MAX_TEXT_SIZE, MAX_PDF_SIZE, MAX_DOCX_SIZE, MAX_PDF_PAGES
from universal_core import FileStructureAnalyzer
from filename_tokenizer import analyze_filenames_for_subjects
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

# NEW: JNIfTI extensions
JNIFTI_EXT = {".jnii", ".bnii"}

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
    """
    Detect file type/kind for classification.
    
    MODIFIED: Added JNIfTI detection (.jnii, .bnii files)
    
    Args:
        p: File path
    
    Returns:
        File kind: 'jnifti', 'mri', 'nirs', 'table', 'array', 
                   'text_doc', 'document', 'archive', 'user_trio', 'other'
    """
    s = p.suffix.lower()
    name = p.name.lower()
    
    if _is_trio_file(name):
        return "user_trio"
    
    # NEW: JNIfTI file detection - MUST be before regular MRI check
    if s in JNIFTI_EXT:
        return "jnifti"
    
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
                                target_samples_per_ext: int = 5,
                                ensure_full_coverage: bool = False) -> Tuple[List[Path], Dict]:
    samples = []
    pattern_summary = {}
    
    for ext, file_list in files_by_ext.items():
        pattern_groups = defaultdict(list)
        
        for filepath in file_list:
            filename = filepath.name
            pattern = re.sub(r'\d+', 'N', filename)
            pattern = re.sub(r'\s*\([^)]*\)', '', pattern)
            pattern_groups[pattern].append(filepath)
        
        n_patterns = len(pattern_groups)
        
        if ensure_full_coverage:
            ext_samples = []
            for pattern, files in sorted(pattern_groups.items()):
                ext_samples.append(files[0])
            
            info(f"  {ext}: Full coverage mode - {len(ext_samples)} patterns sampled")
        else:
            samples_per_pattern = max(1, target_samples_per_ext // n_patterns) if n_patterns > 0 else target_samples_per_ext
            
            ext_samples = []
            for pattern, files in sorted(pattern_groups.items()):
                n_samples = min(len(files), samples_per_pattern)
                group_samples = files[:n_samples]
                ext_samples.extend(group_samples)
            
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
        
        samples.extend(ext_samples)
        
        pattern_info = []
        for pattern, files in sorted(pattern_groups.items()):
            sampled_count = sum(1 for f in ext_samples if f in files)
            pattern_info.append({
                "pattern": pattern,
                "total_files": len(files),
                "sampled": sampled_count,
                "example_files": [f.name for f in files[:2]]
            })
        
        pattern_summary[ext] = {
            "total_patterns": n_patterns,
            "total_files": len(file_list),
            "sampled_files": len(ext_samples),
            "patterns": pattern_info
        }
    
    return samples, pattern_summary

def _collect_participant_metadata_evidence(data_root: Path, 
                                           all_files: List[str],
                                           documents: List[Dict]) -> Dict[str, Any]:
    """Collect all evidence that may help infer participant metadata."""
    evidence = {}
    
    info("  [Evidence 1/5] Scanning for explicit metadata files...")
    metadata_files = []
    
    metadata_patterns = [
        '**/participants.*', '**/subjects.*', '**/metadata.*', 
        '**/demographics.*', '**/phenotype.*',
        '**/participant_data.*', '**/subject_info.*'
    ]
    
    for pattern in metadata_patterns:
        try:
            for match in data_root.glob(pattern):
                if match.is_file() and match.suffix in ['.csv', '.tsv', '.json', '.txt', '.xlsx']:
                    metadata_files.append({
                        "filename": match.name,
                        "path": str(match.relative_to(data_root)),
                        "extension": match.suffix,
                        "size_bytes": match.stat().st_size
                    })
        except Exception as e:
            warn(f"    Error scanning pattern {pattern}: {e}")
            continue
    
    if metadata_files:
        evidence["explicit_metadata_files"] = {
            "found": True,
            "count": len(metadata_files),
            "files": metadata_files,
            "note": "These files may contain participant demographics"
        }
        info(f"    ✓ Found {len(metadata_files)} potential metadata file(s)")
    else:
        evidence["explicit_metadata_files"] = {"found": False}
        info(f"    - No explicit metadata files found")
    
    info("  [Evidence 2/5] Sampling DICOM headers...")
    dicom_patient_info = []
    dicom_files = [f for f in all_files if f.lower().endswith('.dcm')]
    
    if dicom_files:
        sample_size = min(20, len(dicom_files))
        step = max(1, len(dicom_files) // sample_size)
        sampled_indices = list(range(0, len(dicom_files), step))[:sample_size]
        sampled_files = [dicom_files[i] for i in sampled_indices]
        
        try:
            import pydicom
            pydicom_available = True
        except ImportError:
            pydicom_available = False
            warn("    pydicom not installed, skipping DICOM header analysis")
        
        if pydicom_available:
            for dcm_path_str in sampled_files:
                dcm_path = data_root / dcm_path_str
                try:
                    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                    
                    patient_info = {"filename": dcm_path.name}
                    
                    dicom_fields = [
                        ('PatientID', 'PatientID'),
                        ('PatientSex', 'PatientSex'),
                        ('PatientAge', 'PatientAge'),
                        ('PatientName', 'PatientName'),
                        ('StudyDescription', 'StudyDescription'),
                        ('PatientBirthDate', 'PatientBirthDate'),
                        ('PatientWeight', 'PatientWeight')
                    ]
                    
                    for attr_name, field_name in dicom_fields:
                        try:
                            value = getattr(ds, attr_name, None)
                            if value is not None and str(value).strip():
                                patient_info[field_name] = str(value).strip()
                        except:
                            continue
                    
                    if len(patient_info) > 1:
                        dicom_patient_info.append(patient_info)
                        
                except Exception:
                    continue
        
        if dicom_patient_info:
            evidence["dicom_headers"] = {
                "found": True,
                "sampled_count": len(dicom_patient_info),
                "total_dicom_files": len(dicom_files),
                "samples": dicom_patient_info[:10],
                "note": "DICOM headers may contain PatientSex, PatientAge, PatientID"
            }
            info(f"    ✓ Extracted headers from {len(dicom_patient_info)}/{sample_size} DICOM files")
        else:
            evidence["dicom_headers"] = {
                "found": False,
                "total_dicom_files": len(dicom_files),
                "note": "DICOM files exist but no patient info extracted"
            }
            info(f"    - DICOM files present ({len(dicom_files)}) but no patient info found")
    else:
        evidence["dicom_headers"] = {"found": False}
        info(f"    - No DICOM files in dataset")
    
    info("  [Evidence 3/5] Analyzing filename semantic patterns...")
    filename_patterns = {
        "gender_keywords": [],
        "age_patterns": [],
        "group_keywords": []
    }
    
    gender_keywords = [
        'male', 'female', 'man', 'woman', 'boy', 'girl',
        '_m_', '_f_', '_m.', '_f.',
        'VHM', 'VHF',
        'male_', 'female_',
        '_male', '_female'
    ]
    
    age_patterns_regex = [
        r'\d{2}yo', r'\d{2}y\b', r'age\d{2}', r'_\d{2}_', r'y\d{2}',
    ]
    
    group_keywords = [
        'patient', 'control', 'healthy', 'disease',
        'hc', 'pt', 'ctrl', 'case', 'normal',
        'treated', 'untreated'
    ]
    
    sample_files_for_pattern = all_files[:200]
    
    for filepath in sample_files_for_pattern:
        filename = filepath.split('/')[-1]
        filename_lower = filename.lower()
        
        for kw in gender_keywords:
            if kw.lower() in filename_lower:
                filename_patterns["gender_keywords"].append({
                    "keyword": kw,
                    "filename": filename
                })
                break
        
        for pattern in age_patterns_regex:
            if re.search(pattern, filename_lower):
                filename_patterns["age_patterns"].append({
                    "pattern": pattern,
                    "filename": filename
                })
                break
        
        for kw in group_keywords:
            if kw in filename_lower:
                filename_patterns["group_keywords"].append({
                    "keyword": kw,
                    "filename": filename
                })
                break
    
    for key in filename_patterns:
        seen = set()
        unique_patterns = []
        for item in filename_patterns[key]:
            identifier = item.get('keyword') or item.get('pattern')
            if identifier not in seen:
                seen.add(identifier)
                unique_patterns.append(item)
        filename_patterns[key] = unique_patterns[:10]
    
    patterns_found = any(filename_patterns.values())
    
    if patterns_found:
        evidence["filename_semantic_patterns"] = {
            "found": True,
            "patterns": filename_patterns,
            "note": "Filenames may contain demographic keywords"
        }
        
        total_hints = sum(len(v) for v in filename_patterns.values())
        info(f"    ✓ Found {total_hints} semantic patterns in filenames")
    else:
        evidence["filename_semantic_patterns"] = {"found": False}
        info(f"    - No semantic patterns found in filenames")
    
    info("  [Evidence 4/5] Searching documents for demographic keywords...")
    demographic_keywords_in_docs = []
    
    demographic_terms = [
        'male', 'female', 'sex', 'gender', 'age', 'years old',
        'patient', 'control', 'healthy', 'diagnosis', 'participants',
        'subjects', 'volunteers', 'cohort', 'population',
        'cadaver', 'human', 'adult', 'child'
    ]
    
    for doc in documents:
        content = doc.get('content', '').lower()
        doc_name = doc.get('filename', 'unknown')
        
        found_terms = []
        
        for term in demographic_terms:
            if term in content:
                term_index = content.find(term)
                if term_index != -1:
                    start = max(0, term_index - 100)
                    end = min(len(content), term_index + 100)
                    snippet = content[start:end].strip()
                    snippet = ' '.join(snippet.split())
                    
                    found_terms.append({
                        "term": term,
                        "context_snippet": snippet[:200]
                    })
        
        if found_terms:
            demographic_keywords_in_docs.append({
                "document": doc_name,
                "found_terms": found_terms[:5]
            })
    
    if demographic_keywords_in_docs:
        evidence["document_demographic_keywords"] = {
            "found": True,
            "documents_with_keywords": len(demographic_keywords_in_docs),
            "details": demographic_keywords_in_docs[:5],
            "note": "Documents mention demographic terms"
        }
        info(f"    ✓ Found demographic keywords in {len(demographic_keywords_in_docs)} document(s)")
    else:
        evidence["document_demographic_keywords"] = {"found": False}
        info(f"    - No demographic keywords found in documents")
    
    info("  [Evidence 5/5] Analyzing subject grouping patterns...")
    
    from filename_tokenizer import FilenamePatternAnalyzer
    
    filenames = [f.split('/')[-1] for f in all_files]
    analyzer = FilenamePatternAnalyzer(filenames)
    stats = analyzer.analyze_token_statistics()
    
    dominant_prefixes = stats.get('dominant_prefixes', [])
    
    if len(dominant_prefixes) == 2:
        p1, p2 = dominant_prefixes[0], dominant_prefixes[1]
        ratio = min(p1['percentage'], p2['percentage']) / max(p1['percentage'], p2['percentage'])
        
        if ratio > 0.8:
            evidence["balanced_prefix_distribution"] = {
                "found": True,
                "prefix_1": p1['prefix'],
                "prefix_1_percentage": p1['percentage'],
                "prefix_1_count": p1['count'],
                "prefix_2": p2['prefix'],
                "prefix_2_percentage": p2['percentage'],
                "prefix_2_count": p2['count'],
                "distribution_ratio": round(ratio, 2),
                "note": "Two balanced groups may indicate gender/group split"
            }
            info(f"    ✓ Found balanced distribution: {p1['prefix']} ({p1['percentage']}%) vs {p2['prefix']} ({p2['percentage']}%)")
        else:
            evidence["balanced_prefix_distribution"] = {"found": False}
            info(f"    - Two prefixes found but not balanced ({ratio:.2f})")
    else:
        evidence["balanced_prefix_distribution"] = {"found": False}
        info(f"    - No balanced distribution pattern detected")
    
    evidence_types_found = sum(1 for v in evidence.values() 
                               if isinstance(v, dict) and v.get('found'))
    
    evidence["summary"] = {
        "total_evidence_types_found": evidence_types_found,
        "evidence_types": [k for k, v in evidence.items() 
                          if isinstance(v, dict) and v.get('found')]
    }
    
    info(f"\n  Summary: Found {evidence_types_found}/5 types of evidence")
    
    return evidence

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
        warn("  ⚠ No subject pattern detected from directory structure")
    
    duplicates = analyzer.detect_duplicate_filenames()
    if duplicates:
        info(f"  Found {len(duplicates)} duplicate filenames across different paths")
        for fname, paths in list(duplicates.items())[:2]:
            info(f"    '{fname}' appears in {len(paths)} locations")
    
    tree_summary = analyzer.build_directory_tree_summary(max_subjects=50)
    info(f"  Structure summary: {tree_summary['sampled_subjects']}/{tree_summary['total_subjects_detected']} subjects")
    
    info("\nAnalyzing filename token patterns...")
    filename_analysis = analyze_filenames_for_subjects(all_file_paths, {
        "n_subjects": user_n_subjects,
        "user_text": user_text
    })
    
    info(f"  Token-based analysis: {filename_analysis['confidence']} confidence")
    info(f"  {filename_analysis['recommendation']}")
    
    dominant_prefixes = filename_analysis['python_statistics'].get('dominant_prefixes', [])
    if dominant_prefixes:
        info(f"  Dominant filename prefixes:")
        for p in dominant_prefixes[:5]:
            info(f"    '{p['prefix']}': {p['count']} files ({p['percentage']}%)")
    
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
    
    info("\n=== Collecting Participant Metadata Evidence ===")
    participant_evidence = _collect_participant_metadata_evidence(
        data_root=root,
        all_files=all_file_paths,
        documents=documents
    )
    
    evidence_count = participant_evidence['summary']['total_evidence_types_found']
    info(f"\n✓ Evidence collection complete: {evidence_count}/5 types found")
    
    path_based_count = subject_detection_result["best_candidate"]["count"] if subject_detection_result["best_candidate"] else 0
    path_based_confidence = subject_detection_result["confidence"]
    
    filename_based_count = len(filename_analysis['python_statistics'].get('dominant_prefixes', []))
    filename_based_confidence = filename_analysis['confidence']
    
    if user_n_subjects is not None:
        final_count = user_n_subjects
        count_source = "user_provided"
        info(f"\nUsing user-provided subject count: {final_count}")
    elif path_based_confidence == "high":
        final_count = path_based_count
        count_source = "path_based_high_confidence"
        info(f"\nUsing path-based detection (high confidence): {final_count}")
    elif filename_based_confidence in ["high", "medium"] and path_based_count == 0:
        final_count = filename_based_count
        count_source = "filename_based"
        info(f"\nUsing filename token analysis (path-based found 0): {final_count}")
    elif path_based_count > 0:
        final_count = path_based_count
        count_source = "path_based"
        info(f"\nUsing path-based detection: {final_count}")
    else:
        final_count = 1
        count_source = "fallback"
        warn("\n⚠ Could not detect subject count from either path or filename, using fallback: 1")
    
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
        
        "filename_analysis": filename_analysis,
        
        "participant_metadata_evidence": participant_evidence,
        
        "user_hints": {
            "n_subjects": final_count,
            "modality_hint": str(modality_hint) if modality_hint else "",
            "user_text": str(user_text) if user_text else ""
        },
        
        "subject_detection": {
            "method": "hybrid_analysis",
            "path_based_count": path_based_count,
            "path_based_confidence": path_based_confidence,
            "filename_based_count": filename_based_count,
            "filename_based_confidence": filename_based_confidence,
            "final_count": final_count,
            "count_source": count_source,
            "best_pattern": subject_detection_result["best_candidate"]["pattern_display"] if subject_detection_result["best_candidate"] else "none"
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
    info(f"Subject detection (hybrid):")
    info(f"  Path-based: {path_based_count} subjects ({path_based_confidence} confidence)")
    info(f"  Filename-based: {filename_based_count} subjects ({filename_based_confidence} confidence)")
    info(f"  Final decision: {final_count} subjects (source: {count_source})")
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
    info(f"Detection confidence: hybrid ({bundle['subject_detection']['path_based_confidence']} path, {bundle['subject_detection']['filename_based_confidence']} filename)")
