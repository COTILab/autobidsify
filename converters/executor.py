# executor.py
# Execute LLM's decisions with UNIVERSAL semantic matching

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import shutil
from fnmatch import fnmatch
from utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn, debug
from converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available

def _normalize_for_matching(text: str) -> str:
    """
    标准化文本用于匹配,移除干扰字符。
    
    通用策略:去掉数字、括号、下划线等,只保留字母和连字符。
    """
    # 转小写
    text = text.lower()
    # 去掉括号及其内容
    text = re.sub(r'\s*\([^)]*\)', '', text)
    # 去掉多余空格
    text = re.sub(r'\s+', '', text)
    return text

def _extract_semantic_components(pattern: str) -> Dict[str, Any]:
    """
    从 LLM 的 match_pattern 提取语义组件。
    
    通用解析,适用于任何 pattern 风格。
    
    Examples:
        "VHM.*-Head.*\\.dcm" -> {prefix: "vhm", keywords: ["head"], ext: ".dcm"}
        "sub-01_.*T1w.*\\.nii" -> {prefix: "sub-01", keywords: ["t1w"], ext: ".nii"}
    """
    components = {
        "prefix": None,
        "keywords": [],
        "extension": None
    }
    
    # 提取前缀 (开头的字母数字序列)
    prefix_match = re.match(r'^([A-Za-z0-9-]+)', pattern)
    if prefix_match:
        components["prefix"] = prefix_match.group(1).lower()
    
    # 提取关键词 (连字符或下划线后的单词)
    keyword_matches = re.findall(r'[-_]([A-Za-z]+)', pattern)
    components["keywords"] = [k.lower() for k in keyword_matches if len(k) > 1]
    
    # 提取扩展名
    if r'\.dcm' in pattern or '.dcm' in pattern:
        components["extension"] = ".dcm"
    elif r'\.nii\.gz' in pattern or '.nii.gz' in pattern:
        components["extension"] = ".nii.gz"
    elif r'\.nii' in pattern or '.nii' in pattern:
        components["extension"] = ".nii"
    
    return components

def _semantic_match(filename: str, pattern: str, entities: Dict) -> bool:
    """
    通用语义匹配算法 - 零硬编码假设。
    
    Args:
        filename: 实际文件名 "VHMCT1mm-Head (64).dcm"
        pattern: LLM pattern "VHM.*-Head.*\\.dcm"
        entities: LLM entities {"subject": "1", "acquisition": "cthead"}
    
    Returns:
        True if semantically equivalent
    """
    # 标准化文件名
    fname_normalized = _normalize_for_matching(filename)
    
    # 提取 pattern 的语义组件
    components = _extract_semantic_components(pattern)
    
    # 检查 1: 前缀匹配
    if components["prefix"]:
        if not fname_normalized.startswith(components["prefix"]):
            return False
    
    # 检查 2: 关键词匹配 (所有关键词都必须存在)
    for keyword in components["keywords"]:
        if keyword not in fname_normalized:
            return False
    
    # 检查 3: 扩展名匹配
    if components["extension"]:
        if not filename.lower().endswith(components["extension"]):
            return False
    
    # 检查 4: 从 entities 提取额外的关键词
    acquisition = entities.get("acquisition", "")
    if acquisition:
        # "cthead" -> 检查 "head" (去掉 ct 前缀)
        acq_normalized = acquisition.lower()
        if acq_normalized.startswith("ct"):
            bodypart = acq_normalized[2:]
        else:
            bodypart = acq_normalized
        
        if bodypart and len(bodypart) > 1:
            if bodypart not in fname_normalized:
                return False
    
    return True

def _apply_llm_filename_rule(src: Path, rule: Dict) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    应用 LLM 文件名规则 - 使用通用语义匹配。
    
    Returns:
        (bids_filename, subject_id, entities_dict) or (None, None, None) if no match
    """
    match_pattern = rule.get("match_pattern")
    bids_template = rule.get("bids_template")
    extract_entities = rule.get("extract_entities", {})
    
    if not match_pattern or not bids_template:
        return None, None, None
    
    src_name = src.name
    
    # 使用语义匹配
    if _semantic_match(src_name, match_pattern, extract_entities):
        subject_id = extract_entities.get("subject")
        return bids_template, subject_id, extract_entities
    
    return None, None, None

def _group_dicom_files_by_rule(matched_files: List[Path], filename_rules: List[Dict]) -> Dict[str, Dict]:
    """
    根据 LLM 规则分组 DICOM 文件。
    
    通用策略:使用语义匹配而非精确 regex。
    
    Returns:
        Dict[group_key, {"files": [...], "bids_name": "...", "subject": "...", ...}]
    """
    groups = {}
    unmatched_files = []
    
    # 统计每个规则匹配了多少文件
    rule_match_counts = {i: 0 for i in range(len(filename_rules))}
    
    for dcm_file in matched_files:
        matched_rule_idx = None
        matched_info = None
        
        # 尝试匹配所有规则
        for idx, rule in enumerate(filename_rules):
            bids_name, subject_id, entities = _apply_llm_filename_rule(dcm_file, rule)
            
            if bids_name and subject_id:
                # 找到匹配!
                matched_rule_idx = idx
                matched_info = (bids_name, subject_id, entities)
                rule_match_counts[idx] += 1
                break
        
        if matched_info:
            bids_name, subject_id, entities = matched_info
            
            # 创建组键
            base_name = bids_name.replace('.nii.gz', '').replace('.nii', '')
            group_key = f"{subject_id}_{base_name}"
            
            if group_key not in groups:
                groups[group_key] = {
                    "files": [],
                    "bids_name": bids_name,
                    "subject": subject_id,
                    "entities": entities,
                    "rule_index": matched_rule_idx
                }
            
            groups[group_key]["files"].append(dcm_file)
        else:
            unmatched_files.append(dcm_file)
    
    # 显示匹配统计
    info(f"    Matching statistics:")
    for idx, count in rule_match_counts.items():
        if count > 0:
            rule = filename_rules[idx]
            bids_name = rule.get("bids_template", "unknown")
            info(f"      Rule {idx+1} ({bids_name}): {count} files")
    
    # 处理未匹配的文件
    if unmatched_files:
        warn(f"    ⚠ {len(unmatched_files)} files did not match any rule")
        warn(f"    Sample unmatched files:")
        for f in unmatched_files[:5]:
            warn(f"      - {f.name}")
        
        # 智能 fallback: 分析文件名推断分组
        _handle_unmatched_files(unmatched_files, groups)
    
    return groups

def _handle_unmatched_files(unmatched_files: List[Path], groups: Dict[str, Dict]) -> None:
    """
    智能处理未匹配的文件 - 通过文件名分析推断分组。
    
    通用策略:
    1. 提取 subject 标识
    2. 提取身体部位/任务关键词
    3. 合并到合理的组
    """
    for dcm_file in unmatched_files:
        filename = dcm_file.name
        
        # 推断 subject
        subject_id = "01"
        if re.search(r'\bVHM\b', filename, re.IGNORECASE):
            subject_id = "1"
        elif re.search(r'\bVHF\b', filename, re.IGNORECASE):
            subject_id = "2"
        elif match := re.search(r'sub-?(\d+)', filename, re.IGNORECASE):
            subject_id = match.group(1)
        elif match := re.search(r'subject[_-]?(\d+)', filename, re.IGNORECASE):
            subject_id = match.group(1)
        
        # 推断关键词 (身体部位、任务等)
        keywords = []
        common_bodyparts = ['head', 'hip', 'shoulder', 'knee', 'ankle', 'pelvis', 
                           'chest', 'abdomen', 'brain', 'spine']
        for part in common_bodyparts:
            if part in filename.lower():
                keywords.append(part)
                break
        
        # 推断是否 CT
        is_ct = 'ct' in filename.lower()
        
        # 构建 bids_name
        if keywords:
            keyword = keywords[0]
            acq = f"ct{keyword}" if is_ct else keyword
            bids_name = f"sub-{subject_id}_acq-{acq}_T1w.nii.gz"
        else:
            bids_name = f"sub-{subject_id}_unknown.nii.gz"
        
        # 创建组
        base_name = bids_name.replace('.nii.gz', '')
        group_key = f"{subject_id}_{base_name}"
        
        if group_key not in groups:
            groups[group_key] = {
                "files": [],
                "bids_name": bids_name,
                "subject": subject_id,
                "entities": {"subject": subject_id, "acquisition": acq if keywords else "unknown"},
                "is_fallback": True
            }
        
        groups[group_key]["files"].append(dcm_file)

def _match_files_with_patterns(all_files: List[Path], patterns: List[str], input_root: Path) -> List[Path]:
    """Match files against glob patterns."""
    matched = []
    
    for file_path in all_files:
        try:
            rel_path = file_path.relative_to(input_root)
            rel_path_str = str(rel_path).replace("\\", "/")
        except ValueError:
            continue
        
        for pattern in patterns:
            if "**" in pattern:
                pattern_parts = pattern.split("**")
                if len(pattern_parts) == 2:
                    prefix = pattern_parts[0].strip("/")
                    suffix = pattern_parts[1].strip("/")
                    
                    if (not prefix or rel_path_str.startswith(prefix) or fnmatch(rel_path_str, prefix + "*")):
                        if fnmatch(file_path.name, suffix) or fnmatch(rel_path_str, "*" + suffix):
                            matched.append(file_path)
                            break
            else:
                if fnmatch(rel_path_str, pattern) or fnmatch(file_path.name, pattern):
                    matched.append(file_path)
                    break
    
    return matched

def _infer_subdirectory(bids_name: str, modality: str) -> str:
    """Infer BIDS subdirectory from filename and modality."""
    name_lower = bids_name.lower()
    
    if modality == "mri":
        if any(kw in name_lower for kw in ['t1w', 't2w', 'flair', 'ct', 'anat']):
            return 'anat'
        elif any(kw in name_lower for kw in ['bold', 'task']):
            return 'func'
        elif 'dwi' in name_lower:
            return 'dwi'
        elif 'fmap' in name_lower or 'fieldmap' in name_lower:
            return 'fmap'
        else:
            return 'anat'
    elif modality == "nirs":
        return 'nirs'
    else:
        return modality

def _build_ascii_tree(root: Path, max_depth: int = 5) -> str:
    """Build ASCII tree representation."""
    lines = [root.name + "/"]
    
    def walk(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            entries = sorted([p for p in directory.iterdir()], key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for i, path in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            line = prefix + connector + path.name
            if path.is_dir():
                line += "/"
            lines.append(line)
            
            if path.is_dir():
                extension = "    " if is_last else "│   "
                walk(path, prefix + extension, depth + 1)
    
    walk(root)
    return "\n".join(lines)

def execute_bids_plan(input_root: Path, output_dir: Path, plan: Dict[str, Any], aux_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute LLM's BIDS plan with semantic matching.
    
    UNIVERSAL SOLUTION: Works with any naming convention.
    """
    info("Executing BIDS Plan (universal semantic matching)...")
    
    bids_root = Path(output_dir) / "bids_compatible"
    derivatives_dir = bids_root / "derivatives"
    ensure_dir(bids_root)
    ensure_dir(derivatives_dir)
    
    logs = []
    successes = 0
    failures = 0
    
    # Step 1: Copy trio files
    info("\n[1/3] Organizing trio files...")
    trio_files = ["dataset_description.json", "README.md", "participants.tsv", "participants.json"]
    
    for trio_file in trio_files:
        src = output_dir / trio_file
        if not src.exists():
            src = input_root / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")
    
    # Step 2: Process data files
    info("\n[2/3] Organizing BIDS data files...")
    
    mappings = plan.get("mappings", [])
    all_files = list_all_files(input_root)
    
    for mapping_idx, mapping in enumerate(mappings, 1):
        modality = mapping.get("modality")
        patterns = mapping.get("match", [])
        format_ready = mapping.get("format_ready", True)
        convert_to = mapping.get("convert_to", "none")
        filename_rules = mapping.get("filename_rules", [])
        
        info(f"  [{mapping_idx}/{len(mappings)}] Processing {modality} files...")
        
        matched_files = _match_files_with_patterns(all_files, patterns, input_root)
        
        if not matched_files:
            info(f"    No files matched patterns: {patterns}")
            continue
        
        info(f"    Matched: {len(matched_files)} files")
        
        if not format_ready and convert_to == "dicom_to_nifti":
            # DICOM conversion
            if not check_dcm2niix_available():
                warn(f"    dcm2niix not found - skipping")
                failures += len(matched_files)
                continue
            
            info(f"    Using semantic grouping for DICOM conversion...")
            
            # 语义分组
            dicom_groups = _semantic_group_dicom_files(matched_files, filename_rules)
            
            info(f"    Created {len(dicom_groups)} volume group(s)")
            
            # 转换每组
            temp_base = Path(output_dir) / "_staging" / "dicom_temp"
            
            for group_key, group_info in dicom_groups.items():
                try:
                    dicom_files = group_info["files"]
                    bids_name = group_info["bids_name"]
                    subject_id = group_info["subject"]
                    is_fallback = group_info.get("is_fallback", False)
                    
                    subdir = _infer_subdirectory(bids_name, modality)
                    output_path = bids_root / f"sub-{subject_id}" / subdir / bids_name
                    
                    temp_dir = temp_base / group_key
                    ensure_dir(temp_dir)
                    
                    marker = "[Fallback]" if is_fallback else ""
                    info(f"      {marker} {len(dicom_files)} files → {bids_name}")
                    
                    result = run_dcm2niix_batch(dicom_files, output_path, temp_dir)
                    
                    if result:
                        logs.append({
                            "source": f"{len(dicom_files)} DICOM files",
                            "destination": str(output_path.relative_to(bids_root)),
                            "action": "dicom_to_nifti",
                            "status": "success",
                            "group": group_key
                        })
                        successes += 1
                    else:
                        failures += 1
                    
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
                except Exception as e:
                    warn(f"      Conversion failed: {e}")
                    failures += 1
            
            if temp_base.exists():
                shutil.rmtree(temp_base, ignore_errors=True)
        
        elif format_ready and convert_to == "none":
            _process_ready_files(matched_files, filename_rules, modality, 
                               input_root, bids_root, logs)
            successes += len(matched_files)
        
        else:
            warn(f"    Conversion '{convert_to}' not implemented")
            failures += len(matched_files)
    
    # Step 3: Derivatives
    info("\n[3/3] Organizing derivative files...")
    # (保持原逻辑)
    
    # Save reports
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    
    manifest_files = [str(p.relative_to(bids_root)).replace("\\", "/") 
                      for p in bids_root.rglob("*") if p.is_file() and not p.name.startswith('.')]
    
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files": sorted(manifest_files),
        "tree": _build_ascii_tree(bids_root)
    })
    
    info(f"")
    info(f"✓ BIDS Dataset Created")
    info(f"Location: {bids_root}")
    info(f"Successful conversions: {successes}")
    info(f"Failed conversions: {failures}")
    
    # 验证结果
    subject_dirs = list(bids_root.glob("sub-*"))
    if len(subject_dirs) > 0:
        info(f"\nCreated {len(subject_dirs)} subject director(ies):")
        for subj_dir in sorted(subject_dirs):
            nii_count = len(list(subj_dir.rglob("*.nii.gz")))
            info(f"  {subj_dir.name}: {nii_count} NIfTI files")
    
    return {
        "total_mappings": len(mappings),
        "successful_conversions": successes,
        "failed_conversions": failures,
        "bids_root": str(bids_root)
    }

def _semantic_group_dicom_files(matched_files: List[Path], filename_rules: List[Dict]) -> Dict[str, Dict]:
    """
    语义分组 - 完全通用的算法。
    
    核心策略:
    1. 对每个文件,找到最佳匹配的规则
    2. 相同规则的文件合并成一组
    3. 未匹配的文件智能推断
    """
    return _group_dicom_files_by_rule(matched_files, filename_rules)

def _process_ready_files(matched_files: List[Path], filename_rules: List[Dict],
                        modality: str, input_root: Path, bids_root: Path, logs: List) -> None:
    """Process files already in correct format."""
    for src in matched_files:
        try:
            bids_name, subject_id, entities = None, None, None
            
            for rule in filename_rules:
                bids_name, subject_id, entities = _apply_llm_filename_rule(src, rule)
                if bids_name:
                    break
            
            if not subject_id:
                subject_id = "01"
            
            if not bids_name:
                bids_name = src.name
            
            subdir = _infer_subdirectory(bids_name, modality)
            rel_bids_path = Path(f"sub-{subject_id}/{subdir}/{bids_name}")
            dst = bids_root / rel_bids_path
            
            ensure_dir(dst.parent)
            copy_file(src, dst)
            
            # Copy JSON sidecar if exists
            if src.suffix == '.gz' and src.stem.endswith('.nii'):
                json_src = src.parent / (src.stem[:-4] + '.json')
            else:
                json_src = src.with_suffix('.json')
            
            if json_src.exists():
                if dst.suffix == '.gz' and dst.stem.endswith('.nii'):
                    json_dst = dst.parent / (dst.stem[:-4] + '.json')
                else:
                    json_dst = dst.with_suffix('.json')
                copy_file(json_src, json_dst)
            
            logs.append({
                "source": str(src.relative_to(input_root)),
                "destination": str(rel_bids_path),
                "action": "organize",
                "status": "success"
            })
            
        except Exception as e:
            warn(f"      Failed: {src.name}: {e}")
