#!/usr/bin/env python3
"""
诊断工具:分析 BIDS 转换问题

Usage:
    python diagnostic_tool.py outputs/run3
"""

import sys
import json
from pathlib import Path

def diagnose_conversion(output_dir: str):
    """诊断 BIDS 转换过程的问题"""
    
    output_path = Path(output_dir)
    
    print("=" * 80)
    print("BIDS 转换诊断工具")
    print("=" * 80)
    
    # 1. 检查 evidence_bundle
    print("\n[1] Evidence Bundle 检查:")
    evidence_path = output_path / "_staging" / "evidence_bundle.json"
    
    if not evidence_path.exists():
        print(f"  ❌ 未找到: {evidence_path}")
        return
    
    with open(evidence_path) as f:
        evidence = json.load(f)
    
    all_files_count = len(evidence.get("all_files", []))
    samples_count = len(evidence.get("samples", []))
    
    print(f"  ✓ 总文件数: {all_files_count}")
    print(f"  ✓ 采样文件数: {samples_count}")
    
    if "sampling_strategy" in evidence:
        strategy = evidence["sampling_strategy"]
        print(f"  ✓ 采样策略: {strategy.get('method')}")
        print(f"  ✓ 检测到的模式数: {strategy.get('total_patterns_detected')}")
    
    # 2. 检查 BIDSPlan
    print("\n[2] BIDS Plan 检查:")
    plan_path = output_path / "_staging" / "BIDSPlan.yaml"
    
    if not plan_path.exists():
        print(f"  ❌ 未找到: {plan_path}")
        return
    
    import yaml
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    
    subjects = plan.get("subjects", {}).get("labels", [])
    print(f"  ✓ 被试数: {len(subjects)} ({subjects})")
    
    mappings = plan.get("mappings", [])
    print(f"  ✓ Mapping 数量: {len(mappings)}")
    
    for idx, mapping in enumerate(mappings, 1):
        rules = mapping.get("filename_rules", [])
        print(f"    Mapping {idx}: {len(rules)} 个规则")
        
        # 显示前3个规则
        for i, rule in enumerate(rules[:3], 1):
            pattern = rule.get("match_pattern")
            template = rule.get("bids_template")
            print(f"      规则 {i}:")
            print(f"        Pattern: {pattern}")
            print(f"        Template: {template}")
        
        if len(rules) > 3:
            print(f"      ... 还有 {len(rules) - 3} 个规则")
    
    # 3. 检查 conversion_log
    print("\n[3] 转换日志检查:")
    log_path = output_path / "_staging" / "conversion_log.json"
    
    if not log_path.exists():
        print(f"  ❌ 未找到: {log_path}")
        return
    
    with open(log_path) as f:
        logs = json.load(f)
    
    print(f"  ✓ 转换记录数: {len(logs)}")
    
    # 分析转换模式
    file_counts = {}
    subjects_used = set()
    
    for log_entry in logs:
        source = log_entry.get("source", "")
        dest = log_entry.get("destination", "")
        
        # 提取文件数
        if "DICOM files" in source:
            count = int(source.split()[0])
            file_counts[count] = file_counts.get(count, 0) + 1
        
        # 提取被试
        if dest.startswith("sub-"):
            subject = dest.split("/")[0].replace("sub-", "")
            subjects_used.add(subject)
    
    print(f"\n  转换模式分析:")
    for count, occurrences in sorted(file_counts.items()):
        print(f"    {count} 个文件/组: {occurrences} 次")
        if count == 1:
            print(f"      ⚠️ 警告: 单文件转换通常表示分组失败!")
    
    print(f"\n  使用的被试 ID: {sorted(subjects_used)}")
    print(f"  预期被试数: {len(subjects)}")
    
    if len(subjects_used) != len(subjects):
        print(f"  ❌ 被试数不匹配! 预期 {len(subjects)}, 实际 {len(subjects_used)}")
    
    # 4. 测试语义匹配
    print("\n[4] 语义匹配测试:")
    
    # 取第一个规则和第一个实际文件测试
    if mappings and mappings[0].get("filename_rules"):
        test_rule = mappings[0]["filename_rules"][0]
        test_pattern = test_rule.get("match_pattern")
        
        # 从 all_files 找一个应该匹配的文件
        test_files = []
        for filepath in evidence.get("all_files", [])[:100]:
            filename = filepath.split('/')[-1]
            # 寻找包含规则关键词的文件
            if test_pattern:
                # 从 pattern 提取主要关键词
                keywords = re.findall(r'([A-Z][a-z]+)', test_pattern)
                if keywords and all(kw.lower() in filename.lower() for kw in keywords[:2]):
                    test_files.append(filename)
                    if len(test_files) >= 3:
                        break
        
        if test_files:
            print(f"  规则: {test_pattern}")
            print(f"  模板: {test_rule.get('bids_template')}")
            print(f"  应该匹配的文件示例:")
            for tf in test_files:
                print(f"    - {tf}")
        
        # 测试语义提取
        components = _extract_semantic_components(test_pattern)
        print(f"\n  语义组件提取:")
        print(f"    前缀: {components.get('prefix')}")
        print(f"    关键词: {components.get('keywords')}")
        print(f"    扩展名: {components.get('extension')}")
    
    # 5. 检查实际生成的文件
    print("\n[5] 实际输出检查:")
    bids_dir = output_path / "bids_compatible"
    
    if bids_dir.exists():
        nii_files = list(bids_dir.rglob("*.nii.gz"))
        print(f"  ✓ 生成的 NIfTI 文件: {len(nii_files)}")
        
        by_subject = {}
        for nii_file in nii_files:
            parts = nii_file.parts
            for part in parts:
                if part.startswith("sub-"):
                    by_subject[part] = by_subject.get(part, 0) + 1
                    break
        
        print(f"\n  按被试统计:")
        for subject, count in sorted(by_subject.items()):
            print(f"    {subject}: {count} 个文件")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

def _extract_semantic_components(pattern: str) -> Dict:
    """Helper function to extract semantic components"""
    components = {
        "prefix": None,
        "keywords": [],
        "extension": None
    }
    
    # 提取前缀
    prefix_match = re.match(r'^([A-Za-z0-9-]+)', pattern)
    if prefix_match:
        components["prefix"] = prefix_match.group(1).lower()
    
    # 提取关键词
    keyword_matches = re.findall(r'[-_]([A-Za-z]+)', pattern)
    components["keywords"] = [k.lower() for k in keyword_matches if len(k) > 1]
    
    # 提取扩展名
    if r'\.dcm' in pattern or '.dcm' in pattern:
        components["extension"] = ".dcm"
    elif r'\.nii\.gz' in pattern:
        components["extension"] = ".nii.gz"
    
    return components

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_tool.py <output_dir>")
        print("Example: python diagnostic_tool.py outputs/run3")
        sys.exit(1)
    
    diagnose_conversion(sys.argv[1])
