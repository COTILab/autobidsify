# classification.py
# LLM-based classification for mixed modality datasets.

"""
中文说明：
分类模块，用于混合模态数据的智能分流。

功能特性：
1. LLM分类：基于文档内容、文件名、元数据进行分类
2. 三类输出：NIRS文件、MRI文件、不确定文件
3. 物理分流：将文件复制到独立的pool目录
4. 证据追溯：记录分类依据和置信度

工作流程：
1. 将evidence_bundle发送给LLM（包含完整文档内容）
2. LLM分析文档中的描述（如"fNIRS optodes"、"3T MRI scanner"）
3. 根据分析结果生成classification_plan.json
4. 物理复制文件到nirs_pool和mri_pool
5. 不确定的文件放入unknown_pool并生成问题

分类策略：
- 优先使用文档内容（protocol.pdf提到的模态）
- 其次使用文件扩展名和路径
- 最后使用元数据特征（如数组维度）
- 不确定时标记为unknown并要求用户介入

输出文件：
- classification_plan.json：分类结果和依据
- _staging/nirs_pool/：NIRS文件池
- _staging/mri_pool/：MRI文件池
- _staging/unknown/：不确定文件
"""

from pathlib import Path
from typing import Dict, Any, List
import json
from utils import ensure_dir, write_json, copy_file, info, warn, fatal, read_json
from constants import CLASSIFICATION_PLAN, NIRS_POOL, MRI_POOL, UNKNOWN_POOL
from llm import llm_classify

def classify_and_stage(model: str, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Use LLM to classify files into NIRS/MRI/UNKNOWN pools.
    Then physically stage files by copying to pool directories.
    
    Args:
        model: LLM model name
        bundle: Evidence bundle with documents
        out_dir: Output directory
    
    Returns:
        Classification plan with questions
    """
    info("Calling LLM for classification...")
    
    # Prepare payload for LLM
    payload = json.dumps(bundle, ensure_ascii=False)
    
    # Call LLM
    try:
        response_text = llm_classify(model, payload)
        plan = json.loads(response_text)
    except json.JSONDecodeError as e:
        fatal(f"LLM returned invalid JSON: {e}")
        return {}
    except Exception as e:
        fatal(f"Classification failed: {e}")
        return {}
    
    # Save classification plan
    plan_path = Path(out_dir) / "_staging" / "classification_plan.json"
    write_json(plan_path, plan)
    info(f"✓ Classification plan saved: {plan_path}")
    
    # Display classification rationale if provided
    if "classification_rationale" in plan:
        rationale = plan["classification_rationale"]
        info(f"Classification confidence: {rationale.get('confidence_level', 'unknown')}")
        if "key_evidence_from_documents" in rationale:
            info("Key evidence from documents:")
            for evidence in rationale["key_evidence_from_documents"]:
                info(f"  • {evidence}")
    
    # Stage files into pools
    _stage_files_to_pools(bundle, plan, out_dir)
    
    return plan

def _stage_files_to_pools(bundle: Dict[str, Any], plan: Dict[str, Any], out_dir: Path):
    """
    Physically copy files to their respective pools.
    
    Creates directory structure in pools matching original structure.
    """
    root = Path(bundle["root"])
    
    # Create pool directories
    nirs_pool = Path(out_dir) / NIRS_POOL
    mri_pool = Path(out_dir) / MRI_POOL
    unknown_pool = Path(out_dir) / UNKNOWN_POOL
    
    ensure_dir(nirs_pool)
    ensure_dir(mri_pool)
    ensure_dir(unknown_pool)
    
    def stage_file_list(file_list: List[str], pool: Path, label: str):
        """Copy files from list to pool."""
        count = 0
        for relpath in file_list:
            src = root / relpath
            if src.is_file():
                dst = pool / relpath
                ensure_dir(dst.parent)
                copy_file(src, dst)
                count += 1
        
        if count > 0:
            info(f"✓ Staged {count} files to {label} pool")
    
    # Stage each category
    stage_file_list(plan.get("nirs_files", []), nirs_pool, "NIRS")
    stage_file_list(plan.get("mri_files", []), mri_pool, "MRI")
    stage_file_list(plan.get("unknown_files", []), unknown_pool, "UNKNOWN")
    
    # Warn about unknown files
    unknown_count = len(plan.get("unknown_files", []))
    if unknown_count > 0:
        warn(f"{unknown_count} files could not be classified")
        warn(f"Review classification_plan.json and assign manually")

def classify_files(model: str, output_dir: Path) -> None:
    """
    High-level classification function called by CLI.
    
    Args:
        model: LLM model name
        output_dir: Output directory (contains _staging/evidence_bundle.json)
    """
    # Load evidence bundle
    bundle_path = Path(output_dir) / "_staging" / "evidence_bundle.json"
    
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        fatal("Run 'evidence' step first")
        return
    
    bundle = read_json(bundle_path)
    
    # Run classification
    plan = classify_and_stage(model, bundle, output_dir)
    
    # Display summary
    nirs_count = len(plan.get("nirs_files", []))
    mri_count = len(plan.get("mri_files", []))
    unknown_count = len(plan.get("unknown_files", []))
    
    info(f"\nClassification summary:")
    info(f"  NIRS files: {nirs_count}")
    info(f"  MRI files: {mri_count}")
    info(f"  Unknown files: {unknown_count}")
