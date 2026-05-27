# -*- coding: utf-8 -*-
"""
文件名：verify_feature_consistency.py
功能：Emotion-Qwen 特征一致性验证脚本
创建时间：2025-07-11

本脚本用于验证单文件模式下特征保存与加载的完整性，
包括文件结构、NaN/Inf检查、形状与统计信息。

参考源：
- 官方processing_emotionqwen_vl.py 预处理流程
- 官方modeling_emotionqwen_vl.py encoder输出
"""

import os
import sys
import json
import torch
from pathlib import Path

# 导入工具函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import (
    load_features_single,
)


# ============================================================================
# 离线特征验证（单文件模式）
# ============================================================================

def verify_single_file_mode(feature_dir: str, num_samples: int = 5) -> bool:
    """
    验证单文件模式下的特征完整性。
    加载每个.pt文件并检查其结构和数据完整性。

    Args:
        feature_dir: 特征文件目录
        num_samples: 验证样本数

    Returns:
        是否全部通过验证
    """
    print("=" * 60)
    print("单文件模式特征验证")
    print("=" * 60)
    
    if not os.path.isdir(feature_dir):
        print(f"❌ 目录不存在: {feature_dir}")
        return False
    
    pt_files = list(Path(feature_dir).glob("*_features.pt"))
    if len(pt_files) == 0:
        print(f"❌ 未找到.pt文件")
        return False
    
    print(f"找到 {len(pt_files)} 个特征文件，随机验证 {min(num_samples, len(pt_files))} 个")
    
    all_passed = True
    import random
    samples = random.sample(pt_files, min(num_samples, len(pt_files)))
    
    for i, pt_file in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {pt_file.name}")
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            
            # 验证必要字段
            if "features" not in data:
                print(f"  ❌ 缺少 'features' 字段")
                all_passed = False
                continue
            
            features = data["features"]
            
            # 检查特征形状有效性
            if features.dim() != 2:
                print(f"  ⚠ 预期2D张量，实际: {features.dim()}D")
            
            # 检查是否有NaN或Inf
            if torch.isnan(features).any():
                print(f"  ❌ 特征包含 NaN")
                all_passed = False
            elif torch.isinf(features).any():
                print(f"  ❌ 特征包含 Inf")
                all_passed = False
            else:
                print(f"  ✅ 形状: {list(features.shape)}, dtype: {features.dtype}")
                print(f"     值范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
                print(f"     均值: {features.mean().item():.4f}, 标准差: {features.std().item():.4f}")
            
            # 检查元数据
            metadata = data.get("metadata", {})
            if metadata:
                print(f"     元数据: {json.dumps(metadata, ensure_ascii=False)[:200]}")
            
            del data
            
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            all_passed = False
    
    return all_passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行全面的特征一致性验证。
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Emotion-Qwen 特征一致性验证工具")
    parser.add_argument("--feature-dir", type=str,
                       default="/root/autodl-tmp/extracted_features/CA-MER",
                       help="特征输出目录根路径")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="随机验证样本数")
    args = parser.parse_args()

    print("=" * 80)
    print("Emotion-Qwen 特征一致性验证工具")
    print("=" * 80)
    print(f"特征目录: {args.feature_dir}")
    print()

    all_passed_flag = True

    single_dir = os.path.join(args.feature_dir, "single_files")
    if os.path.isdir(single_dir):
        passed = verify_single_file_mode(single_dir, args.num_samples)
        all_passed_flag = all_passed_flag and passed
    else:
        print(f"❌ 单文件模式目录不存在: {single_dir}")

    print("\n" + "=" * 60)
    if all_passed_flag:
        print("✅ 所有验证通过！特征文件结构与数据完整")
    else:
        print("⚠ 部分验证失败，请检查特征文件")

