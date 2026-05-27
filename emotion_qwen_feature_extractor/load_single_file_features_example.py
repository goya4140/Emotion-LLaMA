# -*- coding: utf-8 -*-
"""
文件名：load_single_file_features_example.py
功能：单文件模式特征加载示例
创建时间：2025-07-11

本脚本演示如何加载单文件模式(.pt)保存的特征，
并将其输入LLM进行推理，展示特征一致性验证。
"""

import os
import sys
import time
import torch
from pathlib import Path

# 导入工具函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import load_features_single, get_gpu_memory_usage


def load_and_verify_single_features(feature_dir: str, video_name: str) -> dict:
    """
    加载单个.pt文件中的特征并打印信息。
    
    Args:
        feature_dir: 特征文件目录
        video_name: 视频名称
        
    Returns:
        特征数据字典
    """
    file_path = os.path.join(feature_dir, f"{video_name}_features.pt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"特征文件不存在: {file_path}")
    
    print(f"📂 加载特征文件: {file_path}")
    data = load_features_single(file_path)
    
    features = data["features"]
    metadata = data.get("metadata", {})
    
    print(f"  ✅ 加载成功!")
    print(f"  特征形状: {data['feature_shape']}")
    print(f"  数据类型: {features.dtype}")
    print(f"  文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print(f"  元数据: {metadata}")
    print(f"  保存时间: {data.get('timestamp', 'N/A')}")
    
    return data


def compare_features(features1: torch.Tensor, features2: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """
    比较两个特征张量是否一致。
    用于验证保存后加载的特征与原始特征是否完全一致。
    
    Args:
        features1: 第一个特征张量
        features2: 第二个特征张量
        tolerance: 允许的最大数值误差
        
    Returns:
        是否一致
    """
    if features1.shape != features2.shape:
        print(f"  ❌ 形状不一致: {features1.shape} vs {features2.shape}")
        return False
    
    diff = (features1 - features2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  最大偏差: {max_diff:.10f}")
    print(f"  平均偏差: {mean_diff:.10f}")
    
    if max_diff <= tolerance:
        print(f"  ✅ 特征一致 (误差 <= {tolerance})")
        return True
    else:
        print(f"  ❌ 特征不一致 (误差 > {tolerance})")
        return False


def main():
    """
    主函数：演示单文件模式特征加载流程。
    """
    print("=" * 80)
    print("Emotion-Qwen 单文件模式特征加载示例")
    print("=" * 80)
    
    # 配置（按需修改）
    feature_dir = "/root/autodl-tmp/extracted_features/CA-MER/single_files"
    
    # 检查目录是否存在
    if not os.path.isdir(feature_dir):
        print(f"❌ 特征目录不存在: {feature_dir}")
        print(f"请先运行 extract_emotion_qwen_features.py 提取特征")
        print(f"或修改 feature_dir 变量指向正确的目录")
        return
    
    # 列出可用特征文件
    pt_files = list(Path(feature_dir).glob("*_features.pt"))
    if len(pt_files) == 0:
        print(f"❌ 目录中没有找到.pt文件: {feature_dir}")
        return
    
    print(f"📁 找到 {len(pt_files)} 个特征文件")
    print(f"\n前5个文件:")
    for f in pt_files[:5]:
        print(f"  - {f.name}")
    print()
    
    # 演示加载前几个特征文件
    gpu_mem_before = get_gpu_memory_usage()
    
    for i, pt_file in enumerate(pt_files[:3]):  # 仅加载前3个作为演示
        print(f"\n{'='*40}")
        print(f"示例 {i+1}: {pt_file.name}")
        try:
            data = load_and_verify_single_features(
                feature_dir, 
                pt_file.stem.replace("_features", "")
            )
            
            features = data["features"]
            print(f"  前3个值: {features[0, :3].tolist() if features.shape[0] > 0 else 'N/A'}")
            
            # 释放内存
            del data
            
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    
    gpu_mem_after = get_gpu_memory_usage()
    print(f"\n{'='*40}")
    print(f"GPU显存使用: {gpu_mem_before:.2f}GB -> {gpu_mem_after:.2f}GB")
    
    # 完整性验证示例（如果至少有2个文件）
    if len(pt_files) >= 2:
        print(f"\n{'='*40}")
        print("特征一致性验证:")
        data1 = load_and_verify_single_features(
            feature_dir,
            pt_files[0].stem.replace("_features", "")
        )
        data2 = load_and_verify_single_features(
            feature_dir,
            pt_files[1].stem.replace("_features", "")
        )
        # 不同视频的特征应该不同
        if data1["features"].shape == data2["features"].shape:
            is_consistent = compare_features(data1["features"], data2["features"], tolerance=1.0)
            # 两个不同视频的特征应该有显著差异
            if not is_consistent:
                print("  ✅ 符合预期：不同视频的特征有显著差异")
        else:
            print("  不同视频的特征形状不同（符合预期）")
    
    print(f"\n{'='*40}")
    print("示例完成！")
    print("")
    print("💡 后续使用特征进行LLM推理的步骤:")
    print("  1. 加载Emotion-Qwen完整模型（包含LLM主干）")
    print("  2. 直接输入特征张量作为visual embeddings")
    print("  3. 跳过视频预处理和encoder前向传播")
    print("  4. 与文本token一起输入LLM进行推理")
    print("")
    print("  示例代码:")
    print("  ```python")
    print("  features = load_features_single('path/to/video_features.pt')['features']")
    print("  features = features.to(device='cuda', dtype=model.dtype)")
    print("  # 将features输入模型（具体方式取决于模型架构）")
    print("  output = model.generate(visual_embeds=features, input_ids=input_ids)")
    print("  ```")


if __name__ == "__main__":
    main()
