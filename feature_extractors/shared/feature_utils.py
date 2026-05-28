# -*- coding: utf-8 -*-
"""
文件名：feature_utils.py
功能：视觉特征提取共享工具函数（EVA-CLIP / MAE / VideoMAE 三路提取器共用）

本文件包含特征提取所需的各种辅助工具函数，包括：
- 特征保存与加载（单文件）
- inode使用量检查
- 日志配置
- GPU显存监控
- 视频列表获取（支持标准 JSON 格式及 EMER conversations 格式）
"""

import os
import sys
import json
import time
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any

import torch

# ============================================================================
# 常量定义（来源：Emotion-Qwen官方processing_emotionqwen_vl.py）
# ============================================================================

# 像素限制（来源：官方processor.__init__）
DEFAULT_MAX_PIXELS = 768 * 28 * 28  # = 602112 (qwen-vl-utils upper bound)

# 默认帧率（来源：官方processor中的EmotionQwen_ProcessorKwargs._defaults）
DEFAULT_FPS = 2.0


# ============================================================================
# 日志配置函数
# ============================================================================

def setup_logger(log_dir: str, name: str = "feature_extraction", level: str = "INFO") -> Tuple[logging.Logger, str]:
    """
    配置日志系统，同时输出到控制台和单个日志文件（严格禁止分割成多个小日志文件）。

    Args:
        log_dir: 日志文件存储目录
        name: 日志器名称
        level: 日志级别

    Returns:
        (logger, log_file_path): 配置好的logger对象和日志文件路径
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 按时间戳命名日志文件（格式：feature_extraction_YYYYMMDD_HHMMSS.log）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, f"feature_extraction_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的handler（防止重复添加）
    logger.handlers.clear()
    
    # 文件Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper()))
    file_formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


# ============================================================================
# inode检查与优化函数
# ============================================================================

def check_inode_usage() -> Tuple[int, int, float]:
    """
    检查当前分区的inode使用情况。
    用于AutoDL平台inode限制监控（限制200000个）。

    Returns:
        (used, total, usage_percentage): 已用inode数、总inode数、使用百分比
    """
    try:
        # 在当前工作目录所在分区运行df命令
        result = subprocess.run(
            ["df", "-i", "."],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                total = int(parts[1])
                used = int(parts[2])
                free = int(parts[3])
                usage_pct = float(parts[4].rstrip('%'))
                return used, total, usage_pct
    except Exception as e:
        logging.getLogger("feature_extraction").warning(f"inode检查失败: {e}")
    
    return 0, 0, 0.0


def get_available_inodes() -> int:
    """
    获取当前分区的剩余inode数量。

    Returns:
        剩余inode数量（失败返回-1）
    """
    used, total, _ = check_inode_usage()
    if total > 0:
        return total - used
    return -1


# ============================================================================
# 特征保存与加载函数
# ============================================================================

def save_features_single(
    features: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    保存单个视频的特征到.pt文件（单文件模式）。

    Args:
        features: 特征张量 [seq_len, hidden_dim]
        output_path: 输出文件路径
        metadata: 可选的元数据字典
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_dict = {
        "features": features,
        "feature_shape": features.shape,
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(save_dict, output_path)


def load_features_single(file_path: str) -> Dict:
    """
    从.pt文件加载特征。

    Args:
        file_path: .pt文件路径

    Returns:
        包含features、feature_shape、metadata等键的字典
    """
    return torch.load(file_path, map_location="cpu", weights_only=False)


# ============================================================================
# 显存监控函数
# ============================================================================

def get_gpu_memory_usage(device: str = "cuda:0") -> float:
    """
    获取当前GPU显存使用量（GB）。

    Args:
        device: 设备名称

    Returns:
        GPU显存使用量（GB），如果不可用返回-1.0
    """
    if not torch.cuda.is_available():
        return -1.0
    try:
        # 使用torch.cuda.memory_allocated获取PyTorch分配的显存
        memory_bytes = torch.cuda.memory_allocated(device)
        return memory_bytes / (1024.0 ** 3)  # 转换为GB
    except Exception:
        return -1.0


# ============================================================================
# 数据列表获取函数
# ============================================================================

def _extract_label_from_conversations(conversations: list) -> Optional[str]:
    """从 EMER conversations 格式中解析 <answer> 标签内的情感标签。"""
    import re
    for turn in conversations:
        if turn.get("from") == "gpt":
            m = re.search(r'<answer>(.*?)</answer>', turn.get("value", ""), re.DOTALL)
            if m:
                return m.group(1).strip()
    return None


def get_video_list(dataset_path: str, annotation_file: Optional[str] = None,
                   video_dir: str = "video-aligned",
                   extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    获取待处理的视频列表和对应的标签信息。
    支持JSON标注文件和直接扫描目录两种方式。

    Args:
        dataset_path: 数据集根目录
        annotation_file: 标注JSON文件名（可选）
        video_dir: 视频文件夹相对路径
        extensions: 支持的文件后缀列表

    Returns:
        视频信息列表，每个元素为{"video_path": str, "label": str, "id": str}
    """
    if extensions is None:
        extensions = [".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv"]
    
    video_list = []
    
    # 优先使用JSON标注文件
    if annotation_file and os.path.exists(os.path.join(dataset_path, annotation_file)):
        ann_path = os.path.join(dataset_path, annotation_file)
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 处理不同格式的JSON（来自官方数据集可能有不同结构）
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "data" in data:
            entries = data["data"]
        elif isinstance(data, dict) and "samples" in data:
            entries = data["samples"]
        else:
            entries = []
            # 尝试遍历字典值
            for key, val in data.items():
                if isinstance(val, dict) and ("video" in val or "file" in val):
                    entries.append(val)
        
        for i, entry in enumerate(entries):
            # 尝试多种可能的字段名获取视频路径和标签
            video_file = entry.get("video") or entry.get("file") or entry.get("path") or entry.get("video_path")
            label = entry.get("label") or entry.get("emotion") or entry.get("class") or entry.get("category")
            # EMER conversations 格式：从 <answer> 标签提取情感标签
            if label is None and "conversations" in entry:
                label = _extract_label_from_conversations(entry["conversations"])
            sample_id = entry.get("id") or entry.get("sample_id") or str(i)
            
            if video_file:
                # 构建完整路径
                full_path = os.path.join(dataset_path, video_dir, video_file) if not os.path.isabs(video_file) else video_file
                if os.path.exists(full_path):
                    video_list.append({
                        "video_path": full_path,
                        "label": str(label) if label is not None else "unknown",
                        "id": str(sample_id),
                        "video_file": video_file
                    })
    
    # 如果没有标注文件或解析失败，直接扫描目录
    if len(video_list) == 0:
        vdir = os.path.join(dataset_path, video_dir)
        if os.path.isdir(vdir):
            for fname in sorted(os.listdir(vdir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in extensions:
                    video_list.append({
                        "video_path": os.path.join(vdir, fname),
                        "label": "unknown",
                        "id": os.path.splitext(fname)[0],
                        "video_file": fname
                    })
    
    return video_list


# ============================================================================
# 文件操作辅助函数
# ============================================================================

def count_files_in_directory(directory: str, recursive: bool = True) -> int:
    """
    统计目录中的文件数量（用于inode监控）。

    Args:
        directory: 目标目录
        recursive: 是否递归统计

    Returns:
        文件数量
    """
    if not os.path.isdir(directory):
        return 0
    
    if recursive:
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count
    else:
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


if __name__ == "__main__":
    # 简单自检
    print("feature_utils.py 已加载")
    print(f"inode使用量: {check_inode_usage()}")
