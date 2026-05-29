# -*- coding: utf-8 -*-
"""
文件名：extract_videomae_features.py
功能：使用 VideoMAE-Large (MCG-NJU/videomae-large) 提取视频时空特征

输出格式：[1024]  （spatio-temporal token 均值）
对应 Emotion-LLaMA 中的 feats_llama_proj2 输入边界（video_features[:, 1, :]）

用法：
    python extract_videomae_features.py --config config_videomae.yaml
    python extract_videomae_features.py --config config_videomae.yaml --dry-run
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

# ============================================================================
# 共享工具导入
# ============================================================================

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")
sys.path.insert(0, os.path.abspath(_SHARED_DIR))

from feature_utils import (
    setup_logger,
    save_features_single,
    get_video_list,
    get_gpu_memory_usage,
    check_inode_usage,
)


# ============================================================================
# 常量
# ============================================================================

DEFAULT_NUM_FRAMES = 16       # VideoMAE-Large 标准输入帧数
DEFAULT_IMAGE_SIZE = 224      # VideoMAE 输入分辨率
ENCODER_TYPE = "VideoMAE-Large"

# VideoMAE 归一化参数（ImageNet 标准）
VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD  = [0.229, 0.224, 0.225]


# ============================================================================
# VideoMAE 编码器
# ============================================================================

class VideoMAEEncoder(nn.Module):
    """
    VideoMAE-Large 特征编码器。

    前向传播：
        输入: pixel_values [1, T, 3, 224, 224]  (T=16)
        输出: [1024]  — spatio-temporal token 均值
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-large",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device_str = device
        self.torch_dtype = torch_dtype

        from transformers import VideoMAEModel
        self.model = VideoMAEModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.to(device)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [1, T, 3, 224, 224]  float16 on device

        Returns:
            features: [1024]
        """
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state: [1, seq_len, 1024]
        # seq_len = (T/2) × (224/16)² = 8 × 196 = 1568 (tube_size=2, patch_size=16)
        features = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [1024]
        return features


# ============================================================================
# 视频预处理
# ============================================================================

def preprocess_video_to_frames(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    logger: Optional[logging.Logger] = None,
) -> Optional[torch.Tensor]:
    """
    从视频中均匀采样指定帧数，返回归一化后的张量。

    Args:
        video_path: 视频文件路径
        num_frames: 采样帧数（VideoMAE 需要固定 16 帧）
        image_size: 图像尺寸（224）
        logger: 日志对象

    Returns:
        pixel_values: [1, T, 3, H, W]  float32，或 None（读取失败）
    """
    log = logger or logging.getLogger("videomae_extractor")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"无法打开视频: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        log.warning(f"视频帧数为 0: {video_path}")
        cap.release()
        return None

    # 均匀采样索引（不足 num_frames 时循环填充）
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.extend(indices[: num_frames - len(indices)])
        indices = np.array(indices[:num_frames])

    mean = np.array(VIDEOMAE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(VIDEOMAE_STD,  dtype=np.float32).reshape(1, 1, 3)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # 使用最近一帧复制
            if frames:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros(3, image_size, image_size))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std           # [H, W, 3]
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # [3, H, W]
        frames.append(frame)

    cap.release()

    video_tensor = torch.stack(frames, dim=0)  # [T, 3, H, W]
    return video_tensor.unsqueeze(0)           # [1, T, 3, H, W]


# ============================================================================
# 单视频特征提取
# ============================================================================

def extract_single_video(
    encoder: VideoMAEEncoder,
    video_info: Dict[str, Any],
    output_path: str,
    num_frames: int,
    logger: logging.Logger,
) -> bool:
    """
    提取单个视频的 VideoMAE 特征并保存。

    Returns:
        True = 成功，False = 失败
    """
    video_path = video_info["video_path"]
    video_id   = video_info["id"]
    label      = video_info.get("label", "unknown")

    pixel_values = preprocess_video_to_frames(
        video_path, num_frames=num_frames, logger=logger
    )
    if pixel_values is None:
        logger.warning(f"[跳过] 预处理失败: {video_path}")
        return False

    pixel_values = pixel_values.to(encoder.device_str, dtype=encoder.torch_dtype)

    try:
        features = encoder(pixel_values)          # [1024]
    except RuntimeError as e:
        if _is_oom_error(e):
            logger.error(f"[OOM] {video_id}: {e}")
            torch.cuda.empty_cache()
            return False
        raise

    features_cpu = features.float().cpu()

    metadata = {
        "video_id":    video_id,
        "video_file":  video_info.get("video_file", ""),
        "label":       label,
        "num_frames":  num_frames,
        "encoder_type": ENCODER_TYPE,
        "feature_shape": list(features_cpu.shape),
    }

    save_features_single(features_cpu, output_path, metadata=metadata)
    return True


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda" in msg


# ============================================================================
# 断点续传
# ============================================================================

def load_checkpoint(ckpt_file: str) -> set:
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "r") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_checkpoint(ckpt_file: str, completed: set) -> None:
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    with open(ckpt_file, "w") as f:
        json.dump({"completed": list(completed)}, f, indent=2)


# ============================================================================
# 数据集级提取
# ============================================================================

def extract_dataset(
    encoder: VideoMAEEncoder,
    dataset_cfg: Dict,
    output_root: str,
    resume_cfg: Dict,
    num_frames: int,
    logger: logging.Logger,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    对单个数据集执行 VideoMAE 特征提取。

    Returns:
        {"total": int, "success": int, "skip": int, "fail": int}
    """
    dataset_name  = dataset_cfg["name"]
    dataset_path  = dataset_cfg["dataset_path"]
    annotation    = dataset_cfg.get("annotation_file")
    video_dir     = dataset_cfg.get("video_dir", "video")
    extensions    = dataset_cfg.get("supported_extensions",
                                    [".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv"])

    output_dir = os.path.join(output_root, "VideoMAE", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    video_list = get_video_list(
        dataset_path=dataset_path,
        annotation_file=annotation,
        video_dir=video_dir,
        extensions=extensions,
    )
    logger.info(f"[{dataset_name}] 共找到 {len(video_list)} 个视频")

    # 断点续传
    ckpt_file = resume_cfg.get("checkpoint_file", "").replace("{dataset}", dataset_name)
    completed  = load_checkpoint(ckpt_file) if resume_cfg.get("enabled", False) else set()
    save_interval = resume_cfg.get("save_interval", 50)

    stats = {"total": len(video_list), "success": 0, "skip": 0, "fail": 0}

    for i, video_info in enumerate(video_list):
        video_id  = video_info["id"]
        out_fname = f"{video_id}_features.pt"
        out_path  = os.path.join(output_dir, out_fname)

        # 已完成跳过
        if video_id in completed or os.path.exists(out_path):
            stats["skip"] += 1
            continue

        if dry_run:
            logger.info(f"[dry-run] {video_id} → {out_path}")
            stats["success"] += 1
            continue

        ok = extract_single_video(
            encoder=encoder,
            video_info=video_info,
            output_path=out_path,
            num_frames=num_frames,
            logger=logger,
        )

        if ok:
            stats["success"] += 1
            completed.add(video_id)
            gpu_gb = get_gpu_memory_usage(encoder.device_str)
            logger.info(
                f"[{dataset_name}] [{i+1}/{stats['total']}] {video_id} "
                f"→ 保存完成  GPU={gpu_gb:.2f}GB"
            )
        else:
            stats["fail"] += 1
            logger.warning(f"[{dataset_name}] [{i+1}/{stats['total']}] {video_id} → 失败")

        if resume_cfg.get("enabled", False) and (i + 1) % save_interval == 0:
            save_checkpoint(ckpt_file, completed)
            logger.info(f"[{dataset_name}] 断点已保存（已完成 {len(completed)} 个）")

        torch.cuda.empty_cache()

    if resume_cfg.get("enabled", False):
        save_checkpoint(ckpt_file, completed)

    logger.info(
        f"[{dataset_name}] 完成: total={stats['total']} "
        f"success={stats['success']} skip={stats['skip']} fail={stats['fail']}"
    )
    return stats


# ============================================================================
# inode 检查
# ============================================================================

def _check_inode_and_warn(logger: logging.Logger) -> None:
    used, total, pct = check_inode_usage()
    if total > 0:
        logger.info(f"inode 使用率: {pct:.1f}%  ({used}/{total})")
        if pct > 80:
            logger.warning("inode 使用率超过 80%，请注意磁盘空间！")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VideoMAE-Large 视频特征提取")
    parser.add_argument("--config", required=True, help="配置文件路径 (YAML)")
    parser.add_argument("--dry-run", action="store_true", help="仅打印路径，不实际提取")
    parser.add_argument("--dataset", default=None, help="只处理指定数据集名称")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 日志
    log_cfg = cfg.get("logging", {})
    logger, log_file = setup_logger(
        log_dir=log_cfg.get("output_dir", "/tmp/logs"),
        name="videomae_extractor",
        level=log_cfg.get("level", "INFO"),
    )
    logger.info(f"日志文件: {log_file}")
    logger.info(f"配置: {args.config}")

    _check_inode_and_warn(logger)

    # 模型配置
    model_cfg    = cfg.get("model", {})
    model_name   = model_cfg.get("model_name", "MCG-NJU/videomae-large")
    device       = model_cfg.get("device", "cuda:0")
    precision    = model_cfg.get("precision", "float16")
    torch_dtype  = torch.float16 if precision in ("fp16", "float16") else torch.float32

    # HuggingFace 镜像（优先使用配置文件，其次读取环境变量）
    # AutoDL 推荐镜像：https://hf-mirror.com
    hf_endpoint = model_cfg.get("hf_endpoint", "") or os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        logger.info(f"HuggingFace 镜像: {hf_endpoint}")

    preproc_cfg  = cfg.get("vision_preprocess", {})
    num_frames   = int(preproc_cfg.get("num_frames", DEFAULT_NUM_FRAMES))

    save_cfg   = cfg.get("feature_saving", {})
    output_root = save_cfg.get("output_root", "/tmp/features")
    resume_cfg  = cfg.get("resume", {"enabled": False})

    # 加载模型
    logger.info(f"加载 VideoMAE 模型: {model_name}  dtype={torch_dtype}  device={device}")
    encoder = VideoMAEEncoder(model_name=model_name, device=device, torch_dtype=torch_dtype)
    logger.info("VideoMAE 模型加载完成")

    # 数据集列表
    datasets = cfg.get("datasets", [])
    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            logger.error(f"未找到数据集: {args.dataset}")
            sys.exit(1)

    # 逐数据集提取
    all_stats: Dict[str, Dict] = {}
    for ds_cfg in datasets:
        logger.info(f"========== 开始处理数据集: {ds_cfg['name']} ==========")
        stats = extract_dataset(
            encoder=encoder,
            dataset_cfg=ds_cfg,
            output_root=output_root,
            resume_cfg=resume_cfg,
            num_frames=num_frames,
            logger=logger,
            dry_run=args.dry_run,
        )
        all_stats[ds_cfg["name"]] = stats

    # 汇总
    logger.info("========== 全部完成 ==========")
    for ds_name, s in all_stats.items():
        logger.info(
            f"  {ds_name}: total={s['total']}  "
            f"success={s['success']}  skip={s['skip']}  fail={s['fail']}"
        )


if __name__ == "__main__":
    main()
