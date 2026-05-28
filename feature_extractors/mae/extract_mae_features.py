# -*- coding: utf-8 -*-
"""
文件名：extract_mae_features.py
功能：MAE（Masked Autoencoder，ViT-Large）视觉特征提取脚本

数据流：
  均匀采样 N 帧（默认 8 帧）[N, 3, 224, 224]
    → ViTMAEModel (facebook/vit-mae-large, mask_ratio=0)
    → last_hidden_state [N, 197, 1024]  (CLS + 196 patch tokens)
    → 取 patch tokens [:, 1:, :]        [N, 196, 1024]
    → 时空均值 mean(dim=(0,1))          [1024]  ← 保存此输出

输出格式：每视频一个 .pt 文件，features.shape = [1024]
对应模型层：Emotion-LLaMA 中 feats_llama_proj1 的输入

调用方式：
  python extract_mae_features.py --config config_mae.yaml
  python extract_mae_features.py --config config_mae.yaml --dataset-name CA-MER
"""

import os
import sys
import gc
import json
import logging
import time
import argparse
import traceback
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms

try:
    from torch.cuda import OutOfMemoryError
except ImportError:
    OutOfMemoryError = torch.cuda.OutOfMemoryError

from transformers import ViTMAEModel

# ============================================================================
# 路径配置
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "shared")

if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from feature_utils import (
    setup_logger,
    check_inode_usage,
    get_available_inodes,
    get_gpu_memory_usage,
    get_video_list,
    save_features_single,
    count_files_in_directory,
)

# ============================================================================
# 常量
# ============================================================================

# ImageNet 归一化（MAE 标准，区别于 BLIP2 归一化）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224    # MAE ViT-L/16 标准输入尺寸
DEFAULT_FPS        = 1.0    # 均匀采帧
DEFAULT_MAX_FRAMES = 8      # 默认 8 帧，逐帧提取后做时间均值
MIN_FPS            = 0.25

MAE_MODEL_NAME = "facebook/vit-mae-large"


# ============================================================================
# 图像预处理
# ============================================================================

def build_mae_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """MAE eval 预处理变换（ImageNet 归一化）。"""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def sample_representative_frame(
    video_path: str,
    vis_transform: transforms.Compose,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> torch.Tensor:
    """
    从视频中均匀采样至多 max_frames 帧，返回 [N, 3, H, W]。
    MAE 默认取 1 帧（中间帧），对应原论文的 local encoder 用法。
    """
    logger = logging.getLogger("feature_extraction")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        raise ValueError(f"无效视频 (frames={total_frames}, fps={video_fps}): {video_path}")

    duration_sec  = total_frames / video_fps
    n_sample      = min(max_frames, max(1, int(duration_sec * fps)))
    frame_indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)

    logger.debug(f"  时长={duration_sec:.1f}s, 采样{n_sample}帧 (fps={fps})")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(vis_transform(Image.fromarray(frame_rgb)))

    cap.release()

    if not frames:
        raise ValueError(f"未能提取任何帧: {video_path}")

    return torch.stack(frames)  # [N, 3, H, W]


# ============================================================================
# MAE Encoder
# ============================================================================

class MAEEncoder(nn.Module):
    """
    MAE ViT-Large 视觉编码器。

    forward 输出：patch token 均值 [1024]，
    对应 Emotion-LLaMA feats_llama_proj1 的输入。
    """

    def __init__(
        self,
        model_name: str = MAE_MODEL_NAME,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device      = device
        self.torch_dtype = torch_dtype

        logger = logging.getLogger("feature_extraction")
        logger.info(f"正在加载 MAE 模型: {model_name}...")

        self.model = ViTMAEModel.from_pretrained(model_name)
        # 推理时不需要随机掩码，设置 mask_ratio=0
        self.model.config.mask_ratio = 0.0

        self.model = self.model.to(device=device, dtype=torch_dtype).eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ MAE 加载完成，参数: {param_count:,}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [N, 3, 224, 224] float16

        Returns:
            features: [1024] — N 帧 patch tokens 的时空均值
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.torch_dtype):
                outputs = self.model(pixel_values)
                # last_hidden_state: [N, num_patches+1, 1024]  (包含 CLS)
                # 去除 CLS token，取 patch tokens 的均值
                patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [N, 196, 1024]
                features = patch_tokens.mean(dim=(0, 1))             # [1024]  时空均值
        return features


# ============================================================================
# 工具函数
# ============================================================================

def _is_oom_error(err: Exception) -> bool:
    if isinstance(err, OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _build_feature_filename(video_id: str, video_file: str) -> str:
    base_id = str(video_id) if video_id not in (None, "", "unknown") else ""
    if not base_id or "sample" not in base_id.lower():
        base_id = Path(video_file).stem if video_file else base_id
    safe_id = base_id.replace("/", "_").replace("\\", "_") or "sample_unknown"
    return f"{safe_id}_features.pt"


# ============================================================================
# 单视频特征提取
# ============================================================================

def extract_single_video_features(
    video_info: Dict,
    encoder: MAEEncoder,
    vis_transform: transforms.Compose,
    device: str,
    fps: float,
    max_frames: int,
) -> Tuple[Optional[torch.Tensor], float, str]:
    """Returns: (features, elapsed_sec, status)  status: "ok"|"oom"|"error" """
    logger     = logging.getLogger("feature_extraction")
    video_path = video_info["video_path"]
    video_id   = video_info.get("id", "unknown")

    start_time = time.time()
    frames = features = features_cpu = None

    try:
        frames       = sample_representative_frame(video_path, vis_transform, fps, max_frames)
        frames       = frames.to(device=device, dtype=encoder.torch_dtype)
        features     = encoder(frames)
        features_cpu = features.cpu()
        return features_cpu, time.time() - start_time, "ok"

    except Exception as e:
        elapsed = time.time() - start_time
        if _is_oom_error(e):
            logger.warning(f"  OOM [{video_id}] fps={fps}: {e}")
            return None, elapsed, "oom"
        logger.error(f"  错误 [{video_id}]: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return None, elapsed, "error"

    finally:
        for t in [frames, features]:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                del t
        del frames, features, features_cpu
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)


# ============================================================================
# 单数据集提取主逻辑
# ============================================================================

def extract_dataset(
    dataset_cfg: Dict,
    encoder: MAEEncoder,
    vis_transform: transforms.Compose,
    model_cfg: Dict,
    vision_cfg: Dict,
    saving_cfg: Dict,
    resume_cfg: Dict,
) -> None:
    logger       = logging.getLogger("feature_extraction")
    dataset_name = dataset_cfg["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"{'='*60}")

    video_list = get_video_list(
        dataset_path=dataset_cfg["dataset_path"],
        annotation_file=dataset_cfg.get("annotation_file"),
        video_dir=dataset_cfg.get("video_dir", "video"),
        extensions=dataset_cfg.get("supported_extensions"),
    )
    video_list = [v for v in video_list
                  if not os.path.basename(v.get("video_file", "")).startswith("._")]

    total_count = len(video_list)
    logger.info(f"找到 {total_count} 个视频")
    if total_count == 0:
        logger.error("未找到视频，跳过")
        return

    feature_output_dir = os.path.join(saving_cfg["output_root"], "MAE", dataset_name)
    os.makedirs(feature_output_dir, exist_ok=True)
    logger.info(f"输出目录: {feature_output_dir}")

    checkpoint_file = resume_cfg.get("checkpoint_file", "").replace("{dataset}", dataset_name)
    processed_videos: set = set()
    if resume_cfg.get("enabled", True) and checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            processed_videos = set(json.load(f).get("processed", []))
        logger.info(f"断点续传：跳过 {len(processed_videos)} 个视频")

    remaining = [v for v in video_list
                 if v["id"] not in processed_videos and v["video_file"] not in processed_videos]
    logger.info(f"待处理: {len(remaining)}/{total_count}")

    fps_value   = float(vision_cfg.get("fps", DEFAULT_FPS))
    max_frames  = int(vision_cfg.get("max_frames", DEFAULT_MAX_FRAMES))
    device      = model_cfg["device"]
    save_interval = resume_cfg.get("save_interval", 50)

    success_count = fail_count = 0
    total_infer   = 0.0
    start_time    = time.time()

    for idx, video_info in enumerate(remaining):
        video_id   = video_info.get("id", str(idx))
        video_file = video_info.get("video_file", "unknown")
        label      = video_info.get("label", "unknown")

        logger.info(f"[{idx+1}/{len(remaining)}] {video_file} (label={label})")

        attempt_fps = fps_value
        features = None
        elapsed  = -1.0
        error_type = "error"

        while True:
            gpu_before = get_gpu_memory_usage(device)
            features, elapsed, error_type = extract_single_video_features(
                video_info, encoder, vis_transform, device, attempt_fps, max_frames,
            )
            gpu_after = get_gpu_memory_usage(device)

            if features is not None:
                break
            if error_type == "oom":
                if attempt_fps <= MIN_FPS + 1e-6:
                    logger.error(f"  ❌ OOM 且 fps 已降至最低，跳过")
                    break
                next_fps = max(MIN_FPS, attempt_fps / 2.0)
                logger.warning(f"  ⚠ OOM：fps {attempt_fps} → {next_fps}")
                attempt_fps = next_fps
            else:
                break

        if features is not None:
            success_count += 1
            total_infer   += elapsed
            output_path = os.path.join(feature_output_dir,
                                       _build_feature_filename(video_id, video_file))
            save_features_single(features, output_path, metadata={
                "video_id":      video_id,
                "video_file":    video_file,
                "label":         label,
                "fps":           attempt_fps,
                "feature_shape": list(features.shape),
                "encoder_type":  "mae_vit_large",
            })
            logger.info(
                f"  ✅ shape={list(features.shape)} | {elapsed:.2f}s | "
                f"GPU {gpu_before:.2f}→{gpu_after:.2f}GB | 累计 {success_count}/{idx+1}"
            )
        else:
            fail_count += 1
            logger.error(f"  ❌ 失败 ({error_type}) | {elapsed:.2f}s | 累计失败 {fail_count}")

        if (
            resume_cfg.get("enabled", True)
            and checkpoint_file
            and (success_count + fail_count) % save_interval == 0
        ):
            progress = list(processed_videos | {v["id"] for v in remaining[:idx+1] if v["id"]})
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"processed": progress}, f, ensure_ascii=False, indent=2)

        if (success_count + fail_count) % 100 == 0:
            used, total, pct = check_inode_usage()
            logger.info(f"  inode: {used}/{total} ({pct:.1f}%)")

    total_elapsed = time.time() - start_time
    avg_time      = total_infer / success_count if success_count else 0
    logger.info(f"\n数据集 {dataset_name} 完成：成功 {success_count}，失败 {fail_count}，"
                f"总时长 {total_elapsed:.1f}s，均值 {avg_time:.3f}s/视频")
    logger.info(f"特征文件数: {count_files_in_directory(feature_output_dir)}")


# ============================================================================
# 主函数
# ============================================================================

def main(config_path: str, dataset_name_filter: Optional[str] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg  = config["model"]
    vision_cfg = config["vision_preprocess"]
    saving_cfg = config["feature_saving"]
    resume_cfg = config["resume"]
    log_cfg    = config["logging"]
    datasets   = config["datasets"]

    logger, log_file = setup_logger(log_cfg["output_dir"], "feature_extraction", log_cfg["level"])
    logger.info("=" * 70)
    logger.info("MAE 视觉特征提取工具 启动")
    logger.info(f"配置: {config_path}  |  日志: {log_file}")
    logger.info("=" * 70)

    used, total, pct = check_inode_usage()
    logger.info(f"inode: {used}/{total} ({pct:.1f}%)，剩余 {get_available_inodes()}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("precision", "float16"), torch.float16)

    encoder      = MAEEncoder(
        model_name=model_cfg.get("model_name", MAE_MODEL_NAME),
        device=model_cfg["device"],
        torch_dtype=torch_dtype,
    )
    vis_transform = build_mae_transform(model_cfg.get("image_size", DEFAULT_IMAGE_SIZE))

    for dataset_cfg in datasets:
        if dataset_name_filter and dataset_cfg["name"] != dataset_name_filter:
            continue
        extract_dataset(dataset_cfg, encoder, vis_transform,
                        model_cfg, vision_cfg, saving_cfg, resume_cfg)

    logger.info("\n所有数据集处理完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE 视觉特征提取工具")
    parser.add_argument("--config",
                        default=os.path.join(_SCRIPT_DIR, "config_mae.yaml"))
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()
    main(args.config, args.dataset_name)
