# -*- coding: utf-8 -*-
"""
文件名：extract_eva_clip_features.py
功能：EVA-CLIP 视觉特征提取脚本（不含任何 Emotion-LLaMA 投影层）

数据流：
  frame [N, 3, 448, 448]
    → visual_encoder (EVA-CLIP-G) → [N, 1025, 1408]
    → ln_vision (LayerNorm)       → [N, 1025, 1408]
    → 移除 CLS token              → [N, 1024, 1408]
    → 时间均值（多帧→单向量）     → [1024, 1408]   ← 保存此输出

输出格式：每视频一个 .pt 文件，features.shape = [1024, 1408]
对应模型层：llama_proj 的输入（reshape 后送入 llama_proj(5632→4096)）

调用方式：
  python extract_eva_clip_features.py --config config_eva_clip.yaml
  python extract_eva_clip_features.py --config config_eva_clip.yaml --dataset-name CA-MER
"""

import os
import sys
import gc
import json
import logging
import time
import argparse
import traceback
import importlib.util
import types
from typing import Dict, Optional, Tuple, List
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

# ============================================================================
# 路径配置：直接从 vendor/ 加载 eva_vit，绕过任何 package __init__.py
# ============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR  = os.path.join(_SCRIPT_DIR, "vendor")
_SHARED_DIR  = os.path.join(os.path.dirname(_SCRIPT_DIR), "shared")

# 将 shared/ 加入路径（用于 feature_utils）
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


def _load_module_from_file(dotted_name: str, fs_path: str) -> types.ModuleType:
    """直接从文件路径加载模块，幂等。"""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, fs_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = m
    spec.loader.exec_module(m)
    return m


# 加载 vendor 中的 dist_utils 和 eva_vit
_dist_utils_mod = _load_module_from_file(
    "vendor_dist_utils",
    os.path.join(_VENDOR_DIR, "dist_utils.py"),
)
# 让 eva_vit 能找到 dist_utils 的下载函数
sys.modules.setdefault("minigpt4", types.ModuleType("minigpt4"))
sys.modules.setdefault("minigpt4.common", types.ModuleType("minigpt4.common"))
sys.modules["minigpt4.common"].dist_utils = _dist_utils_mod
sys.modules["minigpt4.common.dist_utils"] = _dist_utils_mod

# 开启下载进度条
_orig_download = _dist_utils_mod.download_cached_file
def _download_with_progress(url, check_hash=True, progress=True):
    return _orig_download(url, check_hash=check_hash, progress=progress)
_dist_utils_mod.download_cached_file = _download_with_progress

_eva_vit_mod      = _load_module_from_file(
    "vendor_eva_vit",
    os.path.join(_VENDOR_DIR, "eva_vit.py"),
)
_create_eva_vit_g = _eva_vit_mod.create_eva_vit_g


# ============================================================================
# FP16 安全 LayerNorm
# ============================================================================

class _FP16LayerNorm(nn.LayerNorm):
    """强制在 float32 精度下执行 LayerNorm，输出转回原始 dtype。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


# ============================================================================
# 图像预处理
# ============================================================================

BLIP2_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP2_STD  = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_IMAGE_SIZE = 448
DEFAULT_FPS        = 1.0
DEFAULT_MAX_FRAMES = 16
MIN_FPS            = 0.25


def build_vis_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """BLIP2 eval 预处理变换（与 Emotion-LLaMA 训练一致）。"""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=BLIP2_MEAN, std=BLIP2_STD),
    ])


def preprocess_video_to_frames(
    video_path: str,
    vis_transform: transforms.Compose,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> torch.Tensor:
    """
    OpenCV 均匀采样帧 → BLIP2 transforms → [N, 3, H, W]。

    Returns:
        frames_tensor: float32 tensor [N, 3, image_size, image_size]
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

    logger.debug(f"  时长={duration_sec:.1f}s, 视频fps={video_fps:.1f}, 采样{n_sample}帧")

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
# EVA-CLIP Encoder
# ============================================================================

class EVACLIPEncoder(nn.Module):
    """
    EVA-CLIP-G 视觉编码器（不含任何 Emotion-LLaMA 投影层）。

    forward 输出：时间均值后的 patch tokens [1024, 1408]，
    对应 Emotion-LLaMA 架构中 llama_proj 的输入（reshape 前）。
    """

    def __init__(
        self,
        img_size: int = DEFAULT_IMAGE_SIZE,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device     = device
        self.torch_dtype = torch_dtype

        logger = logging.getLogger("feature_extraction")
        logger.info("正在加载 EVA-CLIP-G（无投影层）...")

        self.visual_encoder = _create_eva_vit_g(
            img_size=img_size,
            drop_path_rate=0,
            use_checkpoint=False,
            precision="fp16",
        )
        num_features = self.visual_encoder.num_features  # 1408
        self.ln_vision = _FP16LayerNorm(num_features)

        # 移至目标设备并设为 eval
        self.visual_encoder = self.visual_encoder.to(device=device, dtype=torch_dtype).eval()
        self.ln_vision       = self.ln_vision.to(device=device).eval()

        vit_params = sum(p.numel() for p in self.visual_encoder.parameters())
        logger.info(f"✅ EVA-CLIP-G 加载完成，参数: {vit_params:,}，num_features={num_features}")

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_batch: [N, 3, H, W] float16，N 为采样帧数

        Returns:
            features: [1024, 1408] — 时间均值后的 patch tokens（CLS 已移除）
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = self.visual_encoder(image_batch)   # [N, 1025, 1408]
                feats = self.ln_vision(feats)               # [N, 1025, 1408]
                feats = feats[:, 1:, :]                     # [N, 1024, 1408]  移除 CLS
                features = feats.mean(dim=0)                # [1024, 1408]  时间均值
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
    encoder: EVACLIPEncoder,
    vis_transform: transforms.Compose,
    device: str,
    fps: float,
    max_frames: int,
) -> Tuple[Optional[torch.Tensor], float, str]:
    """
    Returns:
        (features, elapsed_sec, status)  status: "ok" | "oom" | "error"
    """
    logger     = logging.getLogger("feature_extraction")
    video_path = video_info["video_path"]
    video_id   = video_info.get("id", "unknown")

    start_time = time.time()
    frames_tensor = features = features_cpu = None

    try:
        frames_tensor = preprocess_video_to_frames(video_path, vis_transform, fps, max_frames)
        frames_tensor = frames_tensor.to(device=device, dtype=torch.float16)
        features      = encoder(frames_tensor)
        features_cpu  = features.cpu()
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
        for t in [frames_tensor, features]:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                del t
        del frames_tensor, features, features_cpu
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)


# ============================================================================
# 单数据集提取主逻辑
# ============================================================================

def extract_dataset(
    dataset_cfg: Dict,
    encoder: EVACLIPEncoder,
    vis_transform: transforms.Compose,
    model_cfg: Dict,
    vision_cfg: Dict,
    saving_cfg: Dict,
    resume_cfg: Dict,
) -> None:
    """对单个数据集执行完整的特征提取流程。"""
    logger      = logging.getLogger("feature_extraction")
    dataset_name = dataset_cfg["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"{'='*60}")

    # 获取视频列表
    video_list = get_video_list(
        dataset_path=dataset_cfg["dataset_path"],
        annotation_file=dataset_cfg.get("annotation_file"),
        video_dir=dataset_cfg.get("video_dir", "video"),
        extensions=dataset_cfg.get("supported_extensions"),
    )
    # 过滤 macOS 隐藏文件
    video_list = [v for v in video_list
                  if not os.path.basename(v.get("video_file", "")).startswith("._")]

    total_count = len(video_list)
    logger.info(f"找到 {total_count} 个视频")
    if total_count == 0:
        logger.error("未找到视频，跳过该数据集")
        return

    # 输出目录：{output_root}/EVA-CLIP/{dataset_name}/
    feature_output_dir = os.path.join(saving_cfg["output_root"], "EVA-CLIP", dataset_name)
    os.makedirs(feature_output_dir, exist_ok=True)
    logger.info(f"输出目录: {feature_output_dir}")

    # 断点续传
    checkpoint_file = resume_cfg.get("checkpoint_file", "").replace(
        "{dataset}", dataset_name
    )
    processed_videos: set = set()
    if resume_cfg.get("enabled", True) and checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            processed_videos = set(json.load(f).get("processed", []))
        logger.info(f"断点续传：已跳过 {len(processed_videos)} 个视频")

    remaining = [v for v in video_list
                 if v["id"] not in processed_videos and v["video_file"] not in processed_videos]
    logger.info(f"待处理: {len(remaining)}/{total_count}")

    fps_value  = float(vision_cfg.get("fps", DEFAULT_FPS))
    max_frames = int(vision_cfg.get("max_frames", DEFAULT_MAX_FRAMES))
    device     = model_cfg["device"]

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
                    logger.error(f"  ❌ OOM 且 fps 已降至 {attempt_fps}，跳过")
                    break
                next_fps = max(MIN_FPS, attempt_fps / 2.0)
                logger.warning(f"  ⚠ OOM：fps {attempt_fps} → {next_fps}")
                attempt_fps = next_fps
            else:
                break

        if features is not None:
            success_count += 1
            total_infer   += elapsed

            output_name = _build_feature_filename(video_id, video_file)
            output_path = os.path.join(feature_output_dir, output_name)
            save_features_single(features, output_path, metadata={
                "video_id":      video_id,
                "video_file":    video_file,
                "label":         label,
                "fps":           attempt_fps,
                "feature_shape": list(features.shape),
                "encoder_type":  "eva_clip_g",
            })
            logger.info(
                f"  ✅ shape={list(features.shape)} | {elapsed:.2f}s | "
                f"GPU {gpu_before:.2f}→{gpu_after:.2f}GB | 累计 {success_count}/{idx+1}"
            )
        else:
            fail_count += 1
            logger.error(f"  ❌ 失败 ({error_type}) | {elapsed:.2f}s | 累计失败 {fail_count}")

        # 断点续传写入
        save_interval = resume_cfg.get("save_interval", 50)
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
    logger.info(f"特征文件: {count_files_in_directory(feature_output_dir)}")


# ============================================================================
# 主函数
# ============================================================================

def main(config_path: str, dataset_name_filter: Optional[str] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg   = config["model"]
    vision_cfg  = config["vision_preprocess"]
    saving_cfg  = config["feature_saving"]
    resume_cfg  = config["resume"]
    log_cfg     = config["logging"]
    datasets    = config["datasets"]

    logger, log_file = setup_logger(log_cfg["output_dir"], "feature_extraction", log_cfg["level"])
    logger.info("=" * 70)
    logger.info("EVA-CLIP 视觉特征提取工具 启动")
    logger.info(f"配置文件: {config_path}  |  日志: {log_file}")
    logger.info("=" * 70)

    used, total, pct = check_inode_usage()
    avail = get_available_inodes()
    logger.info(f"inode: {used}/{total} ({pct:.1f}%)，剩余 {avail}")
    if 0 < avail < 100000:
        logger.warning(f"⚠ inode 剩余不足 100000 ({avail})")

    # 加载编码器
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("precision", "fp16"), torch.float16)

    encoder = EVACLIPEncoder(
        img_size=model_cfg.get("img_size", DEFAULT_IMAGE_SIZE),
        device=model_cfg["device"],
        torch_dtype=torch_dtype,
    )
    vis_transform = build_vis_transform(model_cfg.get("img_size", DEFAULT_IMAGE_SIZE))

    # 逐数据集提取
    for dataset_cfg in datasets:
        if dataset_name_filter and dataset_cfg["name"] != dataset_name_filter:
            continue
        extract_dataset(
            dataset_cfg, encoder, vis_transform,
            model_cfg, vision_cfg, saving_cfg, resume_cfg,
        )

    logger.info("\n所有数据集处理完成。")


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVA-CLIP 视觉特征提取工具")
    parser.add_argument(
        "--config",
        default=os.path.join(_SCRIPT_DIR, "config_eva_clip.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="只处理指定名称的数据集（不填则处理所有）",
    )
    args = parser.parse_args()
    main(args.config, args.dataset_name)
