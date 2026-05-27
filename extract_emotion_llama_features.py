# -*- coding: utf-8 -*-
"""
文件名：extract_emotion_llama_features.py
功能：Emotion-LLaMA 视觉特征提取主脚本
创建时间：2025-05-27

与 emotion_qwen_feature_extractor/extract_emotion_qwen_features.py 完全类比：
  Qwen 提取器：Emotion-Qwen → vpm + compressors + gatenet → [1344, 3584]
  LLaMA 提取器：Emotion-LLaMA → EVA-CLIP + ln_vision + llama_proj → [256, 4096]

调用方式:
  1) 使用配置文件默认参数:
      python extract_emotion_llama_features.py --config feature_extraction_config_llama.yaml
  2) 指定不同数据集与输出目录:
      python extract_emotion_llama_features.py --config feature_extraction_config_llama.yaml \\
         --dataset /path/to/dataset --output-dir /path/to/output
  3) 覆盖支持的视频格式(逗号分隔):
      python extract_emotion_llama_features.py --config feature_extraction_config_llama.yaml \\
         --extensions .mp4,.avi,.mkv
  4) 指定标注文件/视频子目录:
      python extract_emotion_llama_features.py --config feature_extraction_config_llama.yaml \\
         --annotation-file video-aligned.json --video-dir video-aligned

本脚本实现以下功能：
1. 直接加载 EVA-CLIP-G（通过 create_eva_vit_g），无需加载 LLaMA 主干（节省 ~13GB 显存）
2. 从 Emotion-LLaMA 微调 checkpoint 加载 ln_vision、llama_proj 权重
3. 视频均匀采样帧 → Blip2 图像预处理(448×448) → 编码 → 多帧时间平均
4. 保存格式与 Qwen 提取器完全一致（.pt 文件，含 features/feature_shape/metadata/timestamp）
5. 支持断点续传和 inode 监控

数据流（对应 minigpt_v2.py encode_img() 第 92-116 行）：
  frame[N,3,448,448] → visual_encoder(EVA-CLIP) → [N,1025,1408]
                     → ln_vision → [N,1025,1408]
                     → 去CLS → [N,1024,1408]
                     → reshape → [N,256,5632]
                     → llama_proj → [N,256,4096]
                     → 时间均值 → [256,4096]

参考源：
- Emotion-LLaMA-main/minigpt4/models/minigpt_v2.py (encode_img 方法)
- Emotion-LLaMA-main/minigpt4/models/base_model.py (init_vision_encoder, LayerNorm)
- Emotion-LLaMA-main/minigpt4/models/eva_vit.py (create_eva_vit_g)
- Emotion-LLaMA-main/minigpt4/conversation/conversation.py (get_first_frame)
- Emotion-LLaMA-main/minigpt4/processors/blip_processors.py (归一化参数)
"""

import os
import sys
import gc
import json
import logging
import time
import argparse
import traceback
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
# 路径配置（确保可以导入 minigpt4 和 feature_utils）
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 导入 minigpt4 框架（Emotion-LLaMA-main 目录）
_EMOTION_LLAMA_DIR = os.path.join(_SCRIPT_DIR, "Emotion-LLaMA-main")
if _EMOTION_LLAMA_DIR not in sys.path:
    sys.path.insert(0, _EMOTION_LLAMA_DIR)

# 导入 feature_utils（Qwen 提取器工具函数，格式完全复用）
_UTILS_DIR = os.path.join(_SCRIPT_DIR, "emotion_qwen_feature_extractor")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

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
# 常量定义
# ============================================================================

# EVA-CLIP-G 输出维度（来源：eva_vit.py VisionTransformer embed_dim=1408）
EVA_CLIP_NUM_FEATURES = 1408

# llama_proj 输出维度（来源：LLaMA-2-7B config.hidden_size = 4096）
LLAMA2_HIDDEN_SIZE = 4096

# llama_proj 输入维度（来源：minigpt_v2.py img_f_dim = num_features * 4 = 5632）
LLAMA_PROJ_INPUT_DIM = EVA_CLIP_NUM_FEATURES * 4  # = 5632

# BLIP2/CLIP 标准归一化参数（来源：blip_processors.py 第 21-24 行）
BLIP2_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP2_STD = (0.26862954, 0.26130258, 0.27577711)

# 默认图像大小（来源：train_configs/Emotion-LLaMA_finetune.yaml image_size: 448）
DEFAULT_IMAGE_SIZE = 448

# 默认帧率（EVA-CLIP 是图像模型，1fps 通常已足够）
DEFAULT_FPS = 1.0

# 默认最大帧数
DEFAULT_MAX_FRAMES = 16

# OOM 重试最小帧率
MIN_FPS = 0.25


# ============================================================================
# Emotion-LLaMA 视觉 Encoder 封装类
# ============================================================================

class EmotionLlamaEncoder(nn.Module):
    """
    Emotion-LLaMA 视觉 Encoder 封装（只保留视觉部分，不加载 LLaMA 主干）。

    架构（对应 minigpt_v2.py + base_model.py）：
      visual_encoder: EVA-CLIP-G (Emotion_ViT)，输出 [N, 1025, 1408]
      ln_vision:      LayerNorm(1408)，来自 base_model.LayerNorm
      llama_proj:     Linear(5632→4096)，将视觉特征投影到 LLM 嵌入空间

    与 EmotionQwenEncoder 的对应关系：
      EmotionQwenEncoder.vpm              ↔  EmotionLlamaEncoder.visual_encoder
      EmotionQwenEncoder.general/emotion  ↔  无（LLaMA 版只有单路投影）
      EmotionQwenEncoder.gatenet          ↔  无
      EmotionQwenEncoder.forward          ↔  EmotionLlamaEncoder.forward

    优化：直接加载 EVA-CLIP 权重 + 从 checkpoint 提取 ln_vision/llama_proj 权重，
    无需加载 LLaMA-2-7B（节省 ~13GB VRAM）。

    参考源：
    - minigpt_v2.py __init__() 和 encode_img()
    - base_model.py init_vision_encoder() 和 LayerNorm
    - eva_vit.py create_eva_vit_g()
    """

    def __init__(
        self,
        ckpt_path: str = "",
        img_size: int = DEFAULT_IMAGE_SIZE,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        use_llama_proj: bool = True,
    ):
        """
        初始化并加载视觉模块。

        Args:
            ckpt_path:       Emotion-LLaMA 微调 checkpoint 路径（minigptv2_checkpoint.pth）
                             用于加载 ln_vision 和 llama_proj 的微调权重。
                             若为空则使用随机初始化（不推荐用于特征提取）。
            img_size:        图像大小（448，与训练配置一致）
            device:          计算设备
            torch_dtype:     数据类型（float16，与官方训练配置一致）
            use_llama_proj:  是否应用 llama_proj 投影层。
                             True  → 输出 [256, 4096]（LLM 嵌入空间，与训练时一致）
                             False → 输出 [1024, 1408]（EVA-CLIP 原始 patch tokens）
        """
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.use_llama_proj = use_llama_proj

        logger = logging.getLogger("feature_extraction")
        logger.info("=" * 60)
        logger.info("正在加载 Emotion-LLaMA 视觉 Encoder...")
        logger.info("  策略：直接加载 EVA-CLIP + checkpoint 视觉权重，跳过 LLaMA 主干")
        logger.info("=" * 60)

        # ----------------------------------------------------------------
        # 步骤1：加载 EVA-CLIP-G（通过 create_eva_vit_g，自动下载/缓存权重）
        # ----------------------------------------------------------------
        logger.info("步骤1: 加载 EVA-CLIP-G 视觉编码器...")
        try:
            from minigpt4.models.eva_vit import create_eva_vit_g
            from minigpt4.models.base_model import LayerNorm
        except ImportError as e:
            raise ImportError(
                f"无法导入 minigpt4 模块：{e}\n"
                f"请确认 Emotion-LLaMA-main 目录存在于：{_EMOTION_LLAMA_DIR}"
            )

        # create_eva_vit_g 会自动从网络下载 eva_vit_g.pth 并缓存
        # precision="fp16" 对应官方训练时的 vit_precision="fp16"
        self.visual_encoder = create_eva_vit_g(
            img_size=img_size,
            drop_path_rate=0,       # 推理时不需要 drop path
            use_checkpoint=False,   # 推理时不需要 gradient checkpoint
            precision="fp16",       # 来源：minigpt_v2.yaml vit_precision: fp16
        )
        num_features = self.visual_encoder.num_features  # 1408

        vit_params = sum(p.numel() for p in self.visual_encoder.parameters())
        logger.info(f"  ✅ EVA-CLIP-G 加载完成，参数: {vit_params:,}，num_features={num_features}")

        # ----------------------------------------------------------------
        # 步骤2：创建 ln_vision 和 llama_proj（结构与 minigpt_v2.py 完全一致）
        # ----------------------------------------------------------------
        logger.info("步骤2: 创建 ln_vision 和 llama_proj...")
        self.ln_vision = LayerNorm(num_features)  # LayerNorm(1408)，支持 fp16

        if self.use_llama_proj:
            img_f_dim = num_features * 4  # 5632，来源：minigpt_v2.py 第68行
            self.llama_proj = nn.Linear(img_f_dim, LLAMA2_HIDDEN_SIZE)  # Linear(5632, 4096)
            logger.info(f"  ✅ llama_proj: Linear({img_f_dim}, {LLAMA2_HIDDEN_SIZE})")
        else:
            self.llama_proj = None
            logger.info("  ℹ️  use_llama_proj=False，将输出 EVA-CLIP 原始 patch tokens [1024, 1408]")

        # ----------------------------------------------------------------
        # 步骤3：从 Emotion-LLaMA checkpoint 加载微调后的 ln_vision 和 llama_proj 权重
        # ----------------------------------------------------------------
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"步骤3: 从 checkpoint 加载视觉权重: {ckpt_path}")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("model", ckpt)

                # 提取 ln_vision 权重
                ln_state = {
                    k[len("ln_vision."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("ln_vision.")
                }
                if ln_state:
                    self.ln_vision.load_state_dict(ln_state, strict=True)
                    logger.info(f"  ✅ ln_vision 权重加载成功（{len(ln_state)} 个 key）")
                else:
                    logger.warning("  ⚠️  checkpoint 中未找到 ln_vision 权重，使用随机初始化")

                # 提取 llama_proj 权重
                if self.use_llama_proj:
                    proj_state = {
                        k[len("llama_proj."):]: v
                        for k, v in state_dict.items()
                        if k.startswith("llama_proj.")
                    }
                    if proj_state:
                        self.llama_proj.load_state_dict(proj_state, strict=True)
                        logger.info(f"  ✅ llama_proj 权重加载成功（{len(proj_state)} 个 key）")
                    else:
                        logger.warning("  ⚠️  checkpoint 中未找到 llama_proj 权重，使用随机初始化")

                del ckpt, state_dict
                gc.collect()

            except Exception as e:
                logger.error(f"  ❌ checkpoint 加载失败: {e}")
                logger.error(traceback.format_exc())
                raise
        elif ckpt_path:
            logger.warning(f"  ⚠️  checkpoint 路径不存在: {ckpt_path}，使用随机初始化权重")
        else:
            logger.info("步骤3: 未指定 checkpoint，ln_vision/llama_proj 使用默认初始化")

        # ----------------------------------------------------------------
        # 步骤4：移动到目标设备并设为 eval 模式
        # ----------------------------------------------------------------
        logger.info(f"步骤4: 移动到设备 {device}，dtype={torch_dtype}...")
        self.visual_encoder = self.visual_encoder.to(device=device, dtype=torch_dtype).eval()
        self.ln_vision = self.ln_vision.to(device=device).eval()
        if self.llama_proj is not None:
            self.llama_proj = self.llama_proj.to(device=device, dtype=torch_dtype).eval()

        logger.info("✅ Emotion-LLaMA Encoder 加载完成")

    @staticmethod
    def _count_params(module) -> int:
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        视觉 Encoder 前向传播（严格复现 minigpt_v2.py encode_img() 的视觉分支）。

        数据流（对应 encode_img 第 98-105 行）：
          image_batch [N, 3, H, W]
            → visual_encoder → [N, 1025, 1408]  (1 CLS + 1024 patch tokens)
            → ln_vision       → [N, 1025, 1408]
            → 去除 CLS        → [N, 1024, 1408]
            → reshape(4合1)   → [N, 256, 5632]
            → llama_proj      → [N, 256, 4096]
            → 时间均值        → [256, 4096]       ← 保存此输出

        若 use_llama_proj=False:
          → 去除 CLS         → [N, 1024, 1408]
          → 时间均值         → [1024, 1408]

        Args:
            image_batch: 预处理后的图像批次 [N, 3, H, W]，dtype=float16

        Returns:
            features: 时间均值后的特征 [256, 4096] 或 [1024, 1408]
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 步骤1: EVA-CLIP 视觉编码（对应 encode_img 第 98 行）
                image_feats = self.visual_encoder(image_batch)   # [N, 1025, 1408]

                # 步骤2: Layer Norm（对应 encode_img 第 99 行）
                image_embeds = self.ln_vision(image_feats)       # [N, 1025, 1408]

                # 步骤3: 去除 CLS token（对应 encode_img 第 102 行）
                image_embeds = image_embeds[:, 1:, :]             # [N, 1024, 1408]

                if self.use_llama_proj:
                    # 步骤4: 空间 4合1 reshape（对应 encode_img 第 103-104 行）
                    bs, pn, hs = image_embeds.shape               # N, 1024, 1408
                    image_embeds = image_embeds.view(bs, pn // 4, hs * 4)  # [N, 256, 5632]

                    # 步骤5: llama_proj 线性投影（对应 encode_img 第 105 行）
                    image_embeds = self.llama_proj(image_embeds)  # [N, 256, 4096]

                # 步骤6: 时间维度平均（多帧 → 单特征）
                features = image_embeds.mean(dim=0)               # [256, 4096] or [1024, 1408]

        return features


# ============================================================================
# 视频预处理函数
# ============================================================================

def build_vis_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    构建视觉预处理变换（eval 模式，对应 blip2_image_eval）。

    使用 BLIP2/CLIP 标准归一化参数（来源：blip_processors.py 第 21-24 行）。

    Args:
        image_size: 目标图像大小（默认 448，与训练配置一致）

    Returns:
        torchvision.transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC
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
    视频预处理：均匀采样帧 → PIL Image → vis_transform → [N, 3, H, W]。
    与 Qwen 版 preprocess_video_to_tensor() 类比（Qwen 用 process_vision_info，
    LLaMA 版用 OpenCV + torchvision transforms，与 conversation.py get_first_frame() 一致）。

    Args:
        video_path:    视频文件路径
        vis_transform: BLIP2 eval 预处理变换（build_vis_transform() 构建）
        fps:           每秒采样帧数（EVA-CLIP 是图像模型，1fps 通常足够）
        max_frames:    最多采样帧数

    Returns:
        frames_tensor: [N, 3, image_size, image_size] 的 float32 张量

    Raises:
        IOError: 无法打开视频
        ValueError: 视频帧数为 0 或无法读取任何帧
    """
    logger = logging.getLogger("feature_extraction")

    # 打开视频（与 conversation.py get_first_frame() 使用相同的 cv2 接口）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        raise ValueError(f"无效视频（total_frames={total_frames}, fps={video_fps}）: {video_path}")

    # 根据 fps 计算采样帧数
    duration_sec = total_frames / video_fps
    n_sample = min(max_frames, max(1, int(duration_sec * fps)))

    # 均匀采样帧索引
    frame_indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)
    logger.debug(f"  视频时长: {duration_sec:.1f}s, 视频帧率: {video_fps:.1f}fps, "
                 f"采样: {n_sample}帧 (fps={fps})")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # BGR → RGB（与 conversation.py 第 257 行 cv2.cvtColor 一致）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(vis_transform(pil_image))

    cap.release()

    if not frames:
        raise ValueError(f"未能从视频中提取任何帧: {video_path}")

    return torch.stack(frames)  # [N, 3, image_size, image_size]


# ============================================================================
# OOM 检测与工具函数
# ============================================================================

def _is_oom_error(err: Exception) -> bool:
    """检测是否为显存不足错误（与 Qwen 版完全一致）。"""
    if isinstance(err, OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _format_fps_label(fps: float) -> str:
    """格式化 fps 标签（与 Qwen 版完全一致）。"""
    fps_str = f"{fps:.2f}".rstrip("0").rstrip(".")
    return fps_str if fps_str else str(fps)


def _build_feature_filename(video_id: str, video_file: str, fps: float) -> str:
    """
    构建特征文件名（与 Qwen 版完全一致）。
    格式：{video_id}_fps{fps}_features.pt
    """
    base_id = str(video_id) if video_id not in (None, "", "unknown") else ""
    if not base_id or "sample" not in base_id.lower():
        base_id = Path(video_file).stem if video_file else base_id
    safe_id = base_id.replace("/", "_").replace("\\", "_")
    if not safe_id:
        safe_id = "sample_unknown"
    fps_label = _format_fps_label(fps)
    return f"{safe_id}_fps{fps_label}_features.pt"


# ============================================================================
# 主特征提取函数
# ============================================================================

def extract_single_video_features(
    video_info: Dict,
    encoder: EmotionLlamaEncoder,
    vis_transform: transforms.Compose,
    device: str,
    fps: float,
    max_frames: int,
) -> Tuple[Optional[torch.Tensor], float, str]:
    """
    对单个视频进行视觉特征提取（与 Qwen 版 extract_single_video_features 类比）。

    流程：
    1. 视频帧预处理（OpenCV 均匀采样 + BLIP2 归一化）
    2. Encoder 前向传播（EVA-CLIP → ln_vision → llama_proj → 时间均值）
    3. 提取特征到 CPU

    OOM 策略：降低 fps 重试（与 Qwen 版降 fps 逻辑完全一致）

    Args:
        video_info:    视频信息字典（含 video_path 等键）
        encoder:       EmotionLlamaEncoder 实例
        vis_transform: BLIP2 图像预处理变换
        device:        计算设备
        fps:           抽帧帧率
        max_frames:    最大帧数

    Returns:
        (features, elapsed_time, error_type)：特征张量 [256, 4096] 和处理时间（秒），
        失败返回 (None, elapsed_time, "oom"|"error")
    """
    logger = logging.getLogger("feature_extraction")
    video_path = video_info["video_path"]
    video_id = video_info.get("id", "unknown")

    start_time = time.time()

    frames_tensor = features = features_cpu = None
    try:
        # 步骤1: 视频预处理（OpenCV 均匀采样帧 + BLIP2 transforms）
        frames_tensor = preprocess_video_to_frames(
            video_path=video_path,
            vis_transform=vis_transform,
            fps=fps,
            max_frames=max_frames,
        )

        # 步骤2: 移动到 GPU（float16，与 EVA-CLIP 精度一致）
        frames_tensor = frames_tensor.to(device=device, dtype=torch.float16)

        # 步骤3: Encoder 前向传播（EVA-CLIP → ln_vision → llama_proj → 时间均值）
        features = encoder(frames_tensor)

        # 步骤4: 移至 CPU
        features_cpu = features.cpu()

        elapsed = time.time() - start_time
        return features_cpu, elapsed, "ok"

    except Exception as e:
        elapsed = time.time() - start_time
        if _is_oom_error(e):
            logger.warning(f"  视频 {video_id} 在 fps={fps} 时显存不足: {type(e).__name__}: {e}")
            logger.debug(f"  错误堆栈: {traceback.format_exc()}")
            return None, elapsed, "oom"
        logger.error(f"  视频 {video_id} 处理失败: {type(e).__name__}: {e}")
        logger.debug(f"  错误堆栈: {traceback.format_exc()}")
        return None, elapsed, "error"

    finally:
        # 无论成功或失败，强制清理所有 GPU 张量，防止碎片积累
        for t in [frames_tensor, features]:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                del t
        del frames_tensor, features, features_cpu
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)


# ============================================================================
# 主函数
# ============================================================================

def main(
    config_path: str,
    dataset_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    extensions_override: Optional[str] = None,
    annotation_file_override: Optional[str] = None,
    video_dir_override: Optional[str] = None,
):
    """
    特征提取主函数（结构与 Qwen 版 main() 完全一致）。

    流程：
    1. 加载 YAML 配置文件
    2. 检查 inode 状态
    3. 初始化日志
    4. 加载 Emotion-LLaMA 视觉 Encoder（仅 EVA-CLIP + ln_vision + llama_proj）
    5. 构建视觉预处理变换（BLIP2 eval 模式）
    6. 获取视频列表
    7. 逐视频提取特征并保存（.pt 格式，与 Qwen 版完全一致）
    8. 输出统计信息
    """
    # ====================== 加载配置 ======================
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg      = config["model"]
    vision_cfg     = config["vision_preprocess"]
    data_cfg       = config["data"]
    saving_cfg     = config["feature_saving"]
    resume_cfg     = config["resume"]
    log_cfg        = config["logging"]

    # ====================== 初始化日志 ======================
    logger, log_file = setup_logger(
        log_dir=log_cfg["output_dir"],
        name="feature_extraction",
        level=log_cfg["level"],
    )
    logger.info("=" * 80)
    logger.info("Emotion-LLaMA 视觉特征提取工具 启动")
    logger.info("=" * 80)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"日志文件: {log_file}")

    # ====================== inode 检查 ======================
    used_inodes, total_inodes, usage_pct = check_inode_usage()
    available_inodes = get_available_inodes()
    logger.info(f"inode 状态: 已用 {used_inodes}/{total_inodes} ({usage_pct:.1f}%), 剩余 {available_inodes}")
    if available_inodes > 0 and available_inodes < 100000:
        logger.warning(f"⚠ 剩余 inode 不足 100000 个 ({available_inodes})，建议先清理不必要的文件!")

    # ====================== 命令行参数覆盖 ======================
    if dataset_override:
        data_cfg["dataset_path"] = dataset_override
        logger.info(f"数据集路径覆盖: {data_cfg['dataset_path']}")
    if output_dir_override:
        saving_cfg["output_dir"] = output_dir_override
        logger.info(f"特征输出目录覆盖: {saving_cfg['output_dir']}")
    if annotation_file_override is not None:
        data_cfg["annotation_file"] = annotation_file_override
        logger.info(f"标注文件覆盖: {data_cfg['annotation_file']}")
    if video_dir_override is not None:
        data_cfg["video_dir"] = video_dir_override
        logger.info(f"视频子目录覆盖: {data_cfg['video_dir']}")
    if extensions_override:
        raw_exts = [e.strip() for e in extensions_override.split(",") if e.strip()]
        normalized_exts = [
            (ext if ext.startswith(".") else f".{ext}").lower()
            for ext in raw_exts
        ]
        data_cfg["supported_extensions"] = normalized_exts
        logger.info(f"支持的视频后缀覆盖: {normalized_exts}")

    # ====================== 加载 Emotion-LLaMA 视觉 Encoder ======================
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "float16"), torch.float16)

    logger.info("正在加载 Emotion-LLaMA 视觉 Encoder（EVA-CLIP + ln_vision + llama_proj）...")
    logger.info(f"  checkpoint: {model_cfg.get('ckpt', '（未指定）')}")
    logger.info(f"  设备: {model_cfg['device']}, dtype: {model_cfg.get('torch_dtype', 'float16')}")
    logger.info(f"  图像大小: {vision_cfg.get('image_size', DEFAULT_IMAGE_SIZE)}")
    logger.info(f"  use_llama_proj: {model_cfg.get('use_llama_proj', True)}")

    try:
        encoder = EmotionLlamaEncoder(
            ckpt_path=model_cfg.get("ckpt", ""),
            img_size=vision_cfg.get("image_size", DEFAULT_IMAGE_SIZE),
            device=model_cfg["device"],
            torch_dtype=torch_dtype,
            use_llama_proj=model_cfg.get("use_llama_proj", True),
        )
        logger.info("✅ Encoder 加载成功")
    except Exception as e:
        logger.error(f"❌ Encoder 加载失败: {e}")
        logger.error(traceback.format_exc())
        return

    # ====================== 构建视觉预处理变换 ======================
    vis_transform = build_vis_transform(vision_cfg.get("image_size", DEFAULT_IMAGE_SIZE))
    logger.info(f"视觉预处理: Resize→{vision_cfg.get('image_size', DEFAULT_IMAGE_SIZE)}×{vision_cfg.get('image_size', DEFAULT_IMAGE_SIZE)}, "
                f"Normalize(BLIP2 mean/std)")

    # ====================== 获取视频列表 ======================
    logger.info(f"正在扫描视频列表... 数据集路径: {data_cfg['dataset_path']}")
    video_list = get_video_list(
        dataset_path=data_cfg["dataset_path"],
        annotation_file=data_cfg.get("annotation_file"),
        video_dir=data_cfg.get("video_dir", "video"),
        extensions=data_cfg.get("supported_extensions"),
    )
    total_count = len(video_list)
    logger.info(f"✅ 找到 {total_count} 个视频文件")

    if total_count == 0:
        logger.error("未找到任何视频文件，退出")
        return

    # ====================== 准备输出目录 ======================
    # 目录结构：{output_dir}/{dataset_name}/
    # 示例：.../EmotionLlamaEncoder/CA-MER/sample_001_fps1_features.pt
    dataset_name = os.path.basename(data_cfg["dataset_path"].rstrip("/\\"))
    feature_output_dir = os.path.join(saving_cfg["output_dir"], dataset_name)
    os.makedirs(feature_output_dir, exist_ok=True)

    logger.info(f"特征输出目录: {feature_output_dir}")
    logger.info(f"预计产生: {total_count} 个特征文件")
    if total_count > 10000:
        logger.warning(f"⚠ 预计文件数 ({total_count}) 超过 10000，请确认 inode 容量")

    # ====================== 断点续传 ======================
    checkpoint_file = resume_cfg.get("checkpoint_file", "")
    processed_videos: set = set()
    if resume_cfg.get("enabled", True) and checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            processed_videos = set(checkpoint_data.get("processed", []))
        logger.info(f"断点续传：已处理 {len(processed_videos)} 个视频，将跳过")

    remaining_videos = [
        v for v in video_list
        if v["id"] not in processed_videos and v["video_file"] not in processed_videos
    ]
    logger.info(f"待处理: {len(remaining_videos)}/{total_count} 个视频")

    # ====================== 开始特征提取 ======================
    fps_value     = float(vision_cfg.get("fps", DEFAULT_FPS))
    max_frames    = int(vision_cfg.get("max_frames", DEFAULT_MAX_FRAMES))

    logger.info("=" * 80)
    logger.info("开始特征提取...")
    logger.info(f"  fps={fps_value}, max_frames={max_frames}")
    logger.info(f"  特征形状: [256, 4096]（use_llama_proj=True 时）")
    logger.info("=" * 80)

    start_time        = time.time()
    success_count     = 0
    fail_count        = 0
    total_infer_time  = 0.0
    checkpoint_progress = {"processed": list(processed_videos)}

    for idx, video_info in enumerate(remaining_videos):
        video_id   = video_info.get("id", str(idx))
        video_file = video_info.get("video_file", "unknown")
        label      = video_info.get("label", "unknown")

        logger.info(f"[{idx+1}/{len(remaining_videos)}] 处理: {video_file} (标签: {label})")

        # OOM 时降低 fps 重试（与 Qwen 版完全一致）
        attempt_fps = fps_value
        used_fps    = fps_value
        error_type  = "error"
        features    = None
        elapsed     = -1.0

        while True:
            gpu_mem_before = get_gpu_memory_usage(model_cfg["device"])
            features, elapsed, error_type = extract_single_video_features(
                video_info=video_info,
                encoder=encoder,
                vis_transform=vis_transform,
                device=model_cfg["device"],
                fps=attempt_fps,
                max_frames=max_frames,
            )
            gpu_mem_after = get_gpu_memory_usage(model_cfg["device"])
            used_fps = attempt_fps

            if features is not None:
                break

            if error_type == "oom":
                if attempt_fps <= MIN_FPS + 1e-6:
                    logger.error(f"  ❌ OOM 且 fps 已降至 {attempt_fps}，跳过该视频")
                    break
                next_fps = max(MIN_FPS, attempt_fps / 2.0)
                logger.warning(f"  ⚠ OOM：将 fps 从 {attempt_fps} 降至 {next_fps} 重试（仅该视频）")
                attempt_fps = next_fps
                continue

            break  # 非 OOM 错误，直接跳过

        if features is not None:
            success_count += 1
            total_infer_time += elapsed
            feature_shape = list(features.shape)

            # 即时保存（.pt 格式，与 Qwen 版 save_features_single 完全一致）
            output_name = _build_feature_filename(video_id, video_file, used_fps)
            output_path = os.path.join(feature_output_dir, output_name)
            save_features_single(features, output_path, metadata={
                "video_id":      video_id,
                "video_file":    video_file,
                "label":         label,
                "fps":           used_fps,
                "feature_shape": feature_shape,
                "encoder_type":  "emotion_llama_eva_clip",
            })

            logger.info(
                f"  ✅ 成功 | 形状: {feature_shape} | 耗时: {elapsed:.2f}s | "
                f"fps: {used_fps} | GPU: {gpu_mem_before:.2f}GB→{gpu_mem_after:.2f}GB | "
                f"累计成功: {success_count}/{idx+1}"
            )
        else:
            fail_count += 1
            logger.error(
                f"  ❌ 失败 | 原因: {error_type} | fps: {used_fps} | "
                f"耗时: {elapsed:.2f}s | 累计失败: {fail_count}"
            )

        # 断点续传：定期保存进度
        if (
            resume_cfg.get("enabled", True)
            and checkpoint_file
            and (success_count + fail_count) % resume_cfg.get("save_interval", 50) == 0
        ):
            checkpoint_progress["processed"] = list(
                processed_videos.union(
                    {v["id"] for v in remaining_videos[:idx+1] if v["id"] is not None}
                )
            )
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_progress, f, ensure_ascii=False, indent=2)

        # inode 定期检查（每 100 个视频）
        if (success_count + fail_count) % 100 == 0:
            used, total, pct = check_inode_usage()
            logger.info(f"  📊 inode: {used}/{total} ({pct:.1f}%), 成功 {success_count}, 失败 {fail_count}")

    # ====================== 最终统计 ======================
    total_elapsed = time.time() - start_time
    avg_time = total_infer_time / success_count if success_count > 0 else 0

    logger.info("=" * 80)
    logger.info("特征提取完成！最终统计：")
    logger.info(f"  总视频数:         {total_count}")
    logger.info(f"  成功:             {success_count}")
    logger.info(f"  失败:             {fail_count}")
    logger.info(f"  总运行时间:       {total_elapsed:.2f}s")
    logger.info(f"  平均推理时间:     {avg_time:.3f}s/视频")
    logger.info(f"  特征输出目录:     {feature_output_dir}")

    final_used, final_total, final_pct = check_inode_usage()
    logger.info(f"  最终 inode 使用:  {final_used}/{final_total} ({final_pct:.1f}%)")
    logger.info(f"  特征文件数量:     {count_files_in_directory(feature_output_dir)}")
    logger.info(f"  日志文件:         {log_file}")
    logger.info("=" * 80)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-LLaMA 视觉特征提取工具")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "feature_extraction_config_llama.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集路径（覆盖配置文件中的 data.dataset_path）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="特征输出根目录（覆盖配置文件中的 feature_saving.output_dir）",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="支持的视频后缀，逗号分隔（例如 .mp4,.avi,.mkv）",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="标注文件名（覆盖配置文件中的 data.annotation_file）",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="视频子目录名（覆盖配置文件中的 data.video_dir）",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_override=args.dataset,
        output_dir_override=args.output_dir,
        extensions_override=args.extensions,
        annotation_file_override=args.annotation_file,
        video_dir_override=args.video_dir,
    )
