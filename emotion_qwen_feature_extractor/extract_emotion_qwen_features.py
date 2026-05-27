# -*- coding: utf-8 -*-
"""
文件名：extract_emotion_qwen_features.py
功能：Emotion-Qwen 特征提取主脚本
创建时间：2025-07-11

调用方式:
  1) 使用配置文件默认参数:
      python extract_emotion_qwen_features.py --config feature_extraction_config.yaml
  2) 指定不同数据集与输出目录:
      python extract_emotion_qwen_features.py --config feature_extraction_config.yaml \
         --dataset /path/to/dataset --output-dir /path/to/output
  3) 覆盖支持的视频格式(逗号分隔):
      python extract_emotion_qwen_features.py --config feature_extraction_config.yaml \
         --extensions .mp4,.avi,.mkv
  4) 指定标注文件/视频子目录:
      python extract_emotion_qwen_features.py --config feature_extraction_config.yaml \
         --annotation-file video-aligned.json --video-dir video-aligned

本脚本实现以下功能：
1. 只加载Emotion-Qwen的视觉encoder和混合压缩器(HC)部分，不加载LLM主干
2. 使用process_vision_info进行视频预处理，与官方推理流程一致
3. 每个样本保存为单独的.pt特征文件
4. 实时日志记录，每处理完一个视频立即写入日志
5. 支持断点续传和inode监控

所有预处理和编码流程严格遵循官方processing_emotionqwen_vl.py和README.md的描述。
参考源：
- 官方README.md 推理示例代码
- processing_emotionqwen_vl.py 完整预处理流程
- modeling_emotionqwen_vl.py 模型架构
- configuration_emotionqwen_vl.py 视觉编码器配置
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

import yaml
import torch
import torch.nn as nn
try:
    from torch.cuda import OutOfMemoryError
except ImportError:
    OutOfMemoryError = torch.cuda.OutOfMemoryError
from transformers import AutoProcessor, AutoModel

# 导入工具函数（禁止使用相对路径导入，确保脚本可独立运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import (
    setup_logger,
    check_inode_usage,
    get_available_inodes,
    get_gpu_memory_usage,
    get_video_list,
    save_features_single,
    count_files_in_directory,
    DEFAULT_FPS,
    DEFAULT_MAX_PIXELS,
)

# ============================================================================
# 模型加载函数（仅加载encoder部分）
# ============================================================================

class EmotionQwenEncoder(nn.Module):
    """
    Emotion-Qwen的视觉Encoder + 双路混合压缩器(HC)封装。
    只加载视觉相关部分：vpm + generalcompressor + emotioncompressor + gatenet，
    不加载LLM主干(Emotion_QwenModel)和lm_head，以节省约85%参数/显存。

    数据流（经probe_vision_encoder.py验证）：
      pixel_values -> vpm -> (vision_feat[5376,1280], aux[1344])
        vision_feat -> generalcompressor -> gen_feat[1344,3584]
        vision_feat -> emotioncompressor -> emo_feat[1344,3584]
        pooled(aux) -> gatenet -> gate[1,2]
        output = gen_feat * gate[0] + emo_feat * gate[1]

    参考源：
    - probe_vision_encoder.py 实测钩子数据
    - modeling_emotionqwen_vl.py 模型架构
    """

    def __init__(self, model_path: str, device: str = "cuda:0",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "sdpa"):
        """
        初始化并只加载视觉模块。

        Args:
            model_path: 模型路径
            device: 计算设备
            torch_dtype: 数据类型（使用bfloat16以节省显存）
            attn_implementation: 注意力实现方式（默认sdpa，flash_attn未安装）
        """
        super().__init__()
        self.device = device
        self.model_path = model_path

        logger = logging.getLogger("feature_extraction")
        logger.info(f"正在加载Emotion-Qwen模型（仅encoder部分），路径：{model_path}")

        # 加载完整模型（仅在获取子模块后立即释放LLM部分）
        logger.info("步骤1: 加载完整模型权重...")

        full_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device,
            trust_remote_code=True,
        )

        # 记录顶层子模块
        logger.info("步骤2: 模型顶层子模块:")
        for name, mod in full_model.named_children():
            logger.info(f"  {name}: {type(mod).__name__}")

        # 提取视觉模块（probe已验证的属性名）
        # vpm: Vision Perception Module (Emotion_QwenVisionTransformerPretrainedModel)
        self.vpm = None
        if hasattr(full_model, "vpm"):
            self.vpm = full_model.vpm
            logger.info(f"  ✅ 提取 vpm: {type(self.vpm).__name__}")

        # generalcompressor / emotioncompressor: 双路MoE压缩器
        self.general_compressor = None
        self.emotion_compressor = None
        if hasattr(full_model, "generalcompressor"):
            self.general_compressor = full_model.generalcompressor
            logger.info(f"  ✅ 提取 generalcompressor: {type(self.general_compressor).__name__}")
        if hasattr(full_model, "emotioncompressor"):
            self.emotion_compressor = full_model.emotioncompressor
            logger.info(f"  ✅ 提取 emotioncompressor: {type(self.emotion_compressor).__name__}")

        # gatenet: 门控网络（融合general和emotion两路特征）
        self.gatenet = None
        if hasattr(full_model, "gatenet"):
            self.gatenet = full_model.gatenet
            logger.info(f"  ✅ 提取 gatenet: {type(self.gatenet).__name__}")

        # 将模块移动到目标设备并设为eval模式
        for name in ["vpm", "general_compressor", "emotion_compressor", "gatenet"]:
            mod = getattr(self, name, None)
            if mod is not None:
                setattr(self, name, mod.to(device=device, dtype=torch_dtype).eval())

        # 释放LLM主干以节省显存 (llm + lm_head)
        logger.info("步骤3: 释放LLM主干以节省显存...")
        if hasattr(full_model, "llm"):
            del full_model.llm
        if hasattr(full_model, "lm_head"):
            del full_model.lm_head
        del full_model
        gc.collect()
        torch.cuda.empty_cache()

        # 加载processor（用于视频预处理）
        logger.info("步骤4: 加载processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        vpm_params = self._count_params(self.vpm) if self.vpm else 0
        logger.info(f"✅ Encoder加载完成。vpm参数: {vpm_params:,}")

    @staticmethod
    def _count_params(module) -> int:
        """统计模块参数量"""
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Encoder前向传播（经过vpm + 双路compressor + gatenet融合）。
        数据流严格遵循probe_vision_encoder.py实测结果：
          vpm(pixel_values, grid_thw=grid_thw) -> (vision_feat, aux)
          gen_feat = generalcompressor(vision_feat)
          emo_feat = emotioncompressor(vision_feat)
          gate = gatenet(pooled(vision_feat))
          output = gen_feat * gate[0] + emo_feat * gate[1]

        注意：vpm 需要 grid_thw 作为关键字参数（在generate内部如此调用）。
        gatenet 输入是 vision_features 的全局池化（[1, 1280]），而非 aux。

        Args:
            pixel_values: 预处理后的像素值 [num_patches, num_channels]
            grid_thw: 网格信息 [grid_t, grid_h, grid_w]

        Returns:
            features: 编码后的融合特征 [seq_len, hidden_dim]
        """
        with torch.no_grad():
            # 步骤1: vpm 视觉编码
            if self.vpm is None:
                raise RuntimeError("视觉编码器(vpm)未加载")
            vpm_output = self.vpm(pixel_values, grid_thw=grid_thw)
            # vpm 返回 tuple (vision_features [N,1280], auxiliary [M])
            if isinstance(vpm_output, tuple):
                vision_features, aux = vpm_output
            else:
                vision_features = vpm_output
                aux = None

            # 步骤2: 双路压缩器
            gen_feat = self.general_compressor(vision_features) if self.general_compressor else vision_features
            emo_feat = self.emotion_compressor(vision_features) if self.emotion_compressor else vision_features

            # 步骤3: gatenet 融合（输入是vision_features的全局池化，probe显示为[1,1280]）
            if self.gatenet is not None:
                # 对vision_features取全局平均池化
                pooled = vision_features.mean(dim=0, keepdim=True).to(dtype=gen_feat.dtype)  # [1, 1280]
                gate = self.gatenet(pooled)  # [1, 2]
                gate = gate.squeeze(0)  # [2]
                features = gen_feat * gate[0] + emo_feat * gate[1]
            else:
                # 无gatenet时，直接拼接两路特征
                features = torch.cat([gen_feat, emo_feat], dim=-1)

        return features


# ============================================================================
# 视频预处理与特征提取流程
# ============================================================================

def preprocess_video_to_tensor(
    video_path: str,
    processor,
    fps: float = DEFAULT_FPS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单个视频进行官方标准预处理流程，与run_benchmark.py中验证过的流程一致：
    1. 构造 messages 格式（与官方推理完全一致）
    2. process_vision_info 处理视频
    3. processor(text=..., images=..., videos=...) 标准化

    注意：此次修复移除了原有的手动抽帧+人脸检测流程，
    改用qwen_vl_utils的process_vision_info统一处理，
    保持与benchmark推理完全一致的输入分布。
    max_pixels使用768*28*28=602112，避免超出qwen-vl-utils上限。

    参考源：官方README推理示例 + run_benchmark.py（已验证）

    Returns:
        (pixel_values, grid_thw): 预处理后的张量
    """
    logger = logging.getLogger("feature_extraction")

    from qwen_vl_utils import process_vision_info

    # 步骤1：构造与官方推理一致的 messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}",
             "max_pixels": max_pixels, "fps": fps},
            {"type": "text", "text": "Analyze the emotion."}  # 占位文本，仅用于触发视频编码
        ]
    }]

    # 步骤2：使用 process_vision_info 预处理视频
    image_inputs, video_inputs = process_vision_info(messages)

    # 步骤3：通过 processor 获取 pixel_values 和 grid_thw
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # processor 会返回 'pixel_values_videos' 或 'pixel_values'
    if "pixel_values_videos" in inputs:
        pixel_values = inputs["pixel_values_videos"]
    elif "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]
    else:
        raise KeyError(f"无法从processor输出中找到 pixel_values，可用键: {list(inputs.keys())}")

    if "video_grid_thw" in inputs:
        grid_thw = inputs["video_grid_thw"]
    elif "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"]
    else:
        raise KeyError(f"无法从processor输出中找到 grid_thw，可用键: {list(inputs.keys())}")

    logger.debug(f"  pixel_values shape: {pixel_values.shape}, grid_thw: {grid_thw}")
    return pixel_values, grid_thw


def _is_oom_error(err: Exception) -> bool:
    if isinstance(err, OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _format_fps_label(fps: float) -> str:
    fps_str = f"{fps:.2f}".rstrip("0").rstrip(".")
    return fps_str if fps_str else str(fps)


def _build_feature_filename(video_id: str, video_file: str, fps: float) -> str:
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
    encoder: EmotionQwenEncoder,
    device: str,
    fps: float,
    max_pixels: int,
) -> Tuple[Optional[torch.Tensor], float, str]:
    """
    对单个视频进行特征提取。

    流程：
    1. 视频预处理（process_vision_info + processor，与benchmark一致的官方流程）
    2. Encoder前向传播（vpm -> compressors -> gatenet）
    3. 提取特征到CPU

    Args:
        video_info: 视频信息字典（包含video_path等键）
        encoder: EmotionQwenEncoder实例
        device: 计算设备
        fps: 抽帧帧率
        max_pixels: 最大像素数

    Returns:
        (features, elapsed_time, error_type): 特征张量 [seq_len, hidden_dim] 和处理时间（秒），
        失败返回(None, elapsed_time, "oom"|"error")
    """
    logger = logging.getLogger("feature_extraction")
    video_path = video_info["video_path"]
    video_id = video_info.get("id", "unknown")

    start_time = time.time()

    pixel_values = grid_thw = features = features_cpu = None
    try:
        # 步骤1: 视频预处理（process_vision_info流程）
        pixel_values, grid_thw = preprocess_video_to_tensor(
            video_path=video_path,
            processor=encoder.processor,
            fps=fps,
            max_pixels=max_pixels,
        )

        # 步骤2: 移动到GPU
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        grid_thw = grid_thw.to(device=device)

        # 步骤3: Encoder前向传播（vpm需要grid_thw作为kwarg）
        features = encoder(pixel_values, grid_thw)

        # 步骤4: 移至CPU
        features_cpu = features.cpu()

        elapsed = time.time() - start_time
        return features_cpu, elapsed, "ok"

    except Exception as e:
        elapsed = time.time() - start_time
        if _is_oom_error(e):
            logger.warning(f"  视频 {video_id} 在fps={fps}时显存不足: {type(e).__name__}: {e}")
            logger.debug(f"  错误堆栈: {traceback.format_exc()}")
            return None, elapsed, "oom"
        logger.error(f"  视频 {video_id} 处理失败: {type(e).__name__}: {e}")
        logger.debug(f"  错误堆栈: {traceback.format_exc()}")
        return None, elapsed, "error"

    finally:
        # 无论成功或失败，强制清理所有GPU张量，防止碎片积累导致OOM
        for t in [pixel_values, grid_thw, features]:
            if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
                del t
        del pixel_values, grid_thw, features, features_cpu
        gc.collect()
        torch.cuda.empty_cache()
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
    特征提取主函数。

    流程：
    1. 加载配置文件
    2. 检查inode状态
    3. 初始化日志
    4. 加载模型（仅encoder）
    5. 获取视频列表
    6. 逐个处理视频并提取特征
    7. 保存特征（单文件或批量打包模式）
    8. 输出统计信息

    Args:
        config_path: YAML配置文件路径
        dataset_override: 覆盖数据集路径(可选)
        output_dir_override: 覆盖特征输出根目录(可选)
        extensions_override: 覆盖支持的视频后缀(逗号分隔字符串, 可选)
        annotation_file_override: 覆盖标注文件名(可选)
        video_dir_override: 覆盖视频子目录名(可选)
    """
    # ====================== 加载配置 ======================
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 提取所有配置参数
    model_cfg = config["model"]
    vision_cfg = config["vision_preprocess"]
    data_cfg = config["data"]
    saving_cfg = config["feature_saving"]
    resume_cfg = config["resume"]
    log_cfg = config["logging"]

    # ====================== 初始化日志 ======================
    logger, log_file = setup_logger(
        log_dir=log_cfg["output_dir"],
        name="feature_extraction",
        level=log_cfg["level"]
    )
    logger.info("=" * 80)
    logger.info("Emotion-Qwen 特征提取工具 启动")
    logger.info("=" * 80)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"日志文件: {log_file}")

    # ====================== inode检查 ======================
    used_inodes, total_inodes, usage_pct = check_inode_usage()
    available_inodes = get_available_inodes()
    logger.info(f"inode状态: 已用{used_inodes}/{total_inodes} ({usage_pct:.1f}%), 剩余{available_inodes}")
    if available_inodes > 0 and available_inodes < 100000:
        logger.warning(f"⚠ 剩余inode不足100000个({available_inodes})，建议先清理不必要的文件!")

    # ====================== 覆盖参数（命令行优先） ======================
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
        normalized_exts = []
        for ext in raw_exts:
            if not ext.startswith("."):
                ext = f".{ext}"
            normalized_exts.append(ext.lower())
        data_cfg["supported_extensions"] = normalized_exts
        logger.info(f"支持的视频后缀覆盖: {normalized_exts}")

    # ====================== 加载模型（仅encoder） ======================
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg["torch_dtype"], torch.bfloat16)

    logger.info("正在加载Emotion-Qwen Encoder...")
    logger.info(f"  模型路径: {model_cfg['model_path']}")
    logger.info(f"  设备: {model_cfg['device']}")
    logger.info(f"  数据类型: {model_cfg['torch_dtype']}")
    logger.info(f"  注意力实现: {model_cfg['attn_implementation']}")

    try:
        encoder = EmotionQwenEncoder(
            model_path=model_cfg["model_path"],
            device=model_cfg["device"],
            torch_dtype=torch_dtype,
            attn_implementation=model_cfg["attn_implementation"],
        )
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return

    # ====================== 获取视频列表 ======================
    logger.info(f"正在扫描视频列表... 数据集路径: {data_cfg['dataset_path']}")
    video_list = get_video_list(
        dataset_path=data_cfg["dataset_path"],
        annotation_file=data_cfg.get("annotation_file"),
        video_dir=data_cfg.get("video_dir", "video-aligned"),
        extensions=data_cfg.get("supported_extensions"),
    )
    total_count = len(video_list)
    logger.info(f"✅ 找到 {total_count} 个视频文件")

    if total_count == 0:
        logger.error("未找到任何视频文件，退出")
        return

    # ====================== 确定保存模式 ======================
    expected_files = total_count
    logger.info(f"保存模式: single, 预计产生{expected_files}个特征文件")

    if expected_files > 10000:
        logger.warning(f"⚠ 预计文件数({expected_files})超过10000，请确认inode容量")

    # ====================== 准备输出目录 ======================
    # 数据集名称（从路径提取）
    dataset_name = os.path.basename(data_cfg["dataset_path"].rstrip("/\\"))
    output_dir = os.path.join(saving_cfg["output_dir"], dataset_name)

    feature_output_dir = os.path.join(output_dir, "single_files")

    os.makedirs(feature_output_dir, exist_ok=True)

    # ====================== 断点续传 ======================
    checkpoint_file = resume_cfg.get("checkpoint_file", "")
    processed_videos = set()
    if resume_cfg.get("enabled", True) and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            processed_videos = set(checkpoint_data.get("processed", []))
        logger.info(f"断点续传：已处理 {len(processed_videos)} 个视频，将跳过它们")

    # 过滤已处理的视频
    remaining_videos = [v for v in video_list if v["id"] not in processed_videos and v["video_file"] not in processed_videos]
    logger.info(f"待处理: {len(remaining_videos)}/{total_count} 个视频")

    # ====================== 开始特征提取 ======================
    logger.info("=" * 80)
    logger.info("开始特征提取...")
    logger.info(f"  配置: fps={vision_cfg['fps']}, max_pixels={vision_cfg['max_pixels']}")
    logger.info(f"  预处理: process_vision_info (与benchmark推理一致)")
    logger.info("  保存模式: single")
    logger.info("=" * 80)

    start_time = time.time()
    success_count = 0
    fail_count = 0
    total_inference_time = 0.0
    # 进度记录文件
    checkpoint_progress = {"processed": list(processed_videos)}

    min_fps = 0.5

    for idx, video_info in enumerate(remaining_videos):
        video_id = video_info.get("id", str(idx))
        video_file = video_info.get("video_file", "unknown")
        label = video_info.get("label", "unknown")

        # 实时日志：每处理一个视频立即记录
        logger.info(f"[{idx+1}/{len(remaining_videos)}] 处理: {video_file} (标签: {label})")

        # 提取特征（OOM时仅对当前视频降fps重试）
        attempt_fps = float(vision_cfg["fps"])
        used_fps = attempt_fps
        error_type = "error"
        features = None
        elapsed = -1.0
        while True:
            gpu_mem_before = get_gpu_memory_usage(model_cfg["device"])
            features, elapsed, error_type = extract_single_video_features(
                video_info=video_info,
                encoder=encoder,
                device=model_cfg["device"],
                fps=attempt_fps,
                max_pixels=vision_cfg["max_pixels"],
            )
            gpu_mem_after = get_gpu_memory_usage(model_cfg["device"])
            used_fps = attempt_fps

            if features is not None:
                break

            if error_type == "oom":
                if attempt_fps <= min_fps + 1e-6:
                    logger.error(f"  ❌ OOM且fps已降到{attempt_fps}，跳过该视频")
                    break
                next_fps = max(min_fps, attempt_fps / 2.0)
                logger.warning(
                    f"  ⚠ OOM: 将fps从{attempt_fps}降到{next_fps}重试（仅该视频）"
                )
                attempt_fps = next_fps
                continue

            break

        if features is not None:
            success_count += 1
            total_inference_time += elapsed
            feature_shape = list(features.shape)

            # 即时保存（根据模式）
            output_name = _build_feature_filename(video_id, video_file, used_fps)
            output_path = os.path.join(feature_output_dir, output_name)
            save_features_single(features, output_path, metadata={
                "video_id": video_id,
                "video_file": video_file,
                "label": label,
                "fps": used_fps,
                "feature_shape": feature_shape,
            })

            # 日志记录（实时写入）
            logger.info(f"  ✅ 成功 | 形状: {feature_shape} | 耗时: {elapsed:.2f}s | "
                        f"fps: {used_fps} | GPU显存: {gpu_mem_before:.2f}GB -> {gpu_mem_after:.2f}GB | "
                        f"累计成功: {success_count}/{idx+1}")
        else:
            fail_count += 1
            logger.error(
                f"  ❌ 失败 | 原因: {error_type} | fps: {used_fps} | 耗时: {elapsed:.2f}s | 累计失败: {fail_count}"
            )

        # 断点续传：定期保存进度
        if resume_cfg.get("enabled", True) and (success_count + fail_count) % resume_cfg.get("save_interval", 50) == 0:
            checkpoint_progress["processed"] = list(processed_videos.union(
                set(v["id"] for v in remaining_videos[:idx+1] if v["id"] is not None)
            ))
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_progress, f, ensure_ascii=False, indent=2)

        # inode定期检查（每100个视频）
        if (success_count + fail_count) % 100 == 0:
            used, total, pct = check_inode_usage()
            logger.info(f"  📊 inode状态: {used}/{total} ({pct:.1f}%), 成功{success_count}, 失败{fail_count}")

    # ====================== 输出最终统计 ======================
    total_elapsed = time.time() - start_time
    avg_time = total_inference_time / success_count if success_count > 0 else 0

    logger.info("=" * 80)
    logger.info("特征提取完成！最终统计：")
    logger.info(f"  总视频数: {total_count}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  总运行时间: {total_elapsed:.2f}s")
    logger.info(f"  平均推理时间: {avg_time:.3f}s/视频")
    logger.info(f"  特征输出目录: {feature_output_dir}")

    final_used, final_total, final_pct = check_inode_usage()
    logger.info(f"  最终inode使用: {final_used}/{final_total} ({final_pct:.1f}%)")
    logger.info(f"  特征文件数量: {count_files_in_directory(feature_output_dir)}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 80)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-Qwen 特征提取工具")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "feature_extraction_config.yaml"),
        help="YAML配置文件路径"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集路径（覆盖配置文件中的data.dataset_path）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="特征输出根目录（覆盖配置文件中的feature_saving.output_dir）"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="支持的视频后缀，逗号分隔（例如 .mp4,.avi,.mkv）"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="标注文件名（覆盖配置文件中的data.annotation_file）"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="视频子目录名（覆盖配置文件中的data.video_dir）"
    )
    args = parser.parse_args()

    main(
        args.config,
        dataset_override=args.dataset,
        output_dir_override=args.output_dir,
        extensions_override=args.extensions,
        annotation_file_override=args.annotation_file,
        video_dir_override=args.video_dir,
    )



