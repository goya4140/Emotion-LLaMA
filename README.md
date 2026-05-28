# Emotion-LLaMA 特征提取工具集

## 项目背景

基于 Emotion-LLaMA 框架，对情感识别数据集（CA-MER、EMER）提取多模态视觉特征，为后续模型微调训练做数据准备。

---

## 仓库结构

```
Emotion-LLaMA/
├── feature_extractors/                  ← 三路视觉编码器特征提取（主要工作目录）
│   ├── shared/
│   │   └── feature_utils.py             ← 共享工具函数（日志、保存、视频列表）
│   ├── eva_clip/
│   │   ├── extract_eva_clip_features.py ← EVA-CLIP-G 提取器，输出 [1024, 1408]
│   │   ├── config_eva_clip.yaml         ← 配置文件（含 CA-MER + EMER 双数据集）
│   │   └── vendor/
│   │       ├── eva_vit.py               ← EVA ViT 架构（无需安装 minigpt4 包）
│   │       └── dist_utils.py            ← 权重下载工具
│   ├── mae/
│   │   ├── extract_mae_features.py      ← MAE-Large 提取器，输出 [1024]
│   │   └── config_mae.yaml
│   └── videomae/
│       ├── extract_videomae_features.py ← VideoMAE-Large 提取器，输出 [1024]
│       └── config_videomae.yaml
│
├── emotion_qwen_feature_extractor/      ← Qwen VL 版提取器（保持不变）
├── extract_emotion_llama_features.py    ← 旧版 LLaMA 提取器（含 llama_proj，历史存档）
├── emotion_analysis_data.json           ← EMER 数据集标注（332 条，CoT 对话格式）
├── 特征提取脚本说明.md
└── 操作指导.md
```

---

## 三路编码器概览

| 编码器 | 脚本 | 输出形状 | 对应 Emotion-LLaMA 层 | 权重来源 |
|--------|------|---------|----------------------|---------|
| EVA-CLIP-G | `eva_clip/extract_eva_clip_features.py` | `[1024, 1408]` | `llama_proj` 输入边界 | `eva_vit_g.pth`（自动下载，~3.6GB） |
| MAE-Large | `mae/extract_mae_features.py` | `[1024]` | `feats_llama_proj1` 输入边界 | `facebook/vit-mae-large`（HuggingFace） |
| VideoMAE-Large | `videomae/extract_videomae_features.py` | `[1024]` | `feats_llama_proj2` 输入边界 | `MCG-NJU/videomae-large`（HuggingFace） |

三个编码器均为**独立使用**，不加载 Emotion-LLaMA 的投影层权重（`llama_proj`、`feats_llama_proj`、`cls_tk_llama_proj`）。输出向量格式与投影层输入边界对齐，可直接传入对应线性层。

---

## 快速开始

```bash
# 安装依赖
pip install timm omegaconf opencv-python-headless Pillow pyyaml transformers -q

# EVA-CLIP 特征提取（CA-MER + EMER）
python feature_extractors/eva_clip/extract_eva_clip_features.py \
    --config feature_extractors/eva_clip/config_eva_clip.yaml

# MAE 特征提取
python feature_extractors/mae/extract_mae_features.py \
    --config feature_extractors/mae/config_mae.yaml

# VideoMAE 特征提取
python feature_extractors/videomae/extract_videomae_features.py \
    --config feature_extractors/videomae/config_videomae.yaml
```

输出路径：`/root/autodl-fs/features/{EVA-CLIP,MAE,VideoMAE}/{CA-MER,EMER}/{video_id}_features.pt`

---

## 输出文件格式

每个视频对应一个 `.pt` 文件（Python dict）：

```python
{
    "features":      torch.Tensor,   # 形状见上表，dtype=float32（CPU）
    "feature_shape": list,
    "metadata": {
        "video_id":      str,
        "video_file":    str,
        "label":         str,        # 情感标签（从标注 JSON 解析）
        "encoder_type":  str,        # "EVA-CLIP-G" / "MAE-Large" / "VideoMAE-Large"
        ...
    },
    "timestamp": str,
}
```

加载示例：
```python
import torch
data = torch.load("sample_00000001_features.pt", map_location="cpu", weights_only=False)
features = data["features"]   # e.g. torch.Tensor [1024, 1408]
label    = data["metadata"]["label"]
```

---

## 数据集说明

| 数据集 | 标注文件 | 视频目录 | 格式 |
|-------|---------|---------|------|
| CA-MER | `video-aligned.json` | `video-aligned/` | 列表 JSON，`video` 字段为相对路径 |
| EMER | `emotion_analysis_data.json`（仓库根目录） | 绝对路径（`/root/autodl-tmp/.../video/`） | CoT 对话格式，标签在 `<answer>` 标签内 |

---

## 断点续传

所有提取器均支持断点续传。中断后直接重新运行即可，已完成的视频会自动跳过（基于 checkpoint JSON + 输出文件存在检查双重保险）。

```bash
# 只处理单个数据集（--dataset 参数）
python feature_extractors/eva_clip/extract_eva_clip_features.py \
    --config feature_extractors/eva_clip/config_eva_clip.yaml \
    --dataset CA-MER

# 试运行（不实际提取，只打印路径）
python feature_extractors/eva_clip/extract_eva_clip_features.py \
    --config feature_extractors/eva_clip/config_eva_clip.yaml \
    --dry-run
```

---

## 详细说明

- 脚本原理与架构：[`特征提取脚本说明.md`](./特征提取脚本说明.md)
- AutoDL 完整操作步骤：[`操作指导.md`](./操作指导.md)
