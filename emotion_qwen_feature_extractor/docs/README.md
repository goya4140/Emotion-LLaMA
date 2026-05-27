# Emotion-Qwen 特征提取工具使用文档

## 概述

本工具集用于从 Emotion-Qwen 模型中提取视觉编码器（Visual Encoder）和混合压缩器（Hybrid Compressor, HC）的输出特征，实现"一次提取，多次使用"，避免后续推理时重复加载encoder到显存。

所有预处理和编码流程严格遵循 Emotion-Qwen 官方仓库的处理逻辑。

## 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+（推荐2.1+，支持flash_attention_2）
- CUDA 11.8+（GPU推理）
- HuggingFace Transformers 4.45+（支持自定义processor）

### Python依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate qwen-vl-utils
pip install numpy pyyaml Pillow
pip install flash-attn --no-build-isolation  # 优化注意力实现
```

### 模型权重
确保官方预训练权重位于 `/root/autodl-tmp/data/Emotion-Qwen-pretrained/` 目录。

## 快速开始

### 1. 检查inode使用量

```bash
python check_inode_usage.py --check /root/autodl-tmp
```

### 2. 运行特征提取

```bash
cd /root/autodl-tmp/emotion_qwen_feature_extractor
python extract_emotion_qwen_features.py --config feature_extraction_config.yaml
```

### 3. 加载特征（单文件模式）
```python
from feature_utils import load_features_single

data = load_features_single("extracted_features/CA-MER/single_files/sample_00012345_features.pt")
print(data["features"].shape)
```

## 配置文件说明

配置文件 `feature_extraction_config.yaml` 包含所有可配置参数，每个参数均注明来源（官方config.json中的哪个字段）。

### 模型配置 (model)
| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| model_path | `/root/autodl-tmp/data/Emotion-Qwen-pretrained` | 用户指定 | 预训练权重路径 |
| device | `cuda:0` | 用户选择 | 计算设备 |
| torch_dtype | `bfloat16` | 官方README推理示例 | 推理数据类型 |
| attn_implementation | `flash_attention_2` | 官方README | 注意力实现 |

### 视觉预处理参数 (vision_preprocess)
| 参数 | 默认值 | 来源 | 说明 |
|------|--------|------|------|
| min_pixels | 3136 | processor.__init__ | 最小像素数 |
| max_pixels | 1003520 | processor.__init__ (1280×28×28) | 最大像素数 |
| patch_size | 14 | processor.__init__ | 空间patch大小 |
| temporal_patch_size | 2 | processor.__init__ | 时间patch大小 |
| merge_size | 2 | processor.__init__ | HC合并因子 |
| default_fps | 2.0 | processor._defaults | 视频抽帧率 |
| image_mean | [0.48145466, 0.4578275, 0.40821073] | processor.__init__ | 归一化均值 |
| image_std | [0.26862954, 0.26130258, 0.27577711] | processor.__init__ | 归一化标准差 |

### 保存模式（单文件）

- 每个视频对应一个 `.pt` 文件
- 输出目录：`<output_dir>/<dataset_name>/single_files/`

## 常见问题

### Q1: CUDA OOM（显存溢出）
```yaml
# 解决方法：降低batch_size并设置aggressive_memory_cleanup
batch_processing:
  gpu_batch_size: 1
  aggressive_memory_cleanup: true
```

### Q2: inode不足
```bash
# 先检查inode情况
python check_inode_usage.py --check /root/autodl-tmp

# 清理__pycache__
python check_inode_usage.py --clean /root/autodl-tmp --execute

# 建议将输出目录放在inode充足的分区
feature_saving:
  output_dir: "/path/to/output"
```

### Q4: 断点续传
系统自动支持断点续传：
- 已处理视频记录在 `logs/checkpoint.json`
- 重启后自动跳过已处理视频
- 批量模式下也会从已有索引恢复

## 文件结构

```
emotion_qwen_feature_extractor/
├── docs/
│   ├── README.md                       # 使用文档（本文件）
│   └── technical_details.md            # 技术细节文档
├── logs/
│   └── feature_extraction_*.log        # 实时日志文件
├── extract_emotion_qwen_features.py    # 主特征提取脚本
├── feature_extraction_config.yaml      # 配置文件
├── feature_utils.py                    # 工具函数
├── check_inode_usage.py                # inode检查工具
├── load_single_file_features_example.py  # 单文件加载示例
└── verify_feature_consistency.py         # 一致性验证脚本

提取后的特征输出结构：
extracted_features/
└── CA-MER/                             # 数据集名称
  └── single_files/                   # 单文件特征
    ├── sample_00000001_fps2_features.pt
    ├── sample_00000002_fps2_features.pt
    └── ...
```

## 性能参考

基于 NVIDIA RTX 4090 (24GB) 测试：
- 单个视频处理时间：约 2-5 秒（取决于视频长度）
- 编码器显存占用：约 8-12 GB
- 特征输出大小：每个视频约 0.5-5 MB（取决于帧数）

## 许可

本工具集遵循 Apache-2.0 协议。
