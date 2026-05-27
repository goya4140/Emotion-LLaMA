# Emotion-LLaMA 特征提取任务

## 项目背景

基于 Emotion-LLaMA 框架，对三个情感识别数据集提取多模态特征，为后续模型微调训练做数据准备。

---

## 仓库结构预期

```
repo/
├── emotion_qwen_feature_extractor/   ← 师兄提供的 Qwen 版本脚本（参考模板）
├── extract_emotion_llama_features.py ← 待完善的 LLaMA 版本脚本
├── feature_extraction_config.yaml   ← 配置文件
├── Emotion-LLaMA/                   ← clone 的 Emotion-LLaMA 原始仓库
├── datasets/
│   ├── CA-MER/
│   ├── EMER/  (实际目录名 MER2025-ovmer)
│   └── 第三数据集/（待补充）
└── annotations/
    └── emotion_analysis_data.json   ← EMER 训练标注（332条，CoT格式）
```

---

## 核心任务

### 1. 对齐特征提取脚本

参考 `emotion_qwen_feature_extractor/` 中的逻辑，完善 `extract_emotion_llama_features.py`，使其能够提取 Emotion-LLaMA 所需的三路特征：

- **FaceMAE 特征** → 存入 `mae_340_UTT/{video_name}.npy`，shape `(1, 1024)`
- **VideoMAE 特征** → 存入 `maeV_399_UTT/{video_name}.npy`，shape `(1, 1024)`
- **HuBERT 音频特征** → 存入 `HL-UTT/{video_name}.npy`，shape `(1, 1024)`

抽帧和特征提取方式需与师兄示例代码保持一致。

### 2. 修改 Emotion-LLaMA 数据加载代码

修改 `Emotion-LLaMA/minigpt4/datasets/datasets/first_face.py` 中的路径配置，使其指向本地数据集和特征目录。

### 3. 修改训练配置文件

修改 `Emotion-LLaMA/train_configs/Emotion-LLaMA_finetune.yaml`：
- `llama_model` 路径指向本地 LLaMA-2 权重
- `ckpt` 路径指向本地 minigptv2_checkpoint.pth
- `output_dir` 指向本地输出目录

### 4. 数据标注格式对齐

`emotion_analysis_data.json` 为 CoT 对话格式（用于 Qwen VL 微调），需确认其是否需要转换为 Emotion-LLaMA 原版的标注格式（`MERR_coarse_grained.txt`）。

---

## 关键约束

- EMER 数据集视频路径必须与标注 JSON 中的硬编码路径一致：
  `/root/autodl-tmp/datasets/MER2025-ovmer/video/`
- 三路特征 shape 必须为 `(1, 1024)`，与 Emotion-LLaMA 线性投影层输入维度对齐
- 如果 Emotion-LLaMA 无法运行，回退方案为使用 Qwen VL 7B/8B

---

## 待确认项（需联系师兄）

- [ ] CA-MER 数据集的标注 JSON 在哪里
- [ ] 第三个数据集（吴思雨处）
- [ ] 抽帧方式的具体参数（师兄承诺发示例代码）
