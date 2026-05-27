"# Emotion‑Qwen Encoder 内部架构剖析与修复总结

## 动机
本项目从 Emotion‑Qwen 只抽取**视觉编码器**（vpm + 压缩器 + 门控网络）的输出特征，用于下游 MER 任务。  
经过多轮探测与修复，发现了模型内部多条与常规 ViT‑LLM 差异极大的设计细节，本文档记录完整的技术细节与所有修复点。

---

## 1. 内部架构

### 1.1 顶层模块映射
```python
for name, child in full_model.named_children():
    print(f\"{name}: {type(child).__name__}\")
```
输出：
```
vpm: Emotion_QwenVisionTransformerPretrainedModel    # 视觉编码器
llm: Emotion_QwenModel                                # ❌ 不加载
generalcompressor: MoECompressor                      # 通用情感压缩机
emotioncompressor: MoECompressor                      # 特定情感压缩机
gatenet: GateNetWithAttention                         # 动态门控网络
lm_head: Linear                                       # ❌ 不加载
```

### 1.2 完整数据流（经 hook 验证）

| 步骤 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 预处理 | `process_vision_info` | video file | `pixel_values [5376, 1176]` + `grid_thw [4, 28, 48]` |
| 编码 | `vpm` | `pixel_values` + `grid_thw` (kwarg) | `(vision_features [5376, 1280], aux [1344])` |
| 压缩 | `generalcompressor` | `vision_features [5376, 1280]` | `gen_feat [1344, 3584]` |
| 压缩 | `emotioncompressor` | `vision_features [5376, 1280]` | `emo_feat [1344, 3584]` |
| 池化 | 手动 mean | `vision_features [5376, 1280]` | `pooled [1, 1280]` |
| 门控 | `gatenet` | `pooled [1, 1280]` | `gate [1, 2]` |
| 融合 | 手动加权 | `gen_feat * gate[0] + emo_feat * gate[1]` | `features [1344, 3584]` |

### 1.3 双路 MoE + 门控融合（核心创新）

与常规 Vision‑LLM 的单路投影不同，Emotion‑Qwen 的视觉特征经过 **两条独立的 MoE 压缩路径**：

- `generalcompressor`：通用情感压缩 → `gen_feat [1344, 3584]`
- `emotioncompressor`：特定情感压缩 → `emo_feat [1344, 3584]`

然后由 `gatenet` 根据全局视觉内容（`vision_features` 池化）动态生成 2 维权重向量：

```
fused = gen_feat * gate[0] + emo_feat * gate[1]
```

**这意味着单独保存 gen_feat 或 emo_feat 都会丢失一半信息，必须完整保留门控融合后的特征。**

---

## 2. 关键设计细节（与常规假设的差异）

### 2.1 子模块命名完全不常规

| 常规假设 | 实际名称 |
|----------|----------|
| `model.visual` | `model.vpm` |
| `model.vision_model` | 不存在 |
| `model.merger` | `model.generalcompressor` + `model.emotioncompressor` |
| `model.projector` | 不存在 |

### 2.2 `vpm` 必须显式传入 `grid_thw`

Hook 捕获 vpm 的 positional args 只有 `pixel_values`，但实际 `forward()` 需要 `grid_thw` 作为关键字参数：

```python
vpm_output = self.vpm(pixel_values, grid_thw=grid_thw)  # ✅ 正确
vpm_output = self.vpm(pixel_values)                      # ❌ TypeError
```

### 2.3 `gatenet` 输入是 `vision_features` 池化，不是 `aux`

探测数据显示 gatenet 输入为 `[1, 1280]`，与 vision_features 的 hidden dim 一致。而 `aux` 为 `[1344]` 且 dtype 为 int64，不可能是 gate 输入。正确做法：

```python
pooled = vision_features.mean(dim=0, keepdim=True)  # [1, 1280]
gate = self.gatenet(pooled)                          # [1, 2]
```

---

## 3. 所有修复点

| 编号 | 文件 | 问题 | 严重度 | 修复内容 |
|------|------|------|--------|----------|
| 1 | `feature_utils.py` | `DEFAULT_MAX_PIXELS = 1280*28*28` 超出 `qwen_vl_utils` 上限 | 🔴 致命 | 改为 `768*28*28 = 602112` |
| 2 | `feature_extraction_config.yaml` | `max_pixels: 1003520` | 🔴 致命 | 改为 `602112` |
| 3 | `extract_emotion_qwen_features.py` | `EmotionQwenEncoder` 子模块属性名全错（猜测 `visual`/`merger` 等） | 🔴 致命 | 精准映射为 `vpm`, `generalcompressor`, `emotioncompressor`, `gatenet` |
| 4 | 同上 | `forward` 签名缺少 `grid_thw`，`vpm` 调用报 `TypeError` | 🔴 致命 | `forward` 增加 `grid_thw` 参数，并在 `extract_single_video_features` 中传递 |
| 5 | 同上 | `gatenet` 输入错误使用 `aux` (int64)，维度不匹配 | 🔴 致命 | 改为 `vision_features.mean(dim=0, keepdim=True)` |
| 6 | 同上 | `preprocess_video_to_tensor` 手动抽帧+人脸检测，与 benchmark 推理流程不一致 | 🔴 致命 | 改用 `process_vision_info` + `processor(text, images, videos)` 官方流程 |
| 7 | 同上 | 缺少 `import logging` | 🔴 致命 | 添加到文件头部 |
| 8 | `feature_extraction_config.yaml` | `attn_implementation: flash_attention_2`（环境未安装） | 🟡 高 | 改为 `sdpa` |
| 9 | 同上 | `output_dir` 指向不可用分区 | 🟡 高 | 改为 `/root/autodl-fs/data/CA-MER-EmoQwenEncoder` |
| 10 | `extract_emotion_qwen_features.py` | `extract_single_video_features` 参数签名过时 | 🟡 高 | 简化为 `(video_info, encoder, device, fps, max_pixels)` |

---

## 4. 验证方法

```bash
conda run -n emotion_qwen python -c "
import sys, json, torch
sys.path.insert(0, '/root/autodl-tmp/emotion_qwen_feature_extractor')
from extract_emotion_qwen_features import EmotionQwenEncoder, preprocess_video_to_tensor

encoder = EmotionQwenEncoder('/root/autodl-tmp/data/Emotion-Qwen-pretrained',
                              device='cuda:0', torch_dtype=torch.bfloat16)
with open('/root/autodl-fs/benchmark/CA-MER/video-aligned.json') as f:
    data = json.load(f)
entry = data[0]
video_path = f'/root/autodl-fs/benchmark/CA-MER/video-aligned/{entry[\"file_name\"]}.avi'

pv, grid_thw = preprocess_video_to_tensor(video_path, encoder.processor)
pv = pv.to('cuda:0', dtype=torch.bfloat16)
grid_thw = grid_thw.to('cuda:0')
feat = encoder(pv, grid_thw)

print(f'✅ Feature shape: {feat.shape}')  # 预期 [1344, 3584]
"
```

---

## 5. 关键教训

1. **绝不凭经验猜测子模块名** — 必须通过 `named_children()` + `register_forward_hook` 探测实际属性和签名
2. **视觉编码器需要 `grid_thw`** — 即使 hook 只捕获到 positional args，实际 `forward()` 以关键字参数方式接收
3. **双路 MoE + 门控融合** 是 Emotion‑Qwen 的核心创新，抽取特征时必须完整保留整条通路
4. **预处理流程必须与目标场景严格一致** — 任何偏差都会导致特征分布偏移和模型失效
5. **`max_pixels` 受 `qwen_vl_utils` 隐式上限约束** — 超过 `768*28*28` 的值会被静默裁剪，导致视觉信息丢失
"