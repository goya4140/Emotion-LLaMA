# Emotion-Qwen 特征提取技术细节

## 1. 预处理与编码流程

- 使用 `qwen_vl_utils.process_vision_info` 处理视频输入，保持与官方推理流程一致。
- 通过 `processor(text, images, videos)` 获取 `pixel_values_videos` 与 `video_grid_thw`。
- 仅加载视觉 encoder + 压缩器（general/emotion + gatenet），不加载 LLM 主干。

## 2. 特征保存（单文件）

- 输出目录：`<output_dir>/<dataset_name>/single_files/`
- 文件名包含 fps 信息，例如：`sample_00000091_fps2_features.pt`
- 每个文件包含：
  - `features`：特征张量
  - `feature_shape`：形状
  - `metadata`：`video_id`/`video_file`/`label`/`fps`

## 3. OOM 处理

- 单个视频发生 OOM 时，仅对该视频执行 `fps -> fps/2` 重试。
- 最低 fps 为 `0.5`，仍失败则跳过该视频。

## 4. 断点续传

- `logs/checkpoint.json` 记录已处理视频列表。
- 重启后自动跳过已完成样本。

## 5. inode 注意事项

- 单文件输出会产生大量小文件。
- 建议将输出目录放在 inode 充足的分区，必要时分批处理。