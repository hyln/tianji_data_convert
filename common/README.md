# 通用转换工具

本目录放多任务复用的转换器和辅助模块；以下命令均从 `tianji_data_convert` 根目录运行。
任务专用 YAML 和标定文件见 `special/`，不要往 common 添加一次性批次配置。

## ROS bag compact archive

Convert a ROS 2 SQLite bag into H.265 MP4 videos plus one CSV per non-video
topic. Each video also has a frame timestamp CSV for exact ROS time alignment.

Run with ROS Humble's Python 3.10 and source the workspaces that provide custom
message definitions:

```bash
source /opt/ros/humble/setup.bash
source /home/seen-e-embodied/marvin_msgs_ws/install/setup.bash
source ../tianji_arm_teleop/install/setup.bash
/usr/bin/python3 common/bag_to_archive.py BAG_DIR OUTPUT_DIR
```

Use NVIDIA NVENC when a supported GPU is available:

```bash
/usr/bin/python3 common/bag_to_archive.py BAG_DIR OUTPUT_DIR --encoder hevc_nvenc
```

NVENC mode uses three parallel video workers and NVIDIA MJPEG hardware decoding
by default. Pass `--cpu-decode` only for GPUs without MJPEG NVDEC support.

The output is written through an adjacent `.partial` directory and renamed only
after a complete conversion.

## Parallel compressed_records to LeRobot v3

`parallel_compressed_to_lerobot.py` converts every complete MP4+CSV session into
one temporary one-episode shard in parallel, then uses LeRobot's official
`aggregate_datasets()` to build a single multi-episode dataset. H.264 video is
encoded only once; aggregation concatenates the streams without re-encoding.

Default command-line settings match the Tianji training dataset contract: 256x256,
H.264 CRF 18, float32 state[22], float32 action[16], causal timestamp alignment,
and per-frame Enable-or-Home-or-Inference filtering. Before the first Enable,
only its three immediately preceding original head-camera frames are kept regardless of control
state. From that Enable onward, inference frames are kept regardless of Enable.
After Enable release, the same per-frame Enable/Home/Inference rule continues
throughout the recording. There is no idle tail allowance or release-based cutoff.
There is no minimum episode duration.

### YAML 配置（推荐）

参考 [抓糖批次配置](../special/candy/conversion.yaml)，再在 `tianji_data_convert` 根目录运行：

```bash
python common/parallel_compressed_to_lerobot.py \
  --config special/candy/conversion.yaml
```

使用 LeRobot Python 环境。配置包含输入/输出根目录、公共转换参数和数据集列表：

- `conversion`：`image_size`、`video_crf`、`workers`、`keep_work`、`task`。
- `datasets`：每项指定 `source`、`output`，可选 `repo_id`（默认 `local/<输出目录名>`）。
- 多个数据集顺序执行；单个数据集内部按 `workers` 并行。
- 相对根目录基于 YAML 文件所在目录；任务路径分别相对于输入、输出根目录。
- 先校验整批配置；未知字段、路径冲突、已有输出会报错，不覆盖数据。
- 每个任务的完整日志写入 `<output>.conversion.log`；失败时停止后续任务并保留分片。
- 失败后用同一配置重试会复用匹配的分片；成功后按 `keep_work` 决定是否保留。
- YAML 与单任务命令行参数不能混用；转换完成不会自动上传。

这些是已有任务配置；执行前检查输出是否存在，如需重转，请使用新的 `output` 名称。

### 不通过 FPS 抽帧

配置和命令行均不提供 `fps`/`--fps`。所有原始头部相机帧都会进入既有的控制状态与
同步有效性过滤，不再有 `stride` 抽帧。其余相机、state/action 的因果对齐规则不变。

LeRobot 和 H.264 仍需标称帧率：从头部相机原始时间戳的帧间隔中位数计算并四舍五入到
整数（本次录制为 30 FPS），只用于输出时间基准，不用于丢帧或插帧。
时间戳不足两帧、非严格递增、或同一数据集混合不同标称帧率时，转换会报错。
原始时间戳用于对齐；输出 `timestamp` 仍按连续帧索引/FPS 编排，过滤掉的空闲时段不保留。

Both compressed-session converters accept `--image-size 224|256|512` (default: 256).
This sets all three video streams and their feature shapes to the chosen square
resolution without changing filtering, alignment, or CRF. Use a separate
output directory, e.g. `<dataset>_lerobot_512`, for 512x512 conversion. Cached
shards are reused only when the conversion revision, source path, task text, CRF,
and all camera shapes match. Older resampling-capable shards are not reused.
Decoding at 512x512 uses approximately four times the image memory; start with
`--workers 4` and run large datasets sequentially.

Sessions with an empty required camera frame index are skipped and reported as
`skip_invalid` before launching a worker. Camera timestamp/decoded-frame count
mismatches still fail conversion; they are not silently ignored. Camera timestamps
are cached once per session for causal alignment.

Run in the LeRobot conda environment:

```bash
/home/seen-e-embodied/miniconda3/envs/lerobot/bin/python \
  common/parallel_compressed_to_lerobot.py \
  "/media/seen-e-embodied/SANDISK ELE/tianji_record/compressed_records" \
  "/media/seen-e-embodied/SANDISK ELE/tianji_record/tianji_rdc_lerobot" \
  --repo-id local/tianji_rdc_lerobot \
  --workers 8
```


## 共用模块

- `camera_metadata.py` / `wrist_undistortion.py`：读取显式传入的相机配置、生成视图与内参。
- `feedback_state.py` / `feedback_fk.cpp`：实测关节/effort/FK schema，供多种任务调用；不是对插专用标定。
- `conversion_annotations.py`：来源和分段标签导出。
- `conversion_config.py`：严格解析 YAML、检查参数和输入/输出冲突。
- `visualize_episode_alignment.py` / `visualize_joint_alignment.py`：检查已有数据的时间与关节对齐。

直接以脚本路径运行入口；并行 worker 和 FK C++ 桥接文件仍在同目录。此次只整理文件，不改变转换模式、过滤规则或输出 schema。
