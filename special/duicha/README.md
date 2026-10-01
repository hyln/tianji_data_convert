# 对插专项处理

### Legacy 过滤 + 五视图 state36

新批次使用 `duicha_feedback_effort_legacy.yaml`。`conversion_mode: feedback_effort_labeled`
负责五视图/反馈关节/effort/FK 数据构建，独立的 `frame_filter: legacy` 使用原转换器的
首次 Enable 前 3 帧及之后 Enable/Home/Inference 过滤；没有 Enable 时仍保留 Home/Inference。
不添加 idle 尾帧或释放后的截止时间。图像、因果对齐、action16 均不变。
task 为 `Insert the plug into the socket.`，输出到独立的 `feedback_effort_fk_5view224_legacy`。
保留原标注索引和 120 秒上限。过滤设置参与分片缓存校验，旧全帧分片不会复用。
未指定 `frame_filter` 时默认 `mode`，保持原模式行为。

```bash
python common/parallel_compressed_to_lerobot.py \
  --config special/duicha/duicha_feedback_effort_legacy.yaml
```

以下命令从 `tianji_data_convert` 根目录运行。转换核心在 `common/`；本目录保存对插批次配置、标定资产和增量裁剪入口。

### 对插数据：五路 224 图像、实测关节、effort 和反馈 FK（2026-09-29）

使用 `duicha_feedback_effort.yaml`，运行：

```bash
python common/parallel_compressed_to_lerobot.py \
  --config special/duicha/duicha_feedback_effort.yaml
```

本次配置保留五路 224×224 图像：head 原图缩放、两路腕部去畸变后的完整视野缩放、两路腕部去畸变后的原尺度局部裁剪。

新模式 `feedback_effort_labeled` 的 `observation.state` 为 float32[36]：

| 下标（左闭右开） | 内容 | 来源/单位 |
| --- | --- | --- |
| 0:14 | 左 7 + 右 7 实际关节角 | joint_state.position，rad |
| 14:28 | 左 7 + 右 7 关节 effort | 同一条 joint_state.effort，原值不缩放 |
| 28:34 | 左 XYZ + 右 XYZ | 从上述实际关节角计算的 FK，base_link，m |
| 34:36 | 左右夹爪实际反馈 | 同一条 joint_state.position |

Action 仍为 float32[16]，来自 joint_activate_cmd。新模式完全不读取 EEF CSV。
FK 离线编译 motion_control 的厂商 SDK，采用同一机器人参数和 ikMatrixToPose 的坐标变换；
不连接机器人、不发布控制指令。FK 输入就是保存到 State 的 float32 实际关节角。
缺失 effort、关节名称错误或非有限值会报错，不能静默补零。配置与 SDK 哈希写入转换报告。

一条源录制保留一个 episode。此模式保留所有能够因果同步的原始头部相机帧，包括空闲、
复位和推理帧；不执行 legacy 的 Enable/Home 过滤，也不因 Y 标记丢弃整条。
原始相机时间戳和帧号保存在 `source.timestamp_ns`、`source.head_frame_index`，
另有实际关节、Action、控制状态各自的来源时间戳。LeRobot `timestamp` 仍为 frame_index/FPS。

标签均为独立的 int64[1] 字段，不拼入 State：

- `annotation.is_rollout_batch`：用户明确指定的 Rollout+遥操介入批次。
- `annotation.is_teleop`：逐帧 left_enable OR right_enable；是录制的 Enable 信号。
- `annotation.left_enable/right_enable`、`left_arm_state/right_arm_state`：保留双臂原始模式。
- `annotation.is_inference`：任意一臂 arm_state==3；双臂模式可不同，可能与 is_teleop 同时为 1。
- `annotation.control_valid`：是否已有因果可用的控制状态；未知时 arm_state=-1。
- `annotation.has_error_marker`：整条源录制是否有 Y/error_marker；没有 Y 不等于每个插接段成功。
- `annotation.is_empty_scene`：已有标注中用户要求保留的空镜头，6 分保存在分段 JSON。

9 月 21 日批次的 `source_kind` 明确设为 `rollout_with_teleop_intervention`。
其余批次为 `unspecified`，不能仅凭 is_rollout_batch=0 断言是纯遥操数据。
`meta/source_annotations.json` 保留来源、人工结论、评分、复核状态，并把现有插接分段
映射为 `lerobot_frame_spans`（0 起始，左闭右开）。候选分段和首次到位待复核状态均原样保留。

新 YAML 支持 `conversion_mode`、`source_kind`、`motion_control_root`、`annotation_index`、
`max_duration_s`；dataset 可单独覆盖 `source_kind`。本配置严格排除 >120 秒源录制，
并只转换 annotation_index 中保留的 sources。输出使用新目录，已有输出不会覆盖。
分片续跑校验来源、CSV 内容、视频大小/修改时间、FK 实现/参数以及标签配置。
成功后自动清理分片；`meta/fk_config.MvKDCfg` 保留本次 FK 参数快照。

历史模式 `legacy` / `enable_only` / `full_rollout_labeled` 仍保留原来的 CMD-FK schema，
不适用于本次 36 维新数据。训练和推理都需要采用相同的 36 维 schema 并重新计算归一化统计。

### 已有三路数据：增量补充两路 crop（本次使用）

已有 `feedback_effort_fk_224` 的四套三路数据时，使用增量工具：

```bash
python special/duicha/add_wrist_crop_views.py \
  --config special/duicha/duicha_add_wrist_crops.yaml
```

只从原始腕部相机 6、7 生成两路 crop，并严格使用已存在数据表的时间戳逐帧对齐。
原三路视频、data Parquet、原有统计、分段标注和 episode 元数据列全部复用。
输出单独存入 `feedback_effort_fk_5view224`。外置盘为 exFAT，旧文件使用字节复制，
不解码或重新编码；复制后和完成前均核对 SHA-256。原始录像指纹必须匹配初次转换记录。

新增视频按 episode 独立保存，使用 LeRobot v3 标准 chunk/file 路径，fps 和帧数不变。
只新增两路对应的视频索引、feature、内参和统计。crop 统计覆盖每个 RGB 像素，使用 256-bin
精确直方图计算均值、标准差和分位数，先转浮点运算避免 uint8 平方溢出；已有字段统计不变。
新视频和每条转换记录支持校验后续跑，最终数据目录在全部完成后才原子发布。

下面的原始数据完整转换配置适用于新录制或从零生成数据；本次已有三路数据采用以上增量入口。

### 五路图像：完整视野与腕部裁剪并存

`duicha_feedback_effort.yaml` 使用 `shared_camera_intrinsics_5view224.json`。
原始分辨率是 1200×1000，三台相机共用相同的近似基础 K；腕部使用各自 equidistant 系数。

| LeRobot 字段（均在 observation.images 下） | 处理方式 | 输出 |
| --- | --- | --- |
| head | 相机 8 原图直接 AREA 缩放，不去畸变 | 224×224 |
| hik_camera_6 | 相机 6 去畸变，完整视野缩放 | 224×224 |
| hik_camera_7 | 相机 7 去畸变，完整视野缩放 | 224×224 |
| hik_camera_6_crop | 相机 6 原尺度去畸变，裁 x=[488,712)、y=[332,556) | 224×224 |
| hik_camera_7_crop | 相机 7 原尺度去畸变，裁 x=[488,712)、y=[388,612) | 224×224 |

相机 6 局部裁剪框在居中基础上向上移动 56 像素，即 224 的四分之一。局部视野不先缩放。
相机 8 保留完整视野，不做 crop。腕部完整视野沿用 448×448 remap 后 AREA 缩放至 224 的抗锯齿方法。
每台物理相机只解码一次，两种腕部视野由完全相同的原始帧生成，使用相同因果时间对齐。

`meta/camera_intrinsics.json` 保存共用的 `source_camera_matrix`、全部视图定义、裁剪矩形和
`per_camera_intrinsics`。完整视野按缩放比例换算 K；裁剪视野保留 fx/fy，cx/cy 减去裁剪起点。
因此五路的输出 K 有区别，基础 K 共用。`wrist_undistortion.camera_processors` 可供推理复用，
返回四种腕部处理函数；head 使用原图 AREA resize。

新输出根目录为 `feedback_effort_fk_5view224`。原三路缩放版保留。旧缩放与仅 crop 配置继续兼容，
但当前对插 YAML 默认生成五路图像。续跑检查五路字段及尺寸、相机配置哈希，拒绝复用三路分片。
训练时按模型需要选择图像字段；部署推理需要同步五路图像预处理、36 维 State 和新归一化统计。
本次没有修改机器人推理程序。
