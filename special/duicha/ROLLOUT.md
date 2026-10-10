# Rollout State37 协议

使用 `full_rollout_labeled`。保留具有已知 control、state、action 及有效图像的同步帧；
不按 Enable 删除自主推理帧，不因 Y 标记删除整条录制，不输出 `neg_rollout`。

- State[0:14]：反馈关节位置；[14:28]：原始 effort。
- State[28:34]：当前反馈关节 FK 得到的左右 EEF XYZ；[34:36]：左右夹爪。
- State[36]：`is_teleop = left_enable OR right_enable`，float32 0/1。
- Action[16]：现有 joint_activate_cmd 按名称对齐，不改 action 构建。
- 五路 224：head、cam6、cam7、cam6 crop、cam7 crop；沿用共享去畸变和裁剪配置。
- 对齐：原始 head 时间戳，其他信号取不晚于 head 的最近样本，不按 FPS 抽帧或插帧。
- `source.timestamp_ns` 保存原始 head 时间；LeRobot 的 timestamp 仍为 frame_index/fps。
- Task：`Insert the plug into the socket.`

`duicha_rollout_state37_20261001.yaml` 仅选取 `tianji_20261001` 前缀、manifest complete 的录制，
无时长或人工索引筛选，输出到本机 output/lerobot，保留分片便于续跑。
旧 State24 rollout 分片不兼容，转换 revision 已更新以避免复用。

运行需要提供 `lerobot.datasets.lerobot_dataset` 接口的独立转换环境；
不能使用仅有旧版 `lerobot.common.datasets` 的推理环境直接运行。
训练时应明确将前 36 维作为物理状态、最后一维作为标签，避免无意将标签当作状态输入。
