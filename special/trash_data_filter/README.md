# Trash Data Filter

`trash_data_filter` 是转换前独立运行的不良数据清理工具，不依赖也不修改转换程序。

## 左腕 J7 换位

`left_wrist_j7_branch_switch` 检测左臂接近腕部耦合边界时，J6被限制而J7快速换位：

- 只分析左臂 `Enable + Tracking`。
- 1秒窗口，0.1秒步长；跳过 Enable 后0.3秒及 Disable 前0.05秒。
- `|J6| >= 58°` 的样本占窗口至少80%。
- J6窗口极差不超过0.6°。
- J7净变化至少55°。
- `|J7净变化| / J7累计路程 >= 0.8`。

相邻命中窗口合并为一个事件。报告同时给出边界参考量
`1.025|J6| + |J7|`；107°是解释用的耦合边界，不作为额外硬筛选条件。

## 左腕加权期望和超限

`left_wrist_weighted_expectation_over_118` 检查整条左臂关节命令：

```text
1.025 × |J6_cmd| + |J7_cmd| > 118°
```

任意命令样本严格超过118°即判定该session异常。报告包含峰值、峰值时间、峰值处
J6/J7、超限样本数和合并后的超限区间。该规则与J7换位过滤器彼此独立。

先只读扫描并检查候选：

```bash
python3 special/trash_data_filter/scan.py \
  "/media/seen-e-embodied/SANDISK ELE/tianji_record/compressed_records" \
  --filter left_wrist_j7_branch_switch \
  --output trash_data_filter_report.json
```

扫描加权期望和超限：

```bash
python3 special/trash_data_filter/scan.py SOURCE \
  --filter left_wrist_weighted_expectation_over_118 \
  --output weighted_expectation_report.json
```

确认后将命中session移出源目录并归档到 Documents：

```bash
python3 special/trash_data_filter/scan.py SOURCE \
  --filter left_wrist_weighted_expectation_over_118 \
  --output trash_data_filter_report.json \
  --move
```

`--move` 不执行永久删除。默认归档结构为：

```text
~/Documents/tianji_trash_data/
└── <源目录名>/
    └── <批次时间>/
        ├── sessions/                      # 完整原始session
        ├── README.md
        ├── candidate_ids.txt
        ├── move_manifest.json             # 原路径与恢复路径
        └── trash_data_filter_report.json  # 完整判定指标
```

可用 `--archive-dir` 指定其他归档根目录。源目录和 Documents 位于不同文件系统时，
工具会完成复制后再移除对应源session，并持续更新 `move_manifest.json`。

任何session读取失败时不会移动任何数据，命令以非零状态退出。清理完成后再独立运行
LeRobot转换程序即可；转换程序本身不感知该过滤器。
