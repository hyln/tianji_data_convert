# Tianji 数据转换

按复用范围分为两层：**common 是通用能力，special 是特定任务/设备的配置与处理。**

```text
tianji_data_convert/
├── common/                    # MP4+CSV / bag 转换、并行聚合、相机/FK/标签工具
└── special/
    ├── candy/                 # 抓糖批次转换配置
    ├── duicha/                # 对插批次、相机标定、增量腕部裁剪
    └── trash_data_filter/     # 特定关节规则的数据筛查与归档
```

以下命令均从本目录运行，使用已有 LeRobot 环境：
`/home/seen-e-embodied/miniconda3/envs/lerobot/bin/python`。
不新增统一封装入口，直接运行对应脚本即可。

## 常用入口

通用 MP4+CSV → LeRobot：

```bash
python common/parallel_compressed_to_lerobot.py SOURCE_ROOT OUTPUT_DATASET \
  --repo-id local/DATASET_NAME --workers 8
```

按任务配置运行：

```bash
# 对插：五路 224 图像、反馈/effort/FK State[36]
python common/parallel_compressed_to_lerobot.py \
  --config special/duicha/duicha_feedback_effort.yaml

# 已有三路数据增量补充两路腕部 crop
python special/duicha/add_wrist_crop_views.py \
  --config special/duicha/duicha_add_wrist_crops.yaml

# 抓糖：原先 config.yaml 中的 legacy 配置
python common/parallel_compressed_to_lerobot.py \
  --config special/candy/conversion.yaml
```

配置保留了原来的真实输入/输出路径；**不是运行即安全的空白模板**。
执行前检查目标目录，不要覆盖已有数据。相对标定路径仍以 YAML 所在目录解析。

只读筛查异常录制：

```bash
python special/trash_data_filter/scan.py SOURCE \
  --filter left_wrist_weighted_expectation_over_118 \
  --output /tmp/tianji_filter_report.json
```

只有显式加 `--move` 才会移动录制；本次整理不执行扫描、归档或真实转换。

## 详细说明

- [common：转换参数、过滤规则与共用模块](common/README.md)
- [对插：36 维反馈状态、五路图像与增量裁剪](special/duicha/README.md)
- [trash_data_filter：规则、只读检查及归档恢复](special/trash_data_filter/README.md)

默认 legacy 模式是 State[22]/Action[16]，对插 feedback 模式是 State[36]/Action[16]；
两种 schema 不混用。转换、标注与过滤行为未因目录整理而修改。

## 维护约定

- 多任务可复用的实现放 `common/`；具体数据批次、阈值规则和标定资产放 `special/<任务>/`。
- `common` 不自动加载某个 `special` 配置；任务通过命令行或 YAML 显式选择。
- 不在源码目录保存数据集、日志或中间分片。若需本地输出可使用被 Git 忽略的 `output/`。
- 旧 `lerobot_convert/` 和顶层 `trash_data_filter/` 路径已取消，请更新个人命令。
- 原 `tests/` 按要求移除；空的 `config/tianji_config.yaml` 占位文件一并清理。
- 整理前源码和测试备份：`/tmp/tianji-convert-reorganize.etVbnl/source-before.tgz`（临时备份）。
- 本目录使用独立 Git 仓库；父仓库的子模块设置单独维护。
