# 输出目录索引

所有输出均由仓库内脚本生成；原始输入在 [`data/raw/`](../data/raw/)。  
**注意：** 有效帧比例、静止抖动和测试通过都不是机械角度精度。当前视觉是未标定的二维投影角，IMU 安装轴未完成标定。

## 哪一个是最终结果

| 目录 | 状态 | 内容 | 关键数字 |
|---|---|---|---|
| [`fold_angle_fixed/`](fold_angle_fixed/) | **视觉最终版** | 固定下边方向 + 固定虚拟交点 + One Euro | 545 帧，544 有效；交点 (240,148) |
| [`imu_20261016_091225/`](imu_20261016_091225/) | **纯 IMU 最终版** | 52.7 s 固定底座重建 | 4,354 行；初始 180° 来自用户说明 |
| [`experiment_20261016_110453/`](experiment_20261016_110453/) | **同步实验最终版** | 视频 + 双 IMU 同步对照 | 1421 帧，1118 有效；1051 共同帧；固定互补 MAE 0.669° |
| [`conflict_sensitivity_20261020/`](conflict_sensitivity_20261020/) | **补充证据** | keep-first vs keep-last 敏感性 | 主结果 MAE 变化 −0.00065° |

## 过程/历史材料

| 目录 | 性质 |
|---|---|
| [`video_inspection/`](video_inspection/) | 最早的视频勘察：抽样帧、接触表、候选边线；不是最终测量结果 |
| [`fold_angle/`](fold_angle/) | 旧版视觉结果（535/545），保留作对照 |
| [`explanation_20261018/`](explanation_20261018/) | “六个问题”图解的 7 张图和复算脚本 |
| [`explanation_20261019/`](explanation_20261019/) | 通用双 IMU 模型示意图 |
| `experiment_20261016_110453/_probes/` | 已作废的旧初始化试跑，**不是最终证据** |

## 同步实验目录内部

| 子目录/文件 | 内容 |
|---|---|
| `report.md` | 完整中文实验报告（建议先看） |
| `verification.json` | 独立核验记录：指标重算、哈希、PNG、事件顺序、测试 |
| `experiment_manifest.json` | 输入/代码哈希与运行环境 |
| `video/` | 逐帧视觉角度、标注视频、诊断图、角度曲线 |
| `imu/` | 去重数据、逐点 IMU 角度、审计、曲线 |
| `synchronization/` | 时间偏移、角速度对齐、同步诊断 |
| `comparison/` | 共同帧对齐表、残差、按角度/速度分组指标 |
| `ablation/` | 单陀螺消融 |
| `report.md` 中的图片 | 已改为相对路径，GitHub 可直接渲染 |

## 大文件说明

- `experiment_20261016_110453/video/annotated.mp4` 约 **33 MB**，是 1421 帧 1280×720 的标注视频。
- `fold_angle_fixed/annotated.mp4` 约 5.6 MB；`fold_angle/annotated.mp4` 约 5.5 MB。
- 仓库总大小约 85 MB（含 `data/raw/` 原始数据）。

## 不要把输出当精度

- `valid_fraction`：有效输出比例，不是准确率。
- `support_score`：启发式图像支持分数，不是概率。
- 静止窗口 std / frame-change RMS：显示稳定性，不是机械角度误差。
- 0.669°：IMU 与**未标定视觉投影**的 MAE，不是机械绝对精度。
- 时间偏移 −1.217 s：动作层面的对应，不是硬件同步精度。
