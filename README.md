# 折叠机开合角视觉测量

**汇报用总文档（从原始数据到结果，含讲稿、Q&A 和边界说明）：** [docs/2026-10-20-项目汇报_从原始数据到结果.md](docs/2026-10-20-项目汇报_从原始数据到结果.md)

任意手持的通用物理模型见[通用双 IMU 开合角：物理模型与完整数学表达](docs/通用双IMU开合角_物理模型与完整数学表达.md)：从铰链机械约束推导双 Gyro 差分、双 Acc 测角、运动加速度补偿和角度＋零偏卡尔曼模型，并明确与当前实现的区别。

如果不清楚 h、e1/e2 的方向、点积或梯形积分，请先看[手机实景与坐标图解：六个问题逐步解答](docs/手机铰链轴与IMU角度计算_六个问题图解.md)。包含 7 张示意图、原始 CSV 行配对、鲁棒拟合目标函数，以及一条样本的完整投影、积分和校正演算。

从最原始 CSV 开始学习当前模型，请先读[CSV 到下一条 IMU 数据的完整计算教程](docs/从原始CSV到下一条IMU数据_完整计算教程.md)，配合[4085点全部中间量追踪表](outputs/experiment_20261016_110453/csv_to_angle_trace.csv)。教程明确区分参数、逐点状态、当前实现与新采集需要重新初始化的内容。

从采集数据到最终角度的逐步解释见[处理原理与实算教程](docs/2026-10-16-synchronized-experiment-principles.md)：包含去重、两类时间对齐、边线测角、gyro 积分、Acc 重力相位、互补校正、逐点残差和单 gyro 对照，每一步均给出公式、实际数值及适用条件。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `data/raw/` | 原始视频与传感器 CSV（未修改，含 SHA-256） |
| `docs/` | 原理、教程、汇报文档；先看 [汇报总文档](docs/2026-10-20-项目汇报_从原始数据到结果.md) |
| `outputs/` | 实验结果；先看 [输出目录索引](outputs/README.md) |
| `fold_angle/` | 视觉检测、同步、对照模块 |
| `configs/` | 场景初始化配置 |
| `tests/` | 29 个测试 |
| `measure_video.py` | 纯视觉入口 |
| `analyze_imu_log.py` | 纯 IMU 入口 |
| `analyze_synchronized_experiment.py` | 同步实验入口 |
| `verify_synchronized_experiment.py` | 独立核验入口 |

## 2026-10-16 11:04 同次视频与双 IMU 同步实验

本次数据、同步角速度、投影角残差、单 gyro 对照和验证记录见 [实验报告](outputs/experiment_20261016_110453/report.md)。独立入口为 `analyze_synchronized_experiment.py`，新视频初始化配置为 `configs/experiment_20261016_110453.json`。本次输出位于独立实验目录，不覆盖早期结果；视觉参考仍是未经几何标定的二维投影角，单 gyro 对照仅适用于当前固定底座。

```powershell
python analyze_synchronized_experiment.py --video 'data/raw/1016110453.mp4' --imu 'data/raw/sensor_log_2026-10-16_11-04-53.csv' --config configs/experiment_20261016_110453.json --output outputs/experiment_20261016_110453 --rerun-video
python verify_synchronized_experiment.py --output outputs/experiment_20261016_110453
```

详细阶段总结见 [折叠手机开合角估计：阶段工作总结与后续路线](docs/2026-10-16-fold-angle-progress-summary.md)，涵盖视觉、IMU 数据处理与滤波、实测结果，以及实时融合和减少一个 gyro 的后续方案。

## 2026-10-16 纯 IMU 实测分析

已增加 `analyze_imu_log.py` 与 `imu_angle.py`。本次传感器采集独立于此前视频，尚未做视觉融合。用户确认 gyro 单位 rad/s，主 IMU 固定、副 IMU 转动；acc1/gyro1 对应副 IMU 的映射由信号变化推断。主结果利用固定底座约束，对副 IMU 的陀螺积分并用低动态加速度校正。双 IMU 曲线另外保留，但其安装轴向与跨时钟同步尚未标定。

- `outputs/imu_20261016_091225/angle_curve.png`：加速度角、陀螺积分、固定底座融合与暂定双 IMU 对比。
- `outputs/imu_20261016_091225/angles.csv`：4,354 个实际独立 gyro1 时刻的逐点估计。
- `outputs/imu_20261016_091225/report.md`：数据质量问题、静止段统计及复现命令。
- `outputs/imu_20261016_091225/data_audit.json`、`summary.json`、`verification.json`：完整诊断与核验。

```powershell
python analyze_imu_log.py --input 'data/raw/sensor_log_2026-10-16_09-12-25.csv' --output outputs/imu_20261016_091225
python -m pytest tests -q
```

当前数据包含大量缓存重复记录、少量同时间戳不同值以及两种时钟。结果不裁剪到 0–180°，初始 180° 来自用户动作说明，不能根据端点读数推断全程物理精度。

## 视觉测量

利用上下半机身靠近相机的两条长侧边，逐帧测量**二维投影开合角**，约定合拢0°、展开180°。

当前默认版本采用**固定下边方向＋固定公共交点**：下边仅初始化一次，逐帧只更新上边方向。输出画面先补偿误操作产生的平移，两条参考射线始终从同一个固定点出发；上边角度使用自适应One Euro滤波，静止时平滑、转动时提高响应速度。原始读数、滤波读数和实际观测线段分开导出。

本版本针对给定视频采用手动初始化ROI、OpenCV模板跟踪和受约束的LSD直线检测，**没有训练或使用YOLO模型**。普通目标框不提供侧边方向；后续扩展到更多手机或视角时，可标注侧边端点并训练关键点检测模型。

## 查看结果

- `outputs/fold_angle_fixed/annotated.mp4`：稳定版视频。绿线为固定参考下边，红线为滤波后的上边射线，黄色交点固定；灰线为实际观测上边。
- `outputs/fold_angle_fixed/angles.csv`：原始/滤波角度、固定模型射线、原视频中的观测线段、平移补偿量和无效状态。
- `outputs/fold_angle_fixed/angle_curve.png`：时间—角度曲线，无效位置断开。
- `outputs/fold_angle_fixed/diagnostics.jpg`：24个时刻的叠加图。
- `outputs/fold_angle_fixed/summary.json`：方法、配置和运行统计。
- `outputs/fold_angle_fixed/verification.json`：测试及固定几何/输出完整性检查。
- `outputs/fold_angle_fixed/stability_comparison.json`及`.png`：与旧版的静止窗口稳定性对比。

当前版本：545帧、30 fps，544帧有结果，1帧角度留空。**99.82%是有效输出比例，不是准确率**。合拢读数约0°，末尾展开约179.5°～179.7°，没有强制把端点改写成0°或180°。旧版535/545帧结果保留在`outputs/fold_angle/`，对应配置为`configs/sample_video_legacy.json`。

## 运行

需要 Python 3.10 或以上。以下命令在仓库根目录运行：

```powershell
python measure_video.py --video 'data/raw/243eca009e4449f611a92a5437cc4d66.mp4'
```

其他环境先安装依赖，再运行和验证：

```powershell
python -m pip install -r requirements.txt
python measure_video.py --video 'data/raw/243eca009e4449f611a92a5437cc4d66.mp4' --config configs/sample_video.json --output outputs/fold_angle_fixed
python -m unittest discover -s tests -v
```

重复运行会覆盖指定输出目录中的同名结果。标注视频不保留原始音轨。

## 配置与适用范围

按固定机位、下半机身固定处理。片段中误操作引起的小幅位移由局部平移匹配补偿；未实现任意相机运动或三维旋转下的跟踪。

`configs/sample_video.json`针对当前480×256视频首帧手动初始化：

- `base_template_box`：下半机身纹理条带，格式为左、上、右、下，裁剪不包含右下边界。
- `lower_edge`：下半机身侧边的两个参考端点。
- `hinge`：约束搜索的铰链附近锚点，不是精确机械转轴。
- `body_length`：约150像素的单侧机身长度，用于过滤其他长线。
- `fixed_geometry`：默认true。固定初始化下边方向和虚拟交点，不逐帧重估下边方向。
- `angle_filter`：`min_cutoff_hz`控制静止平滑，`beta`控制转动时跟随速度，`derivative_cutoff_hz`控制速度估计的平滑。

更换视频时应重新设置初始化坐标。改变拍摄方向、颜色、背景或尺寸都可能需要重新调整视觉约束。当前算法使用暗色边框先验。

真实边框有厚度，观测线段不必严格交于同一像素；将测得的方向移到公共虚拟交点绘制，不用强迫实际边框穿过该点来拟合角度。近合拢时仍需看到独立上边，不能重复选择同一条下边。固定下边的已知几何不因某一帧检测不到下边而丢失，但模板失配或没有可信上边时仍拒绝输出角度。

## CSV字段

| 字段 | 含义 |
|---|---|
| `frame_index` | 从0开始的帧号 |
| `timestamp_s` | 帧号/标称帧率；按恒定帧率处理 |
| `projected_angle_deg` | 固定下边与兼容上边方向融合后的原始投影角度；无效时为空 |
| `smoothed_projected_angle_deg` / `stabilized_projected_angle_deg` | 相同的One Euro滤波角度；视频显示此值；无效后重置 |
| `selected_edge_projected_angle_deg` | 单独选中观测上边相对固定下边的角度，可由观测线段复算 |
| `fusion_count` | 融合的兼容上边线段数；角度仅在选中方向附近2.5°内融合 |
| `valid` | 1为通过当前图像约束，0为缺少可用测量 |
| `support_score` | 启发式支持分数，不是概率或准确率 |
| `base_match_score` | 下半机身模板匹配分数 |
| `failure_reason` | 失败原因 |
| `interpolated` | 始终为0；不填补缺失角度 |
| `lower_*` / `upper_*` | 稳定画面坐标中的模型射线；共同起点固定；射线夹角等于滤波角度 |
| `observed_lower_*` / `observed_upper_*` | 原视频坐标中的实际观测线段；没有观测时为空 |
| `hinge_anchor_*` | 稳定画面坐标中的固定虚拟交点 |
| `registration_dx` / `registration_dy` | 原视频相对参考画面的平移；显示时取负值补偿 |
| `geometry_coordinate_system` | 固定模式为`stabilized_fixed_rays`；旧版为原图观测线段 |

## 几何与验证边界

两条边均从铰链端指向自由端，向量为u和v：

`angle = atan2(abs(u.x*v.y - u.y*v.x), dot(u,v)) * 180/pi`

合拢方向相同，展开方向相反，可区分30°与150°。机身厚度和保护壳会造成可见轮廓与理想中面的差异。

当前没有相机/运动平面标定或独立机械角度真值。人工图像点位粗查只能检查投影几何，不能用于声称真实角度精度。真实角度需校正透视，并用已知角度夹具或角度计验证；只用0°和180°做线性缩放，不能保证消除中间角度的透视误差。

滤波会产生响应延迟；本片段快速开合时主要为几十毫秒量级，具体对照原始和滤波列。无效帧不插值、不沿用旧角度。稳定视频中的暗色边界来自平移补偿，不改变角度几何。

静止窗口对比以0–3秒合拢、15.6–18.1秒展开为诊断区间。相邻帧角度变化的RMS：合拢旧版中值0.02264°、新版0.01803°；展开旧版0.06030°、新版0.00779°。整段窗口的标准差也在JSON中保留，包含慢漂移，与帧间抖动不同。上述指标只说明这一片段的显示稳定性，不能视作真实机械角度误差。
