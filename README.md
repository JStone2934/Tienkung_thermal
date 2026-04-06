# tienkung_thermal

腿部关节温度采集、神经网络温度预测、以及与 **TienKung-Lab** / **Deploy_Tienkung** 对接的约定与工具。

## 安装

在仓库根目录（本目录）执行：

```bash
pip install -e .
# 训练：pip install -e ".[train]"
# rosbag：pip install -e ".[rosbag]"
```

在 **TienKung-Lab** 环境中可通过 path 依赖引用本包，例如在 Lab 的 `pyproject.toml` 或 `requirements.txt` 中加入：

```text
tienkung-thermal @ file:///absolute/path/to/tienkung_thermal
```

## 数据格式约定

| 字段 | 说明 |
|------|------|
| `timestamp` | 单调时间戳（秒，仿真或 ROS `header.stamp` 转秒） |
| `T_leg` | 形状 `(n_leg_motors,)` 的腿部温度向量，**单位 °C**；顺序见 `configs/leg_index_mapping.yaml` |
| （可选）`q_leg`, `qdot_leg`, `tau_leg` | 与 Lab/Deploy 腿索引对齐后可一并存表，供监督或多任务 |

导出推荐：**Parquet** 或 **NPZ**（训练）；原始 **rosbag2** 由 `scripts/extract_from_rosbag.py` 解析。

## ROS 2 Topic 约定（Deploy_Tienkung）

与部署栈对齐时，腿部状态来自 `bodyctrl_msgs/msg/MotorStatusMsg`（见 Deploy `rl_control_new` 订阅 `/leg/status`）。单轴 `MotorStatus` 中温度字段为 **`float32 temperature`**，**单位摄氏度 (°C)**；权威定义见 **`TienKung_ROS`** 仓库 `src/bodyctrl_msgs/msg/MotorStatus.msg`（与 `plan.md` §0 一致）。

记录 rosbag 时请固定：

- 话题名（示例）：`/leg/status`
- 与 `configs/leg_index_mapping.yaml` 中 **`motor_name` → 列顺序** 一致，便于与 `Temperature` 向量对齐。

## P0 实机确认脚本

`scripts/check/p0_check.py` **仅在实机**（已 `source` ROS2、`/leg/status` 有数据）运行，用于 `ros2 interface show` 与话题样本核对；**不**扫描三仓库源码。说明见 `docs/todo.md`「P0 核对脚本」。

```bash
python scripts/check/p0_check.py
python scripts/check/p0_check.py --dt 0.0025   # 可选：与 tg22_config 的 dt 对比消息间隔
```

## 与 TienKung-Lab 的版本约定

- **Python**：与 Lab 一致建议 **≥3.10**（Lab README 为 3.10）。
- **关节顺序（硬性）**：**Ultra**（`TienKung-Lab` 的 `ultra_env.py` 中腿 `find_joints` 顺序）为 **`T_leg[0..11]` 唯一准则**；`configs/leg_index_mapping.yaml` 与之对齐。Deploy `bodyIdMap` 腿向量顺序不同，**禁止**按下标与 `T_leg` 混用，须按关节语义映射后再接入训练/落盘。
- **观测拼接**：`tienkung_thermal/lab_hooks/` 仅描述 **拼接维度、历史长度、归一化统计量路径**；具体改 `UltraEnv.compute_observations` 等需在 Lab 侧按约定接入（本包不强制改 Lab 源码）。

## 与 Deploy_Tienkung 的版本约定

- 推理若部署在 C++：可先用 **TorchScript / ONNX**（`scripts/export_onnx_or_torchscript.py`）；与现有 **OpenVINO** 策略并行加载需自行在 `RobotInterface` 或独立节点中集成（本包提供导出脚本占位）。
- 控制周期：Deploy `tg22_config.yaml` 中 `dt` 与训练数据时间对齐方式应在 `configs/thermal_predictor.yaml` 中写明 `target_hz` 或 `decimation` 约定。

## 目录说明

- `configs/`：特征列、历史长度、腿索引映射、模型超参。
- `tienkung_thermal/datasets/`：rosbag/表格 → `Dataset`。
- `tienkung_thermal/models/`：温度预测网络与轻量 torch 推理 API。
- `tienkung_thermal/lab_hooks/`：与 Lab 对接的接口与形状约定（无强制 patch）。
- `scripts/`：提取、训练、导出命令行入口。

## License

BSD-3-Clause（与上层项目保持一致时可再统一）。
