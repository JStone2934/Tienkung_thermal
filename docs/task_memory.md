# TienKung Ultra 腿部热建模 — 任务记忆（关键信息汇总）

> **关节顺序（唯一准则）**：以 **Ultra** 为准（`TienKung-Lab/legged_lab/envs/ultra/ultra_env.py` 中 `find_joints`）；**禁止**用 Deploy 腿向量下标代替 `T_leg[i]`。  
> **数据**：**仅**使用 **`Deploy_Tienkung`** 与 **`TienKung-Lab`** 两仓库内**已定义**的接口（源码/配置/README/注释中的布局）；其余一律为 **待获取备选**，不得默认存在。  
> 详细条款见 `Tienkung_thermal/docs/plan.md` §0、§1。  
> **`temperature` 单位**：**摄氏度 (°C)**，类型 **`float32` 单标量**；IDL 见 **`TienKung_ROS`** `src/bodyctrl_msgs/msg/MotorStatus.msg`，解码见同仓库 `MotorDevice.cpp`（与 `plan.md` §0 一致）。

---

## 1. 目标与范围

| 项 | 内容 |
|:---|:-----|
| 机器人 | 天工 **Ultra**（与 Lab `ultra_env` 一致） |
| 自由度 | **12** 腿，顺序 **= Ultra**（`ultra_env`）；`leg_index_mapping.yaml` 仅与之对齐引用，**不以 Deploy 下标代替** |
| 验收 | 12 关节**等权**平均 MAE ≤ 1.5°C（@15s 等视距以 `plan.md` 为准） |
| 权重 | 配置 `w_i` + 可选可学习；Gate 只看**等权** MAE |
| 推理 | ≤ 5 ms FP16 |

---

## 2. 准许使用的数据（白名单）

### 2.1 Deploy_Tienkung（`RLControlNewPlugin.cpp` + `tg22_config.yaml` + `bodyIdMap.h`）

**Topic**

| Topic | 消息类型（头文件） |
|:------|:-------------------|
| `/leg/status` | `bodyctrl_msgs::msg::MotorStatusMsg` |
| `/arm/status` | 同上 |
| `/imu/status` | `bodyctrl_msgs::msg::Imu` |
| `/sbus_data` | `sensor_msgs::msg::Joy` |
| `/leg/cmd_ctrl`、`/arm/cmd_ctrl`、`/waist/cmd_pos` | 发布侧，见插件 |

**每条 `MotorStatusMsg.status[]` 元素 `one`（插件已读）**

- `pos`, `speed`, `current`, **`temperature`（`float32`，°C）**, `name`（用于 `getIndexById`）；与 **TienKung_ROS** `MotorStatus.msg` 一致

**`tg22_config.yaml`（插件加载）**

- `dt`, `ct_scale`, `motor_num`, …（热相关主要为 **dt、ct_scale**）

**`Imu`（插件已读成员）**

- `euler.{yaw,pitch,roll}`, `angular_velocity.{x,y,z}`, `linear_acceleration.{x,y,z}`（共 9 标量）

**映射表**

- `bodyIdMap.h`：腿 **0–11** 为 Deploy **中间向量**顺序（髋 R–P–Y，**≠** Ultra）；仅用于 **名称/CAN → j**，再 **按语义映射** 到 **Ultra** 的 `T_leg[0..11]`（唯一准则见 `plan.md` §1.2）。

### 2.2 TienKung-Lab

- `README.md`：版本、任务 `ultra_walk`/`ultra_run`、数据集路径、`train`/`play`/`sim2sim_ultra` 命令。  
- `ultra_env.py`：**腿关节名顺序**；`visualize_motion` **44 维**帧说明；`get_amp_obs_for_expert_trans` **AMP 布局**文档字符串。  
- `motion_loader*.py`：JSON `Frames` / `FrameDuration` / `MotionWeight`（**无温度**）。  
- 仿真：`robot.data.joint_pos` / `joint_vel` + `*_leg_ids`（**无温度字段**）。

---

## 3. 待获取备选数据（两仓库未定义或未接入）

| 项 | 说明 |
|:---|:-----|
| ~~`MotorStatus.temperature` 类型与单位~~ | **已确定**：见 **`TienKung_ROS`** `MotorStatus.msg`（`float32`，单通道）及 **`plan.md` §0**（**°C**） |
| `MotorStatusMsg` 与实机固件是否分叉 | 以 bag / `ros2 interface show` 与 **TienKung_ROS** 定义交叉核对 |
| 电压、原生 `ddq`、故障/CRC | 插件未读；其它 Topic 未在两仓库列出 |
| BMS、主板温、风扇 | 插件未订阅 |
| 环境温度计 | 非仓库接口 |
| Lab 电机温度仿真 | 仓库内未出现 |
| 第三方机器人 DDS | 本专项基线排除 |

**允许的后处理**：对已白名单的 `speed` 等做**数值差分**、EMA，**不新增 Topic**。

---

## 4. Ultra 顺序为准（Deploy 仅为数据源）

**准则**：**Ultra**（`ultra_env.py` 中 `find_joints` 顺序）是 **`T_leg[0..11]` 的唯一编号准则**；`leg_index_mapping.yaml` 与之对齐。  

Deploy `bodyIdMap.h` 腿中间向量顺序：**l_hip_roll, l_hip_pitch, l_hip_yaw, …**（髋 **R–P–Y**），与 Ultra 每侧 **hip_roll → hip_yaw → hip_pitch → …**（髋 **R–Y–P**）在髋部三轴上**不一致**。  

→ **`T_leg[i]` 绝不等于** Deploy 中间向量第 `i` 个槽位；必须 **按关节语义名** 将 `one.pos/speed/current/temperature` 与 `ct_scale[j]` 对齐到 Ultra 第 `i` 路后再落盘或训练。

---

## 5. 关键路径

| 路径 | 用途 |
|:-----|:-----|
| `Tienkung_thermal/docs/plan.md` | 完整计划与白名单 |
| `Tienkung_thermal/configs/leg_index_mapping.yaml` | 与 **Ultra** 同序的 `motor_names` |
| `TienKung-Lab/legged_lab/envs/ultra/ultra_env.py` | **Ultra 关节顺序权威（T_leg 唯一准则）** |
| `Deploy_Tienkung/.../RLControlNewPlugin.cpp` | Topic 与 status/IMU 读取 |
| `Deploy_Tienkung/.../bodyIdMap.h` | Deploy 腿索引与 CAN |
| `Deploy_Tienkung/rl_control_new/config/tg22_config.yaml` | `dt`, `ct_scale` |
| `TienKung_ROS/src/bodyctrl_msgs/msg/MotorStatus.msg` | **`float32 temperature`（°C）** 的权威 IDL |

---

## 6. 参考 `thermal_lstm_modeling.md` 搭建 Ultra LSTM 的调整要点

### 6.1 可复用的方法论

- 保留**因果 LSTM** 主线：输入投影 → LSTM 编码器 → 预测头。  
- 保留**滑动窗口**、**session 划分**、**Z-score**、**Huber/MAE**、**ONNX/TensorRT**、**ring buffer 在线推理**等工程流程。  
- 保留派生特征思路：`\tau^2`、`|dq|`、EMA、由 `speed` 数值差分得到的 `|ddq|`。  

### 6.2 必须替换的前提

| G1 规格书假设 | Ultra 基线应改为 |
|:--------------|:-----------------|
| 全身 **29** 关节 | 仅 **12** 条腿关节 |
| 按 `GearboxS/M/L` 分 **3** 个模型 | 先做 **Ultra 12 腿统一基线**，后续再视数据量决定是否分组 |
| 双温度通道：`temperature[0]` / `temperature[1]` | 基线仅承认 **`one.temperature` 单标量 `float32`，单位 °C** |
| 原始 `tau_est` / `ddq` / `vol` 可直接读取 | `tau_est = current * ct_scale`；`ddq` 仅能由 `speed` 数值差分；`vol` 暂不纳入 |
| DDS / SDK 低层接口 | **ROS2** `/leg/status` + 可选 `/imu/status` |
| 关节下标可直接用于训练 | 必须先映射到 **Ultra 顺序 `T_leg[0..11]`**（唯一准则） |

### 6.3 Ultra 基线输入 / 输出建议

**每关节基础输入（建议 v1）**

- `q = one.pos`
- `dq = one.speed`
- `tau_est = one.current * ct_scale[index]`
- `T = one.temperature`（**°C**）
- `tau_sq = tau_est^2`
- `dq_abs = |dq|`
- `ddq_abs = |diff(speed) / dt|`

**可选附加输入**

- 邻域温度：仅 `T_leg` 内相邻关节温度（如同侧 `i-1`, `i+1`）  
- IMU 9 标量：`euler(3) + angular_velocity(3) + linear_acceleration(3)`  

**输出**

- 单头预测未来多步 **标量温度** `y_temp`
- 验收仍以 **12 关节等权平均 MAE** 为准

### 6.4 不应直接沿用 G1 规格书的内容

- `temperature[1]` 外壳温度辅助头  
- `vol` 电压输入  
- `mainboard` / `bms` / `fan_state` / 环境温度默认入模  
- 全身 29 关节 HDF5 shape 与电机类型分桶假设  
- DDS CRC / `motorstate_` 位域这类仅在 G1 低层链路出现的处理

### 6.5 最小可行模型（建议）

- 序列长度：`L = 100`（约 5 s @ 20 Hz）
- Horizon：先对齐 `plan.md` 的 **15 s** 目标，不必保留 G1 的 20 s 远视距
- 架构：`InputProjection -> CausalLSTM -> Single PredictionHead`
- 损失：单通道 `Huber` 或 `MAE`
- 权重：训练可带 `w_0..w_11`，但 Gate 固定看等权 MAE

---

## 7. 需要调整的数据接口 / 数据契约

### 7.1 采集入口

**由 G1 风格接口切换为 Ultra 白名单接口：**

- 必需订阅：`/leg/status`
- 可选订阅：`/imu/status`
- 不把 `mainboard`、`bms`、环境传感器当作现成输入

### 7.2 关节索引与名称映射

所有训练、落盘、评估、在线推理统一使用 **Ultra 顺序 `T_leg[0..11]`**（与 `ultra_env` / `leg_index_mapping.yaml` 一致）。

最低要求：

- 采集阶段就完成 **Deploy 中间向量 → Ultra `T_leg`** 的固定语义映射（**不以位置下标硬拷贝**）
- 在元数据中写明映射表与顺序来源（**Ultra 为准**）
- 禁止下游代码假设 Deploy 下标 `j` = `T_leg` 下标 `i`

### 7.3 标签接口

G1 风格：

- `targets_coil`
- `targets_shell`

Ultra 基线：

- `targets_temp`

即：

- Dataset 返回 `(x, y_temp)` 或 `(x, y_temp, joint_idx)`
- 模型输出从双头改为单头
- Loss 从双通道联合损失改为单通道损失

### 7.4 原始特征接口

建议将原始白名单字段固定为：

- `q`
- `dq`
- `current`
- `temperature`
- `imu/*`（可选）

由此派生：

- `tau_est`
- `tau_sq`
- `dq_abs`
- `ddq_num`

### 7.5 中间数据 / HDF5 Schema

建议 Ultra 版中间格式以 **12 路腿关节** 为核心，不再复用 G1 的 `(N, 29)` 结构。

建议字段：

- `timestamps`: `(N,)`
- `joints/q`: `(N, 12)`
- `joints/dq`: `(N, 12)`
- `joints/current`: `(N, 12)` 或直接存 `tau_est`
- `joints/tau_est`: `(N, 12)`
- `joints/temperature`: `(N, 12)`
- `joints/ddq_num`: `(N, 12)`
- `imu/euler`: `(N, 3)`
- `imu/angular_velocity`: `(N, 3)`
- `imu/linear_acceleration`: `(N, 3)`
- `metadata/sample_rate`
- `metadata/t_leg_order`
- `metadata/deploy_to_t_leg_mapping`

### 7.6 Dataset 接口

Dataset 不再默认：

- 输入维度固定为 `D=8`
- 目标为双通道温度
- 按 `GearboxS/M/L` 分桶

建议改为参数化接口：

- `joint_indices`: 默认 `0..11`
- `feature_config`: 控制是否拼接邻居温度 / IMU
- `target_kind`: 当前仅 `temperature`
- `horizon_steps`: 由 Ultra 验收目标配置

### 7.7 在线推理接口

在线推理链路应改为：

- `/leg/status` 维护 `T_leg` 顺序 ring buffer
- 可选拼接 `/imu/status`
- 在线执行 EMA、数值差分、归一化
- 输出未来温度轨迹供热保护或监控模块消费

---

## 8. 实施计划（建议顺序）

### Phase A：先打通数据链路

1. ~~明确 `one.temperature` 的形态与单位~~ **已约定**：**单标量 `float32`，°C**（`TienKung_ROS` `MotorStatus.msg` + `plan.md` §0）；实机仍建议抽样对照。  
2. 实现并单测 `Deploy -> Ultra T_leg[0..11]` 映射。  
3. 落盘 Ultra 专用中间数据格式（至少含 `q/dq/current/tau_est/temperature`）。  

### Phase B：建立单温度基线模型

1. 用 `q, dq, tau_est, T, tau_sq, |dq|, |ddq|` 训练单头因果 LSTM。  
2. 以 **12 关节等权 MAE** 做主指标。  
3. 先不引入 `temperature[1]`、电压、BMS、主板温、风扇。  

### Phase C：逐步扩展特征

1. 加邻域温度，验证是否优于纯本体特征。  
2. 加 IMU 9 维，验证是否对负载切换和跑跳工况有收益。  
3. 若后续确认 `temperature` 实为多通道，再讨论双头网络。  

### Phase D：部署与闭环

1. 导出 ONNX / TensorRT。  
2. 接入在线 ring buffer 推理。  
3. 若业务需要，再接热保护阈值逻辑。  

### 当前优先级

- ~~**P0**：`one.temperature` 语义确认~~ **已确定：°C、单通道 `float32`**（见上）  
- **P0**：`Deploy → Ultra T_leg` 语义映射实现与单测（**Ultra 顺序为准**）  
- **P1**：Ultra 专用 HDF5 / Dataset 契约定稿  
- **P1**：单温度基线 LSTM 跑通  
- **P2**：邻域温度 / IMU 消融  

---

*与 `plan.md` 同步；待获取项澄清后可迁入 `plan.md` §1.3–1.4 白名单。*
