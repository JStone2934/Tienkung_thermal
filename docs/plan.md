# 天工 Ultra 人形机器人腿部（12 关节）热动力学 LSTM 建模工程实施计划书

## 0. 文档说明与项目基准

* **文档状态**: Ultra 专版；**不**再以第三方厂商 IDL（如宇树 G1）为接口依据。
* **LSTM 代码落地与配置（方案 A）**: 里程碑、目录与 `configs/ultra_thermal_lstm.yaml` 约定见 **`docs/ultra_thermal_lstm_implementation.md`**；与旧 `configs/thermal_predictor.yaml`（MLP）**分文件**，避免混用。
* **关节顺序的权威来源（唯一准则）**: **Ultra**，即 **`TienKung-Lab/legged_lab/envs/ultra/ultra_env.py`** 中 `left_leg_ids` / `right_leg_ids` 的 `find_joints` 顺序。`Tienkung_thermal/configs/leg_index_mapping.yaml` 仅与该顺序**对齐引用**，**不以 YAML 覆盖 Lab**；**禁止**以 `Deploy_Tienkung` 腿中间向量下标替代 `T_leg[i]`。
* **消息定义的权威来源**: **`Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs/msg/`** 下的 `.msg` 文件为 ROS2 接口的唯一权威定义。本文档所有字段名、类型与语义**以此为准**；若实机固件与 `.msg` 分叉，以 bag / `ros2 interface show` 核对后更新。
* **数据使用原则（硬性）**: 建模、训练与评估所用的**原始观测**来自 **`/leg/status`** 话题（消息类型 `bodyctrl_msgs/msg/MotorStatusMsg`）及可选的 **`/imu/status`**（`bodyctrl_msgs/msg/Imu`）。字段范围以 ros2ws 中 `.msg` 定义为界。  
  * **允许**：对上述已定义标量/向量做**确定性后处理**（例如对 `speed` 做时间差分得到数值角加速度、对温度做 EMA），不新增数据源。  
  * **禁止**：把未在 ros2ws `.msg` 定义中出现的量直接写入「基线特征」；此类一律列入 **§1.5 待获取备选数据**。
* **机器人型号**: **TienKung Ultra**（与 Lab 中 Ultra 资产及 `ultra_env` 一致）。
* **建模范围（硬性）**: **仅** Lab 定义的 **12** 个腿部关节（与 `leg_index_mapping.yaml` 中 `n_leg_motors: 12` 及 `motor_names` 一致）；不含上肢、腰、头等（与本专项无关）。
* **核心目标**: 在实机侧使用 Deploy 已定义的 ROS2 状态流，对 **12 路腿部** 构建因果 LSTM 热预测；工况与任务命名可与 Lab 的 `ultra_walk` / `ultra_run` 及数据集目录对齐。
* **验收标准（初稿）**: 主温度预测未来 **15 s**，**12 关节等权算术平均 MAE ≤ 1.5°C**。  
* **关节权重接口**: 配置权重 `w_0..w_11`（顺序同 `motor_names`），默认全 **1**；训练可用加权损失 \(\mathcal{L} = \sum_i w_i \mathcal{L}_i / \sum_i w_i\)；**验收与 Gate 仅以等权平均 MAE 为准**；可选可学习 \(w_i\)（归一化后），见 §6.1。  
* **运行约束**: 单次前向推理 **≤ 5 ms**（FP16，机载或 PC）。
* **电机温度 `temperature` 的类型与单位（已确定）**:  
  * **IDL**：`bodyctrl_msgs/msg/MotorStatus.msg`（ros2ws 权威路径：`Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs/msg/MotorStatus.msg`）定义为 **`float32 temperature`**，**单标量**（非数组）。  
  * **单位**：**摄氏度 (°C)**。与消息内 `pos`（rad）、`current`（A）并列时，`temperature` 在工程上按 **°C** 解释。  
  * **本专项**：监督与预测一律在 **°C** 下计算 MAE 等指标；与 `plan.md` §2.1 标量温度假设一致。
* **电机电压 `voltage` 的类型与单位（已确定）**:  
  * **IDL**：`MotorStatus.msg` 定义 **`float32 voltage`**，**单标量**。  
  * **单位**：**伏特 (V)**。实测 bag 中每电机约 **61–62 V**，为母线电压经驱动分配后的电机端电压。  
  * **本专项**：作为 LSTM 输入特征；电压降可反映负载与发热关联。

---

## 1. 系统架构、关节顺序与**准许使用的数据**

### 1.1 硬件与软件栈（两仓库角色）

| 层级 | 内容 |
|:-----|:-----|
| 仿真与策略 | **TienKung-Lab**：Isaac Sim 4.5 + Isaac Lab 2.1（见 `TienKung-Lab/README.md`），任务 `ultra_walk` / `ultra_run`，Sim2Sim `sim2sim_ultra.py` |
| 实机采集与控制 | **Deploy_Tienkung**：ROS2 Humble，`rl_control_new` 插件（见 `Deploy_Tienkung/README.md`） |
| **ROS2 消息定义（权威）** | **`Tienkung/ros2ws/install/bodyctrl_msgs/`**：全部 `.msg` / `.srv` 的唯一权威来源（16 个 ament 包；腿部核心为 `MotorStatus.msg` + `MotorStatusMsg.msg` + `MotorName.msg`） |

### 1.2 腿部关节顺序（**以 Ultra 为准，唯一准则**）

**权威实现**: `TienKung-Lab/legged_lab/envs/ultra/ultra_env.py`（`find_joints` 的 `name_keys` 顺序）。所有 **`T_leg[i]`**、特征列、损失权重 \(w_i\)、HDF5 腿维**一律**按此 Ultra 顺序编号；仿真、训练与实机落盘对齐时**均**以此为准。

每侧：**hip_roll → hip_yaw → hip_pitch → knee_pitch → ankle_pitch → ankle_roll**（髋 **R–Y–P**）。

全局 **`T_leg[0..11]`** 与 `Tienkung_thermal/configs/leg_index_mapping.yaml` 中 `motor_names` 一致：

| 下标 | 关节名 |
|:----:|:-------|
| 0–5 | `hip_roll_l_joint` … `ankle_roll_l_joint` |
| 6–11 | `hip_roll_r_joint` … `ankle_roll_r_joint` |

**Deploy 侧**（`Deploy_Tienkung/.../bodyIdMap.h`）：腿中间向量 **0–11** 为 `l_hip_roll`, `l_hip_pitch`, `l_hip_yaw`, …（髋 **R–P–Y**）。**不得**将 Deploy 下标 `j` 与 Ultra 槽位 `i` 按位置一一等同；**必须**按**关节语义名**（及 CAN id→名）将 `/leg/status` 与 `ct_scale[j]` 对齐到上表 **`T_leg[i]`** 后再进入训练或特征管线。

#### 1.2.1 CAN ID → `T_leg[i]` 完整映射表（已确认）

**来源**：`bodyctrl_msgs/msg/MotorName.msg`（左腿 51–56，右腿 61–66）+ Deploy 侧髋部 R–P–Y 排列。

| CAN ID | `MotorName` 常量 | Deploy 语义名 | `T_leg[i]` | Ultra 关节名 |
|:------:|:-----------------|:-------------|:----------:|:------------|
| 51 | `MOTOR_LEG_LEFT_1` | `l_hip_roll` | **0** | `hip_roll_l_joint` |
| 52 | `MOTOR_LEG_LEFT_2` | `l_hip_pitch` | **2** | `hip_pitch_l_joint` |
| 53 | `MOTOR_LEG_LEFT_3` | `l_hip_yaw` | **1** | `hip_yaw_l_joint` |
| 54 | `MOTOR_LEG_LEFT_4` | `l_knee` | **3** | `knee_pitch_l_joint` |
| 55 | `MOTOR_LEG_LEFT_5` | `l_ankle_pitch` | **4** | `ankle_pitch_l_joint` |
| 56 | `MOTOR_LEG_LEFT_6` | `l_ankle_roll` | **5** | `ankle_roll_l_joint` |
| 61 | `MOTOR_LEG_RIGHT_1` | `r_hip_roll` | **6** | `hip_roll_r_joint` |
| 62 | `MOTOR_LEG_RIGHT_2` | `r_hip_pitch` | **8** | `hip_pitch_r_joint` |
| 63 | `MOTOR_LEG_RIGHT_3` | `r_hip_yaw` | **7** | `hip_yaw_r_joint` |
| 64 | `MOTOR_LEG_RIGHT_4` | `r_knee` | **9** | `knee_pitch_r_joint` |
| 65 | `MOTOR_LEG_RIGHT_5` | `r_ankle_pitch` | **10** | `ankle_pitch_r_joint` |
| 66 | `MOTOR_LEG_RIGHT_6` | `r_ankle_roll` | **11** | `ankle_roll_r_joint` |

**注意**：CAN 52/53（左髋 pitch/yaw）与 CAN 62/63（右髋 pitch/yaw）在 Deploy 序号与 Ultra `T_leg` 下标之间**交叉**（Deploy 髋 R–P–Y vs Ultra 髋 R–Y–P），其余关节序号不变。采集代码**必须**按此表重排，不可逐位置拷贝。

### 1.3 `/leg/status` — 准许作为原始数据的接口（ros2ws 消息定义为准）

**数据源范式**：本专项直接消费 **`/leg/status`** 话题上 `bodyctrl_msgs/msg/MotorStatusMsg` 的**全部已定义字段**，以 `Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs/msg/` 下的 `.msg` 文件为权威接口，不再仅限于 Deploy 插件中显式读取的子集。

#### 1.3.1 ROS2 话题（本专项使用）

| Topic | 消息类型 | 用途 |
|:------|:---------|:-----|
| **`/leg/status`** | `bodyctrl_msgs/msg/MotorStatusMsg` | **主数据源**：12 路腿部电机状态（含位置、速度、电流、温度、误差、电压） |
| `/imu/status` | `bodyctrl_msgs/msg/Imu` | 可选上下文特征（姿态、角速度、线加速度） |

#### 1.3.2 `MotorStatusMsg` 结构（ros2ws 权威定义）

**`MotorStatusMsg.msg`**：

```
std_msgs/Header header
MotorStatus[] status
```

**`MotorStatus.msg`**（每个电机一条，`/leg/status` 中 `status` 数组恒含 **12** 条）：

| 字段 | 类型 | 单位 | 说明 |
|:-----|:-----|:-----|:-----|
| `name` | `uint16` | — | CAN ID；左腿 51–56，右腿 61–66（见 `MotorName.msg` 及 §1.2.1 映射表） |
| `pos` | `float32` | rad | 关节位置 |
| `speed` | `float32` | rad/s | 关节角速度 |
| `current` | `float32` | A | 电机电流；可与 `ct_scale[j]` 相乘得力矩估计 |
| `temperature` | `float32` | °C | 电机温度（**监督标签**，见 §0） |
| `error` | `uint32` | — | 电机故障码；正常时为 **0**。用作**数据质量过滤**：`error ≠ 0` 的帧标记/排除（见 §2.1.1） |
| `voltage` | `float32` | V | 电机端电压（实测约 61–62 V）；作为 **LSTM 输入特征**（见 §2.1.2） |

**原始频率**：实测 bag 中 `/leg/status` 约 **1 kHz**（时间戳间隔 ~1 ms）。本专项降采样至 **500 Hz** 后使用（见 §4）。

**配置参考**：`Deploy_Tienkung/rl_control_new/config/tg22_config.yaml` 中的 **`ct_scale`** 可用于将 `current` 转换为力矩估计 \(\tau_{est}\)（与热特征相关的为 **`dt`、`ct_scale`**）。

#### 1.3.3 `Imu` 结构（ros2ws 权威定义）

**`Imu.msg`**：

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `header` | `std_msgs/Header` | 时间戳与帧 ID |
| `orientation` | `geometry_msgs/Quaternion` | 四元数姿态 |
| `angular_velocity` | `geometry_msgs/Vector3` | 角速度 x/y/z |
| `linear_acceleration` | `geometry_msgs/Vector3` | 线加速度 x/y/z |
| `euler` | `bodyctrl_msgs/Euler` | 欧拉角 roll/pitch/yaw（`float64`） |
| `error` | `uint32` | IMU 故障码 |
| `angular_velocity_covariance` | `float64[3]` | 角速度协方差 |
| `orientation_covariance` | `float64[3]` | 姿态协方差 |
| `linear_acceleration_covariance` | `float64[3]` | 线加速度协方差 |

**基线使用的 IMU 标量**（9 维）：`euler.roll`、`euler.pitch`、`euler.yaw` + `angular_velocity.{x,y,z}` + `linear_acceleration.{x,y,z}`。

### 1.4 TienKung-Lab — 准许作为原始数据或布局参考的接口（仓库明示）

* **`TienKung-Lab/README.md`**：Isaac/Lab 版本、任务名 `ultra_walk` / `ultra_run`、数据集子目录 `motion_visualization/`、`motion_amp_expert/`、训练/播放/Sim2Sim 命令行。  
* **`legged_lab/envs/ultra/ultra_env.py`**：Ultra **腿 6+6 关节名顺序**；`visualize_motion` 文档字符串中的 **44 维**可视化帧布局（含 root、左右腿 `dof_pos`/`dof_vel` 等）；`get_amp_obs_for_expert_trans` 的 **AMP 观测拼接布局**（文档字符串）。  
* **运动数据加载**：`rsl_rl/.../motion_loader.py`、`motion_loader_for_display.py` 等对 JSON `Frames`、`FrameDuration`、`MotionWeight` 的用法（**无温度维度**）。  
* **仿真内关节状态**：环境中通过 **`self.robot.data.joint_pos` / `joint_vel`** 按 `*_leg_ids` 访问（Isaac Lab `Articulation` 数据）；**仓库内未出现电机温度读数**。

**结论（热监督）**: **温度真值**仅来自 `/leg/status` 中 `MotorStatus.temperature`（经 §1.2.1 映射到 `T_leg[i]`）；**不得**假设 Lab 仿真提供同名热标签。

### 1.5 待获取备选数据（当前基线未纳入）

以下**不作为**当前基线数据源；若后续纳入，须确认 topic 名称并在文档中升格为「准许」。

| 类别 | 说明 |
|:-----|:-----|
| 原生角加速度 `ddq` 话题 | ros2ws 中**未见**独立 `ddq` Topic；当前由 `speed` 数值差分替代（§2.1.2） |
| `MotorStatus1`（双温度通道） | `MotorStatus1.msg` 定义 `float32 motortemperature` + `float32 mostemperature`（电机绕组温 + MOS 温），通过 `MotorStatusMsg1.msg` 发布。若对应 topic 可录制，**双通道温度**可显著改善热建模。需确认实机 topic 名 |
| `PowerStatus`（板级热/电数据） | `PowerStatus.msg` 包含 `leg_a_temp` / `leg_b_temp`（腿部驱动板 MOS 温度，含 min/max）、`leg_a_curr` / `leg_b_curr`（板级电流，含 min/max）、`leg_a_volt` / `leg_b_volt`、`bus_volt`（母线电压）、`battery_voltage` / `battery_current` / `battery_power`。需确认实机 topic 名后可作为环境热参考 |
| `PowerBatteryStatus`（电池详情） | 含主电池/小电池电压、电流、电量 |
| BMS、风扇、机内环境温度 | ros2ws 中**未见**对应 `.msg` |
| 实验室环境温度、湿度计 | **非** ros2ws 接口，属外设记录 |
| Lab 内电机热仿真或 `temperature` 域 | **未在仓库出现** |
| 宇树等第三方低层 DDS 字段 | **明确排除**于本专项基线 |

---

## 2. 核心数据字段（基于 §1.3 ros2ws 消息定义）

| 逻辑量 | 准许来源（`/leg/status` 中 `MotorStatus` 字段） |
|:-------|:---------|
| \(q\) | `pos`（映射到 `T_leg` 顺序后，§1.2.1） |
| \(dq\) | `speed` |
| \(I\) | `current` |
| \(\tau_{est}\) | `current × ct_scale[j]`（`ct_scale` 来自 `tg22_config.yaml`） |
| \(T\) | `temperature`（**监督标签**） |
| \(V\) | `voltage`（电机端电压，输入特征） |
| `error` | `error`（数据质量过滤，非输入特征） |
| IMU 上下文 | `/imu/status` 中 §1.3.3 所列 9 个标量（可选特征） |
| 工况/轨迹参考 | Lab 数据集与 `dof_pos`/`dof_vel` 布局（**无温度**，仅作实验设计或离线对齐） |

**温度通道与单位**: 基线为 **`temperature` 单一 `float32`，单位 °C**（见 §0）。若将来消息扩展为多元素（如 `MotorStatus1` 的双温度通道，见 §1.5），再评估双头网络；当前构建 LSTM 的准许数据见 **§2.1**。

### 2.1 假设 `temperature` 为单一标量（°C）时的 LSTM 数据清单

以下严格对应 **§1.3–§1.4 白名单**与 **§0** 允许的确定性后处理（不新增 Topic）。

#### 2.1.1 监督标签（预测目标）与数据质量过滤

* **每关节、每时刻**：将 `/leg/status` 中每条 `status` 按 §1.2.1 CAN ID 映射到 **`T_leg[0..11]`** 后，取 **`temperature`（标量，°C）** 作为温度真值。  
* **多步预测**：在同一时间序列上对 \(T\)（°C）构造滑动窗口，用未来 \(h\) 步的标量温度（°C）作监督，**不引入**其它温度通道或仿真热标签。  
* **error 过滤（硬性）**：若某帧中任一腿部电机的 `error ≠ 0`，该帧**标记为异常**并从训练/评估窗口中排除（或截断该 session 片段）。`error` 字段本身**不**作为 LSTM 输入特征，仅用于数据清洗。

#### 2.1.2 每关节输入特征（针对每个 `T_leg[i]`，\(i=0..11\)）

| 特征（逻辑名） | 准许来源 | 说明 |
|:---------------|:---------|:-----|
| \(q\) | `pos`（经 §1.2.1 映射到 \(i\)） | 位置 (rad) |
| \(dq\) | `speed`（映射到 \(i\)） | 角速度 (rad/s) |
| \(I\) | `current`（映射到 \(i\)） | 电机电流 (A)；原始值，未乘 `ct_scale` |
| \(\tau_{est}\) | `current × ct_scale[j]` | \(j\) 为 Deploy 中间向量下标，经 §1.2.1 映射表与 \(i\) 对应；`ct_scale` 来自 `tg22_config.yaml` |
| \(T\) | `temperature`（标量 **°C**，映射到 \(i\)） | 当前温；可做 EMA 等平滑 |
| \(V\) | `voltage`（映射到 \(i\)） | 电机端电压 (V)；电压降反映负载，与热产生关联；释义见 **§2.1.2.3** |
| \(\tau_{sq}\) | \(\tau_{est}^2\) | 焦耳热代理（由 \(\tau_{est}\) 平方导出）；释义见 **§2.1.2.1** |
| \(\|dq\|\) | \(\|dq\|\) | 摩擦 / 机械损耗相关代理（由 `speed` 取绝对值）；释义见 **§2.1.2.2** |
| \(\|ddq\|\)（数值） | 对已落盘的 `speed` 按时间差分 | **仅允许**此类后处理；原生 `ddq` Topic 属 **§1.5** |

#### 2.1.2.1 估计力矩 \(\tau_{est}\) 与焦耳热代理 \(\tau_{sq}\)

* **\(\tau_{est}\) 的来源（Deploy 插件一致）**  
  对每个电机状态：\(\tau_{est} = \texttt{one.current} \times \texttt{ct\_scale}[j]\)。其中 `ct_scale` 来自 `tg22_config.yaml`，下标 \(j\) 为 Deploy 中间向量索引，与 `T_leg[i]` 之间用固定映射表对齐。**不是**力矩传感器直读值，而是**电流 × 标定系数**得到的估计。

* **\(\tau_{sq} = \tau_{est}^2\) 如何得到**  
  在同一时间步、同一关节上，对 \(\tau_{est}\) 做标量平方即可：\(\tau_{sq} = (\tau_{est})^2\)。实现上等价于 `tau_est * tau_est`，**不新增**任何 Topic 或字段，仅属 §0 允许的确定性后处理。

* **为何称为「焦耳热代理」**  
  在简化电机模型中，绕组铜耗常与 **电流平方** \(I^2\) 成正比（焦耳定律）。若在该工作区可近似认为 \(\tau_{est} \propto I\)，则 \(\tau_{est}^2\) 与 \(I^2\) **同阶**，可作为**电气铜耗主导发热**的粗代理，便于网络学习「负载大 → 温升快」的趋势。  
  **注意**：这是**工程近似**而非严格热路模型；实际还存在铁耗、逆变器损耗、传动效率等，\(\tau_{sq}\) **不能**等同真实焦耳功率，故文档中统一称**代理**。

* **与 \(\|dq\|\) 的分工（直觉）**  
  \(\tau_{sq}\) 偏**电磁/负载电流侧**损耗线索；\(\|dq\|\) 偏**与运动学相关的机械损耗**线索，二者互补而非重复。

#### 2.1.2.2 角速度幅值 \(\|dq\|\)（摩擦 / 机械损耗相关代理）

* **如何得到**  
  \(dq\) 即该关节的 `one.speed`（映射到 `T_leg[i]`）；\(\|dq\| = |dq|\)，对同一标量取绝对值即可，**不新增**数据源。

* **为何与「摩擦」联系起来**  
  轴承、密封、润滑等引起的摩擦损耗，其功率往往随**相对运动**增强；在**粘性摩擦**等模型中，损耗还与 \(|\omega|\) 或 \(\omega^2\) 相关。因此**转速越大**，摩擦类机械损耗通常越显著。用 **\(|dq|\)** 刻画「转得多快」，与这类损耗在统计上**相关**。

* **为何用绝对值**  
  摩擦产热通常与**转向无关**，只与转速大小有关；取 **\(|\cdot|\)** 避免符号在特征中引入无意义翻转，并与「幅值越大 → 摩擦相关损耗往往越大」的直觉一致。

* **为何仍称「代理」而非「摩擦热」**  
  本栈**没有**直接测量摩擦扭矩或摩擦功率；\(|dq|\) 只是由速度导出的标量。高速工况下电流、负载也往往同时变化，故该特征会与**多种机理混杂**。文档中称**摩擦相关代理**（亦可理解为**与速度相关的机械损耗代理**），表示**启发式特征**，**不是**严格力学意义上的摩擦功率。

#### 2.1.2.3 电机端电压 \(V\)（负载/供电状态特征）

* **如何得到**  
  直接取 `/leg/status` 中同一电机的 `voltage` 字段（经 §1.2.1 映射到 `T_leg[i]`）；**不新增** Topic。

* **物理意义与热建模关联**  
  `voltage` 反映电机端实际供电电压。负载增加时母线电压可出现轻微下降（压降效应），同时驱动器 MOS 管与电机绕组的电功率与 \(V \times I\) 直接相关。因此电压信号可为网络提供：  
  1. **电功率代理**（与 `current` 联合刻画实际电功率 \(P = V \times I\)）；  
  2. **供电状态上下文**（母线电压水平影响驱动效率与散热条件）。

* **与 \(\tau_{sq}\) 的分工**  
  \(\tau_{sq}\) 基于电流侧估计力矩的平方（偏负载端），\(V\) 来自电压端实测值；二者从**电功率等式两侧**互补。

**邻域耦合（可选）**：仅使用 **`T_leg` 内**相邻关节的标量温度（如 \(T_{i-1}, T_{i+1}\) 在同侧且存在时），**不**拼接臂、腰等未建模自由度。

#### 2.1.3 全局 / 上下文特征（可选）

来自 **`/imu/status`** 的 **9 个标量**（见 §1.3.3）：

* `euler.yaw`, `euler.pitch`, `euler.roll`  
* `angular_velocity.x`, `angular_velocity.y`, `angular_velocity.z`  
* `linear_acceleration.x`, `linear_acceleration.y`, `linear_acceleration.z`  

可对上述量做模长、帧间差分等确定性变换，**仍属白名单内派生**。

#### 2.1.4 不得作为本基线 LSTM 输入的（见 §1.5）

* 原生 `ddq` 话题、BMS、主板温、风扇、外置温湿度计（未入库前）。  
* `MotorStatus1` 双温度通道、`PowerStatus` 板级数据：虽有对应 `.msg` 定义，**topic 名未确认**前不纳入基线。  
* **TienKung-Lab 仿真**：虽有 `joint_pos` / `joint_vel` 等，**无**与实机一致的标量 `temperature`，**不能**作为本热 LSTM 的**温度监督**；仅可用于工况设计、轨迹参考或与 Deploy 日志**时间对齐**的辅助记录（若业务需要），**不替代** §2.1.1 的标签来源。

#### 2.1.5 张量组织（与 §4、§6 衔接）

* **输入序列**：在 **500 Hz**（从原始 ~1 kHz 降采样）的统一网格上，对每个关节拼接 §2.1.2（及可选 §2.1.3）特征，得到 `[B, L, D]`。  
* **输出**：对每个关节预测未来 \(H\) 个时间步的 **标量温度（°C）**；12 关节可 **12 路独立头** 或 **批量维 `×12`**，由实现选定；损失与验收仍遵循 **§0** 等权 MAE 与 **§6.1** 权重约定。

---

## 3. 腿部 12 关节与工况备注

顺序同 §1.2（与 Lab `ultra_env` 一致）。热工况优先级（膝、髋 pitch 等）见前文各表，不重复。

---

## 4. 特征工程与状态空间（在准许数据内）

**标量温度基线下的具体特征枚举与监督定义见 §2.1**；本节为工程约定摘要。

* **500 Hz** 为工程目标频率（从原始 ~1 kHz 降采样，见 §1.3.2）。  
* **力矩/速度/温度/电压** 变换（\(\tau^2\)、\(|dq|\)、EMA 等）仅作用于 §2、§2.1 所列准许量。  
* **\(ddq\)**：无原生字段时，**仅允许**由已记录的 `speed` 序列数值差分得到；若将来有驱动器原生 `ddq` Topic，先移入 §1.3 再使用。  
* **邻域耦合**: 仅 **\(T_leg\)** 相邻索引（**Ultra** 顺序下），见 §2.1.2。  
* **全局特征**: 在基线中**仅**可使用 §1.3.3 的 IMU 9 维（§2.1.3）；PowerStatus 等属 §1.5。  
* **张量形状**: `[B, L, D]`，`L = 2500`（约 5 s @ 500 Hz）；输出为多步标量温度，见 §2.1.5。与 `thermal_lstm_modeling.md` 对齐前须同步修订该文档。  
* **error 过滤**：`error ≠ 0` 的帧排除，见 §2.1.1。

---

## 5. 数据集采集协议

* **录制**：至少 `/leg/status`（及若使用 IMU 上下文则 `/imu/status`）；与插件一致的字段。  
* **ROS 2 `ros2 bag record` 示例**（实机已 `source` 工作空间、topic 在播；输出目录按需修改）：  
  * 仅腿部状态（含每关节 `temperature`）：

```bash
ros2 bag record -o ~/bags/thermal_leg_$(date +%Y%m%d_%H%M%S) /leg/status
```

  * 腿部 + IMU（与 §2.1.3 可选上下文一致）：

```bash
ros2 bag record -o ~/bags/thermal_leg_imu_$(date +%Y%m%d_%H%M%S) /leg/status /imu/status
```

* **映射**：Deploy `index` / `bodyIdMap` 名称 → **Ultra** 顺序的 `T_leg[0..11]`（固定语义映射表，写入采集代码注释；**Ultra 为准**）。  
* **工况矩阵、时长、冷却段**：同前版 §5.2–5.3；**室温**属 §1.5 外设记录，可保留为实验笔记，**不**声称来自两仓库接口。  
* **实操流程与 session 设计**：分步检查清单、`ros2 bag record` 用法、建议工况类型与 Train/Val/Test 衔接见 **`docs/recording_operations.md`**。

---

## 6. LSTM 网络与训练规范

* **输入 / 标签** 以 **§2.1**（标量 `temperature` 假设）为准；架构为**因果 LSTM**（禁用 BiLSTM）。  
* 损失按关节 Huber/MAE + §0 权重约定。  
* **§6.1 权重**（同前）：静态 `w_i`、等权 MAE 门槛、加权训练、可选可学习 \(w_i\)。

---

## 7. 工程部署与整合

* ONNX → TensorRT；在线：仅消费 `/leg/status`（§1.3）及可选 `/imu/status` 的已定义字段。  
* 与 Lab 的协同：任务名、数据目录、Sim2Sim 流程以 `TienKung-Lab/README.md` 为准。

---

## 8. 风险与待办

* **P0**: CAN ID → `T_leg` 映射已确认（§1.2.1）；**温度单位 °C** 已约定（§0）。需实机验证：`error` 字段在故障工况下的实际取值范围与含义。  
* **P0**: 500 Hz 降采样实现——确认 `/leg/status` 原始频率确实 ~1 kHz，且时间戳单调递增无跳变。  
* **P1**: `MotorStatus1`（`motortemperature` + `mostemperature` 双通道）与 `PowerStatus`（板级热数据）的 topic 名确认（§1.5）；确认后可升格为准许数据源。  
* **P2**: 是否仅在 §1.3 范围内扩展特征；任何新 Topic 须先确认并写入 §1.3。

### 8.1 仍待您确认（与 §1.5 区分）

1. RL 闭环是否接入热预测及软/硬阈值数值。  
2. `thermal_lstm_modeling.md`：文首与 §1、§2、§3 已与 Ultra+**°C**+单通道对齐；后续章节若仍含 G1 字样，以本文 **`plan.md`** 为准。  
3. `L = 2500`（5 s @ 500 Hz）窗口长度是否合适，或需调整窗口时长。

---

## 9. 实验阶段与门控（简表）

| Phase | 内容 | Gate |
|:------|:-----|:-----|
| 0 | `/leg/status` 可读、CAN ID → `T_leg` 映射正确（§1.2.1）、`error` 过滤逻辑 | 数据链路 OK |
| 1 | 实机与 ros2ws `MotorStatus.msg` 定义一致；500 Hz 降采样验证；`voltage` 信号合理性 | 标签链路与 °C 标定可信 |
| 2–5 | 采集 / 训练 / 离线 MAE / 在线延迟 | 等权 MAE 与 ≤5 ms |

---

## 附录 A: 与旧版（G1）文档的关系

* G1、`unitree_sdk2`、全身 29DOF **不适用于**本专项。  
* 可复用**思想**（EMA、因果 LSTM、冷却段、Huber、session 划分、TensorRT），但**数据源以本文 §1 为准**。

---

*文档结束 — 关节顺序以 **Ultra**（`ultra_env`）为准；消息定义以 **ros2ws** `bodyctrl_msgs` 为权威；原始数据以 `/leg/status` 话题全字段为基线。*
