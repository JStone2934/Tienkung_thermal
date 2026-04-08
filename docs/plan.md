# 天工 Ultra 人形机器人腿部（12 关节）热动力学 LSTM 建模工程实施计划书

## 0. 文档说明与项目基准

* **文档状态**: Ultra 专版；**不**再以第三方厂商 IDL（如宇树 G1）为接口依据。
* **LSTM 代码落地与配置（方案 A）**: 里程碑、目录与 `configs/ultra_thermal_lstm.yaml` 约定见 **`docs/ultra_thermal_lstm_implementation.md`**；与旧 `configs/thermal_predictor.yaml`（MLP）**分文件**，避免混用。
* **关节顺序的权威来源（唯一准则）**: **Ultra**，即 **`TienKung-Lab/legged_lab/envs/ultra/ultra_env.py`** 中 `left_leg_ids` / `right_leg_ids` 的 `find_joints` 顺序。`Tienkung_thermal/configs/leg_index_mapping.yaml` 仅与该顺序**对齐引用**，**不以 YAML 覆盖 Lab**；**禁止**以 `Deploy_Tienkung` 腿中间向量下标替代 `T_leg[i]`。
* **数据使用原则（硬性）**: 建模、训练与评估所用的**原始观测**只能来自 **`Deploy_Tienkung`** 与 **`TienKung-Lab`** 两仓库中**已写明**的接口（源码、配置、README、脚本与注释中的数据布局）。**不**将两仓库未定义的 Topic/字段/消息成员当作既定事实。  
  * **允许**：对上述已定义标量/向量做**确定性后处理**（例如对插件已订阅的 `speed` 做时间差分得到数值角加速度、对温度做 EMA），不新增数据源。  
  * **禁止**：把仅出现在其它文档或口头约定、且未在上述两仓库出现的量，直接写入「基线特征」；此类一律列入 **§1.5 待获取备选数据**。
* **机器人型号**: **TienKung Ultra**（与 Lab 中 Ultra 资产及 `ultra_env` 一致）。
* **建模范围（硬性）**: **仅** Lab 定义的 **12** 个腿部关节（与 `leg_index_mapping.yaml` 中 `n_leg_motors: 12` 及 `motor_names` 一致）；不含上肢、腰、头等（与本专项无关）。
* **核心目标**: 在实机侧使用 Deploy 已定义的 ROS2 状态流，对 **12 路腿部** 构建因果 LSTM 热预测；工况与任务命名可与 Lab 的 `ultra_walk` / `ultra_run` 及数据集目录对齐。
* **验收标准（初稿）**: 主温度预测未来 **15 s**，**12 关节等权算术平均 MAE ≤ 1.5°C**。  
* **关节权重接口**: 配置权重 `w_0..w_11`（顺序同 `motor_names`），默认全 **1**；训练可用加权损失 \(\mathcal{L} = \sum_i w_i \mathcal{L}_i / \sum_i w_i\)；**验收与 Gate 仅以等权平均 MAE 为准**；可选可学习 \(w_i\)（归一化后），见 §6.1。  
* **运行约束**: 单次前向推理 **≤ 5 ms**（FP16，机载或 PC）。
* **电机温度 `temperature` 的类型与单位（已确定）**:  
  * **IDL**：`bodyctrl_msgs/msg/MotorStatus.msg`（见 **`TienKung_ROS`** 仓库 `src/bodyctrl_msgs/msg/MotorStatus.msg`）定义为 **`float32 temperature`**，**单标量**（非数组）。  
  * **单位**：**摄氏度 (°C)**。与消息内 `pos`（rad）、`current`（A）并列时，`temperature` 在工程上按 **°C** 解释；底层总线字节在 `TienKung_ROS` 的 `MotorDevice.cpp` 中解码为 **`temperature = (msg.data[6] - 50) / 2`**，结果仍为浮点 **°C** 尺度。  
  * **本专项**：监督与预测一律在 **°C** 下计算 MAE 等指标；与 `plan.md` §2.1 标量温度假设一致。

---

## 1. 系统架构、关节顺序与**准许使用的数据**

### 1.1 硬件与软件栈（两仓库角色）

| 层级 | 内容 |
|:-----|:-----|
| 仿真与策略 | **TienKung-Lab**：Isaac Sim 4.5 + Isaac Lab 2.1（见 `TienKung-Lab/README.md`），任务 `ultra_walk` / `ultra_run`，Sim2Sim `sim2sim_ultra.py` |
| 实机采集与控制 | **Deploy_Tienkung**：ROS2 Humble，`rl_control_new` 插件（见 `Deploy_Tienkung/README.md`） |

### 1.2 腿部关节顺序（**以 Ultra 为准，唯一准则**）

**权威实现**: `TienKung-Lab/legged_lab/envs/ultra/ultra_env.py`（`find_joints` 的 `name_keys` 顺序）。所有 **`T_leg[i]`**、特征列、损失权重 \(w_i\)、HDF5 腿维**一律**按此 Ultra 顺序编号；仿真、训练与实机落盘对齐时**均**以此为准。

每侧：**hip_roll → hip_yaw → hip_pitch → knee_pitch → ankle_pitch → ankle_roll**（髋 **R–Y–P**）。

全局 **`T_leg[0..11]`** 与 `Tienkung_thermal/configs/leg_index_mapping.yaml` 中 `motor_names` 一致：

| 下标 | 关节名 |
|:----:|:-------|
| 0–5 | `hip_roll_l_joint` … `ankle_roll_l_joint` |
| 6–11 | `hip_roll_r_joint` … `ankle_roll_r_joint` |

**Deploy 侧**（`Deploy_Tienkung/.../bodyIdMap.h`）：腿中间向量 **0–11** 为 `l_hip_roll`, `l_hip_pitch`, `l_hip_yaw`, …（髋 **R–P–Y**）。**不得**将 Deploy 下标 `j` 与 Ultra 槽位 `i` 按位置一一等同；**必须**按**关节语义名**（及 CAN id→名）将 `/leg/status` 与 `ct_scale[j]` 对齐到上表 **`T_leg[i]`** 后再进入训练或特征管线。

### 1.3 Deploy_Tienkung — 准许作为原始数据的接口（源码明示）

以下均出自 `rl_control_new/.../RLControlNewPlugin.cpp` 及其包含的配置路径，**以外不再扩展**。

**ROS2 Topic（`onInit()` 硬编码）**

| Topic | 方向 | 消息类型（头文件） |
|:------|:-----|:-------------------|
| `/leg/status` | Sub | `bodyctrl_msgs::msg::MotorStatusMsg` |
| `/arm/status` | Sub | 同上 |
| `/imu/status` | Sub | `bodyctrl_msgs::msg::Imu` |
| `/sbus_data` | Sub | `sensor_msgs::msg::Joy` |
| `/leg/cmd_ctrl` | Pub | `bodyctrl_msgs::msg::CmdMotorCtrl` |
| `/arm/cmd_ctrl` | Pub | 同上 |
| `/waist/cmd_pos` | Pub | `bodyctrl_msgs::msg::CmdSetMotorPosition` |

**`MotorStatusMsg` → 插件对每条 `status` 元素 `one` 的读取（热建模相关）**

* `one.pos` → 位置  
* `one.speed` → 角速度  
* `one.current` → 与配置 `ct_scale` 相乘得力矩估计（见 `tg22_config.yaml` 中 `ct_scale`）  
* `one.temperature` → **电机温度，单精度浮点，单位 °C**（与 `MotorStatus.msg` 及 **TienKung_ROS** 解码一致；插件赋给 `temperature_midVec`）  
* `one.name` → 传入 `idMap.getIndexById(...)` 以得到 `index`（`bodyIdMap.h` 中 `getIndexById(int canId)`）

**配置 `Deploy_Tienkung/rl_control_new/config/tg22_config.yaml`（插件 `LoadConfig` 使用）**

* `motor_num`, `actions_size`, `dt`, `ct_scale`, `joint_kp_p`, `joint_kd_p`, `simulation` 等键（与热特征相关的为 **`dt`、`ct_scale`**）。

**`bodyctrl_msgs::Imu`（插件对 `/imu/status` 的读取）**

* `euler.yaw`, `euler.pitch`, `euler.roll`  
* `angular_velocity.{x,y,z}`  
* `linear_acceleration.{x,y,z}`  

**说明**: `MotorStatus.msg` 全文可在 **`TienKung_ROS`** 仓库查阅。`temperature` 为 **`float32` 单标量**；**单位已约定为 °C**（见 §0）。若实机消息与 **TienKung_ROS** 定义不一致，以实机 `ros2 interface show` 与 bag 为准并更新文档。

### 1.4 TienKung-Lab — 准许作为原始数据或布局参考的接口（仓库明示）

* **`TienKung-Lab/README.md`**：Isaac/Lab 版本、任务名 `ultra_walk` / `ultra_run`、数据集子目录 `motion_visualization/`、`motion_amp_expert/`、训练/播放/Sim2Sim 命令行。  
* **`legged_lab/envs/ultra/ultra_env.py`**：Ultra **腿 6+6 关节名顺序**；`visualize_motion` 文档字符串中的 **44 维**可视化帧布局（含 root、左右腿 `dof_pos`/`dof_vel` 等）；`get_amp_obs_for_expert_trans` 的 **AMP 观测拼接布局**（文档字符串）。  
* **运动数据加载**：`rsl_rl/.../motion_loader.py`、`motion_loader_for_display.py` 等对 JSON `Frames`、`FrameDuration`、`MotionWeight` 的用法（**无温度维度**）。  
* **仿真内关节状态**：环境中通过 **`self.robot.data.joint_pos` / `joint_vel`** 按 `*_leg_ids` 访问（Isaac Lab `Articulation` 数据）；**仓库内未出现电机温度读数**。

**结论（热监督）**: **温度真值**在两仓库明示范围内**仅**能来自 Deploy 的 **`one.temperature`**（经腿状态 Topic）；**不得**假设 Lab 仿真提供同名热标签。

### 1.5 待获取备选数据（两仓库未定义或未接入）

以下**不作为**当前基线数据源；若后续纳入，须在**两仓库之一**增加明确接口（或独立文档经项目批准）后再升格为「准许」。

| 类别 | 说明 |
|:-----|:-----|
| `MotorStatusMsg` 与实机一致性 | **TienKung_ROS** 已给出 `MotorStatus.msg`：`temperature` 为 **float32、单通道、°C**；若实机固件与仓库分叉，以 bag / `interface show` 核对 |
| 电压、原生角加速度 `ddq`、故障/CRC/状态字 | Deploy 插件当前**未读**；若存在于其它 Topic，需单独列出并入库定义 |
| BMS、主板温、风扇、机内环境温度 | 插件**未订阅** |
| 实验室环境温度、湿度计 | **非**两仓库接口，属外设记录 |
| Lab 内电机热仿真或 `temperature` 域 | **未在仓库出现** |
| 宇树等第三方低层 DDS 字段 | **明确排除**于本专项基线 |

---

## 2. 核心数据字段（仅基于 §1.3–1.4）

| 逻辑量 | 准许来源 |
|:-------|:---------|
| \(q\) | Deploy `one.pos`（映射到 `T_leg` 顺序后） |
| \(dq\) | Deploy `one.speed` |
| \(\tau_{est}\) | Deploy `one.current * ct_scale[index]` |
| \(T\) | Deploy `one.temperature` |
| IMU 上下文 | Deploy §1.3 所列 9 个标量分量（可选特征） |
| 工况/轨迹参考 | Lab 数据集与 `dof_pos`/`dof_vel` 布局（**无温度**，仅作实验设计或离线对齐） |

**温度通道与单位**: 基线为 **`one.temperature` 单一 `float32`，单位 °C**（见 §0、**TienKung_ROS** `MotorStatus.msg`）。若将来消息扩展为多元素，再评估双头网络；当前构建 LSTM 的准许数据见 **§2.1**。

### 2.1 假设 `temperature` 为单一标量（°C）时的 LSTM 数据清单

以下严格对应 **§1.3–§1.4 白名单**与 **§0** 允许的确定性后处理（不新增 Topic）。

#### 2.1.1 监督标签（预测目标）

* **每关节、每时刻**：将 `/leg/status` 中每条 `status` 经 **Deploy → `T_leg[0..11]`** 映射后，取 **`one.temperature`（标量，°C）** 作为温度真值。  
* **多步预测**：在同一时间序列上对 \(T\)（°C）构造滑动窗口，用未来 \(h\) 步的标量温度（°C）作监督，**不引入**其它温度通道或仿真热标签。

#### 2.1.2 每关节输入特征（针对每个 `T_leg[i]`，\(i=0..11\)）

| 特征（逻辑名） | 准许来源 | 说明 |
|:---------------|:---------|:-----|
| \(q\) | `one.pos`（映射到 \(i\)） | 位置 |
| \(dq\) | `one.speed`（映射到 \(i\)） | 角速度 |
| \(\tau_{est}\) | `one.current × ct_scale[j]` | \(j\) 为 Deploy 中间向量下标，需与 \(i\) 经固定映射表对应；`ct_scale` 来自 `tg22_config.yaml` |
| \(T\) | `one.temperature`（标量 **°C**，映射到 \(i\)） | 当前温；可做 EMA 等平滑 |
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

**邻域耦合（可选）**：仅使用 **`T_leg` 内**相邻关节的标量温度（如 \(T_{i-1}, T_{i+1}\) 在同侧且存在时），**不**拼接臂、腰等未建模自由度。

#### 2.1.3 全局 / 上下文特征（可选）

来自 **`/imu/status`**、且插件已读的 **9 个标量**（见 §1.3）：

* `euler.yaw`, `euler.pitch`, `euler.roll`  
* `angular_velocity.x`, `angular_velocity.y`, `angular_velocity.z`  
* `linear_acceleration.x`, `linear_acceleration.y`, `linear_acceleration.z`  

可对上述量做模长、帧间差分等确定性变换，**仍属白名单内派生**。

#### 2.1.4 不得作为本基线 LSTM 输入的（见 §1.5）

* 电压、原生 `ddq` 话题、故障/CRC/状态字、BMS、主板温、风扇、外置温湿度计（未入库前）。  
* **TienKung-Lab 仿真**：虽有 `joint_pos` / `joint_vel` 等，**无**与实机一致的标量 `temperature`，**不能**作为本热 LSTM 的**温度监督**；仅可用于工况设计、轨迹参考或与 Deploy 日志**时间对齐**的辅助记录（若业务需要），**不替代** §2.1.1 的标签来源。

#### 2.1.5 张量组织（与 §4、§6 衔接）

* **输入序列**：在 **20 Hz**（或自 `dt` 降采样后的统一网格）上，对每个关节拼接 §2.1.2（及可选 §2.1.3）特征，得到 `[B, L, D]`。  
* **输出**：对每个关节预测未来 \(H\) 个时间步的 **标量温度（°C）**；12 关节可 **12 路独立头** 或 **批量维 `×12`**，由实现选定；损失与验收仍遵循 **§0** 等权 MAE 与 **§6.1** 权重约定。

---

## 3. 腿部 12 关节与工况备注

顺序同 §1.2（与 Lab `ultra_env` 一致）。热工况优先级（膝、髋 pitch 等）见前文各表，不重复。

---

## 4. 特征工程与状态空间（在准许数据内）

**标量温度基线下的具体特征枚举与监督定义见 §2.1**；本节为工程约定摘要。

* **20 Hz** 落盘为工程建议（可由允许的 `dt` 序列降采样实现）。  
* **力矩/速度/温度** 变换（\(\tau^2\)、\(|dq|\)、EMA 等）仅作用于 §2、§2.1 所列准许量。  
* **\(ddq\)**：无原生字段时，**仅允许**由已记录的 `speed` 序列数值差分得到；若将来有驱动器原生 `ddq` Topic，先移入 §1.3 再使用。  
* **邻域耦合**: 仅 **\(T_leg\)** 相邻索引（**Ultra** 顺序下），见 §2.1.2。  
* **全局特征**: 在基线中**仅**可使用 §1.3 的 IMU 9 维（§2.1.3）；BMS/主板等属 §1.5。  
* **张量形状**: `[B, L, D]`，`L = 100`（约 5 s @ 20 Hz）；输出为多步标量温度，见 §2.1.5。与 `thermal_lstm_modeling.md` 对齐前须同步修订该文档。

---

## 5. 数据集采集协议

* **录制**：至少 `/leg/status`（及若使用 IMU 上下文则 `/imu/status`）；与插件一致的字段。  
* **映射**：Deploy `index` / `bodyIdMap` 名称 → **Ultra** 顺序的 `T_leg[0..11]`（固定语义映射表，写入采集代码注释；**Ultra 为准**）。  
* **工况矩阵、时长、冷却段**：同前版 §5.2–5.3；**室温**属 §1.5 外设记录，可保留为实验笔记，**不**声称来自两仓库接口。

---

## 6. LSTM 网络与训练规范

* **输入 / 标签** 以 **§2.1**（标量 `temperature` 假设）为准；架构为**因果 LSTM**（禁用 BiLSTM）。  
* 损失按关节 Huber/MAE + §0 权重约定。  
* **§6.1 权重**（同前）：静态 `w_i`、等权 MAE 门槛、加权训练、可选可学习 \(w_i\)。

---

## 7. 工程部署与整合

* ONNX → TensorRT；在线：仅消费 §1.3 已存在 Topic/字段。  
* 与 Lab 的协同：任务名、数据目录、Sim2Sim 流程以 `TienKung-Lab/README.md` 为准。

---

## 8. 风险与待办

* **P0**: `T_leg` 与 Deploy 腿向量的**名称映射**实现与单测；**温度单位 °C** 已约定（§0），映射与 `name` 字段语义仍须实机核对。  
* **P1**: 是否仅在 §1.3–1.4 范围内扩展特征；任何新 Topic 须先入库再写入 §1.3。

### 8.1 仍待您确认（与 §1.5 区分）

1. 监督是否**仅** Deploy 实机日志（推荐，与 §1.4 结论一致）。  
2. RL 闭环是否接入热预测及软/硬阈值数值。  
3. `thermal_lstm_modeling.md`：文首与 §1、§2、§3 已与 Ultra+**°C**+单通道对齐；后续章节若仍含 G1 字样，以本文 **`plan.md`** 为准。

---

## 9. 实验阶段与门控（简表）

| Phase | 内容 | Gate |
|:------|:-----|:-----|
| 0 | 仅准许 Topic 可读、`T_leg` 映射正确 | 数据链路 OK |
| 1 | 实机与 **TienKung_ROS** `MotorStatus` 定义一致；映射与 `name` 语义 | 标签链路与 °C 标定可信 |
| 2–5 | 采集 / 训练 / 离线 MAE / 在线延迟 | 等权 MAE 与 ≤5 ms |

---

## 附录 A: 与旧版（G1）文档的关系

* G1、`unitree_sdk2`、全身 29DOF **不适用于**本专项。  
* 可复用**思想**（EMA、因果 LSTM、冷却段、Huber、session 划分、TensorRT），但**数据源以本文 §1 为准**。

---

*文档结束 — 关节顺序以 **Ultra**（`ultra_env`）为准；原始数据以 Deploy_Tienkung 与 TienKung-Lab 明示接口为界。*
