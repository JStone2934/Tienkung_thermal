# Ultra 热 LSTM — 待实机或另行确认项（详细版）

## 文档目的

本文件列出：**在把采集、训练、在线推理写进代码之前**，必须通过实机、`ros2 interface show`、Deploy 源码或日志**核实**的参数与接口。

- **原则**：与 `plan.md` §0–§1.5、`task_memory.md` §3 一致——两仓库未写明的字段不得默认存在。电机温度 **`MotorStatus.temperature`** 的类型与 **°C** 单位已在 **`TienKung_ROS`** `MotorStatus.msg` 与 `plan.md` §0 **确定**；实机仍以 bag / `interface show` 做一致性抽查。
- **用法**：自上而下按 **P0 → P1 → P2** 勾选；P0 未完成前，不建议大规模录数或锁死 HDF5 schema。
- **记录建议**：每勾一项，在团队 wiki 或 PR 里补一行「结论 + 证据」（接口输出截图、bag 统计、`interface show` 粘贴）。

---

## 优先级说明

| 级别 | 若不确认会怎样 | 典型动作 |
|:-----|:---------------|:---------|
| **P0** | 关节错位、标签错关节、力矩代理错、时间基准错 → 数据**整体不可信** | 先做完再批量录数 |
| **P1** | 特征物理意义漂移、IMU 与腿不同步、在线与离线分布不一致 | 特征定型前完成 |
| **P2** | 主要影响**效果与工程取舍**，可迭代 | 可与基线训练并行 |

---

## P0 — 阻塞数据采集与标签正确性

### P0 说明

P0 解决三件事：**（1）消息里到底有什么字段、什么类型**；**（2）第 `i` 个训练槽位是否真是 `T_leg[i]` 那条腿**；**（3）`current × ct_scale` 是否真是可用的力矩代理、时间步长怎么取**。任何一条错了，LSTM 学的是噪声或错标签。

### P0 两仓库静态可答项（`Deploy_Tienkung` / `TienKung-Lab`）

以下仅通过**阅读两仓库源码与配置**可归纳的结论，用于勾 P0 时对照；**不能**替代实机 `ros2 interface show`、bag 统计与物理对照。路径均相对各仓库根目录。

**`Deploy_Tienkung`**

| P0 TODO 主题 | 静态可答内容 | 证据位置 |
|:-------------|:-------------|:---------|
| 话题与消息类型 | 订阅 `/leg/status`，类型为 `bodyctrl_msgs::msg::MotorStatusMsg` | `rl_control_new/src/plugins/rl_control_new/src/RLControlNewPlugin.cpp`（`onInit` 中 `create_subscription`） |
| 每条 `status` 读哪些字段 | 对 `msg->status` 遍历；使用 `one.pos`、`one.speed`、`one.current`、`one.temperature`；`one.name` 传入 `idMap.getIndexById(one.name)` 得到 `index` | 同上，`LegMotorStatusMsg` 回调循环 |
| `name` / 中间向量下标 | `getIndexById` 与 CAN id 表对应；腿部 `index` 0–11 与 `l_hip_roll`…`r_ankle_roll` 固定顺序见映射表初始化 | `rl_control_new/src/plugins/rl_control_new/include/bodyIdMap.h`（`legIds` / `legNames`） |
| \(\tau_{est}\) 与 `ct_scale` 对齐 | `tau_fed_midVec(index) = one.current * ct_scale_midVec(index)`；腿部标定：`ct_scale_midVec.head(12) << ct_scale.head(12)`，与腿 `index` 同一下标 | `RLControlNewPlugin.cpp`（`onInit` 与状态回调） |
| `temperature` 在插件侧形态 | 写入 `temperature_midVec(index) = one.temperature`，与 `pos/speed/tau` 同一 `index`；IDL：**`float32` 标量**，**°C**（**TienKung_ROS** `MotorStatus.msg` + `plan.md` §0） | 同上 + **TienKung_ROS** |
| `dt` 含义（配置层） | `tg22_config.yaml` 中 `dt`；`LoadConfig` 读入 `dt`、`ct_scale`；控制循环里用 `dt` 做周期（与 `/leg/status` 发布频率是否一致**无法**在仓库内证明） | `rl_control_new/config/tg22_config.yaml`、`RLControlNewPlugin.cpp`（`LoadConfig`、`rlControl` 内定时） |
| `ct_scale` 与 `motor_num` | 配置中声明 `motor_num`；`ct_scale` 为 YAML 数组；**当前仓库中 `ct_scale` 元素个数与 `motor_num` 可能不一致**（需与机载一致配置核对） | `tg22_config.yaml` |

**`TienKung-Lab`**

| P0 TODO 主题 | 静态可答内容 | 证据位置 |
|:-------------|:-------------|:---------|
| 训练用 `T_leg` 槽位顺序（**唯一准则**） | **以 Ultra 为准**：左腿 / 右腿各 6 关节 `hip_roll` → `hip_yaw` → `hip_pitch` → `knee_pitch` → `ankle_pitch` → `ankle_roll`。Deploy 腿中间向量髋序为 R–P–Y，**不得**与 `T_leg[i]` 按下标混用；须按关节语义映射到 Ultra 第 `i` 槽位 | `legged_lab/envs/ultra/ultra_env.py`（`left_leg_ids` / `right_leg_ids` 的 `find_joints` `name_keys` 顺序）；`configs/leg_index_mapping.yaml` 与之对齐 |
| 与 USD/MJCF 关节名 | 上述 `*_joint` 名称与 Ultra 资产一致（用于和 `leg_index_mapping.yaml` 对齐） | 同上 |

**已可由 TienKung_ROS + plan 静态回答的部分**

- `MotorStatus` 单条：`MotorStatus.msg`（**TienKung_ROS**）— `float32 temperature`，**无 `#` 注释写单位时仍以工程约定 °C 执行**（见 `plan.md` §0）。
- **温度单位**：**已约定为 °C**（见 `plan.md` §0、`MotorDevice.cpp` 解码）。

**仍须实机或 bag 核对**

- **`/leg/status` 实际频率**、stamp 抖动、与 `dt` 是否一致：须 bag 或 `scripts/check/p0_check.py` 在实机运行（**TienKung_ROS 不包含固定发布周期**）。
- **`current` 符号与力矩方向**：IDL 仅声明 **A**；正电流与关节正方向的约定需小运动对照。
- **Deploy 栈与 TienKung_ROS 是否同版 `bodyctrl_msgs`**：若实机来自不同分支，以实机 `interface show` 为准。

---

### 由 `TienKung_ROS` 仓库可静态回答的补充项（归档 IDL + 部分实现）

以下路径均相对 **`TienKung_ROS`** 仓库根目录；用于勾掉或收窄 `todo` 里「IDL 未知」类问题。**若实机固件/消息包与仓库分叉，仍以实机为准。**

| 原 todo 疑问 | 在 TienKung_ROS 中的结论 | 证据 |
|:-------------|:-------------------------|:-----|
| `MotorStatusMsg` 聚合结构 | `std_msgs/Header header` + **`MotorStatus[] status`** → **`status` 为变长序列**（非固定长度数组） | `src/bodyctrl_msgs/msg/MotorStatusMsg.msg` |
| 每条 `status` 是否带独立时间戳 | **否**；仅整条消息有 **`Header`**，元素无独立 stamp | 同上 |
| `MotorStatus` 除文档列外是否还有字段 | **否**（当前 IDL 仅 5 个成员） | `src/bodyctrl_msgs/msg/MotorStatus.msg`：`name`, `pos`, `speed`, `current`, `temperature` |
| `one.name` / `name` 类型与语义 | **`uint16 name # MotorName`**，为 **`MotorName.msg` 中枚举常量**（如 `MOTOR_LEG_LEFT_1`…），**不是**字符串 | `MotorStatus.msg` + `MotorName.msg` |
| `current` 单位（安培/毫安） | **`float32 current # A`** → **安培 (A)** | `MotorStatus.msg`；`MotorDevice.cpp` 中 `uint_to_float` 与 `CUR_MIN_*`/`CUR_MAX_*`（安培量级）一致 |
| `pos` / `speed` 单位 | **`pos`：`# rad`**；**`speed` 行注释写 `# rad`（疑似笔误，物理上应为 rad/s）** | `MotorStatus.msg`；与 `MotorDevice::OnStatus` 中 `speed = uint_to_float(spd_int, SPD_MIN, SPD_MAX, …)` 的角速度解码一致 |
| `temperature` | **`float32`**，单字段；**°C 工程约定**见 `plan.md` §0、`MotorDevice.cpp` 解码 | `MotorStatus.msg`、`MotorDevice.cpp` |
| `Imu` 中 `euler` 单位（与 P1 相关） | **`bodyctrl_msgs/Euler`**：`roll`/`pitch`/`yaw` 为 **`float64`**；Xsens 插件中 **`/180*pi` → 发布为弧度**（该路径见 `XSensImuPlugin.cpp`，若 Deploy 用另一 IMU 插件需单独核对） | `Imu.msg`、`Euler.msg`、`body_control/.../XSensImuPlugin.cpp` |
| 消息是否只含腿 | **IDL 不区分身体部位**；`status` 长度 = 当帧聚合进来的电机数。**腿 12 路须按 `name`∈{`MOTOR_LEG_*`} 过滤**，不能假定「前 12 条即腿」 | `MotorName.msg` 枚举 + 变长 `status[]` |

**TienKung_ROS 仍无法仅由仓库回答的**

- 与 **Deploy_Tienkung** 当前编译的 `bodyctrl_msgs` 是否**字节级一致**（需 `ros2 interface show` 或对比 commit）。
- **`/leg/status` 话题名**在 TienKung_ROS 的 `BodyControlPlugin` 里可见 `motor_state` 等（ROS1 风格）；**ROS2 + `/leg/status`** 以你方 **Deploy** 配置为准，本仓库不替代实机图。

### P0 核对脚本（`scripts/check/p0_check.py`）

**定位**：仅在**实机**（已 `source` ROS2、话题有数据）运行，用于确认 `ros2 interface show` 与 `/leg/status` 上**离线无法确定**的接口类型与样本形态。**不**读取 `Deploy_Tienkung` / `TienKung-Lab` 源码做静态审计；仓库与配置的一致性请在开发机通过文档、代码审阅与 `leg_index_mapping.yaml` 自行完成。

脚本内嵌与 `configs/leg_index_mapping.yaml` 一致的 `T_leg` 关节名列表，仅用于核对**实机一帧** `status` 中的 `name` 能否映射满 12 路腿电机。

**用法（在机器人或已接入实机 ROS 域的机器上）**：

```bash
cd Tienkung_thermal

python scripts/check/p0_check.py

# 与 Deploy tg22_config.yaml 中的 dt 对比消息 stamp 间隔（可选）
python scripts/check/p0_check.py --dt 0.0025

python scripts/check/p0_check.py --json
```

常用参数：`--topic`（默认 `/leg/status`）、`--dt-tolerance-ratio`（与 `--dt` 对比时的相对容差，默认 `0.2`）、`--echo-multi-timeout`（连续 echo 采集秒数，默认 `8`）。

**预期终端输出（非 `--json`）**：

1. 标题：`TienKung Thermal — P0 实机确认`，分隔线，一行统计：`统计: FAIL=… PASS=… PENDING=…`。
2. 每条检查：`[PASS|FAIL|PENDING] p0.xxx`，一行摘要，可选 `  - ` 明细（接口字段解析、echo 样本统计、stamp 间隔等）。
3. **退出码**：存在任一 `FAIL` 时为 **1**，否则为 **0**（`PENDING` 不导致失败）。

**状态含义**：

| 状态 | 含义 |
|:-----|:-----|
| `PASS` | 实机侧该项检查通过 |
| `FAIL` | 无 `ros2`、interface 失败、话题无数据、字段与基线假设不一致、或（若提供 `--dt`）间隔与 `dt` 偏差过大等 |
| `PENDING` | 脚本不判定物理量（电流安培/符号、温度 °C 等），或未能采到足够多帧计算间隔 |

**与下方 P0 TODO**：脚本对应「实机 `interface show`、单帧 `/leg/status` 形态、消息时间间隔」；映射表与插件代码级结论不在此脚本中重复验证。

### P0 TODO List

- [x] **拿到 `MotorStatusMsg` 的权威定义并归档**（**以 TienKung_ROS 为权威 IDL**）
  - **结论**：见 **`TienKung_ROS/src/bodyctrl_msgs/msg/MotorStatusMsg.msg`**、**`MotorStatus.msg`**。
  - **已确定**：
    - [x] `status` 为 **`MotorStatus[]`** → **变长序列**（非固定长度）。
    - [x] 每条元素 **无** 独立时间戳；**仅**整条消息的 **`Header`**。
    - [x] `MotorStatus` **仅** 五字段；**无** 错误码等扩展成员（若固件后续加字段需改 `.msg` 并再版）。
  - **仍建议**：实机 `ros2 interface show` 与仓库 IDL **diff**，确认无分叉。

- [x] **确认 `one.temperature` 的形态、单位与语义**（**已基线确定**）
  - **结论**：**单标量 `float32`**；**单位 °C**；IDL 见 **TienKung_ROS** `MotorStatus.msg`；总线解码见 **TienKung_ROS** `MotorDevice.cpp`（`(data[6]-50)/2` → 浮点 °C 尺度）。与 `plan.md` §0、§2.1 一致。
  - **仍建议实机抽查**：静止/负载 bag 曲线是否合理、与机载显示是否一致；若固件与 **TienKung_ROS** 分叉，更新文档。
  - ~~多元素数组 / 非 °C 码值~~：不作为当前基线；若实机发现与 IDL 不符，再开项。

- [ ] **确认 `one.current` 的单位与符号约定**
  - **为何重要**：`tau_est = current * ct_scale` 是核心特征；单位或符号错会导致 `tau_sq` 与真实负载无关。
  - **已由 TienKung_ROS 静态回答**：
    - [x] **单位**：IDL **`float32 current # A`** → **安培**（非 mA 整型）。
  - **仍须实机/小运动**：
    - [ ] **符号**：正电流与关节正方向、`tau_est` 是否一致（IDL 不定义）。
  - **建议**：单关节小幅度正弦运动，对照 `pos/speed` 与 `current` 符号是否合理。

- [ ] **实机验证 `Deploy → Ultra T_leg[0..11]` 映射**
  - **为何重要**：**`T_leg` 以 Ultra 顺序为唯一准则**（见 `plan.md` §1.2、`leg_index_mapping.yaml`）。Deploy 中间向量髋序为 **R–P–Y**，Ultra 为 **R–Y–P**；若按下标硬拷贝会错三条髋轴。
  - **待确定**：
    - [ ] 每条 `status` 经 `name`/id 先到 Deploy 中间向量，再**按语义**对齐到 Ultra 的 12 个 `motor_names` 槽位，对应表已固定且无歧义
    - [ ] 左/右腿、膝/踝与 Ultra USD/MJCF 关节名一致
  - **建议**：一次只动一条腿一个关节，看哪一路 `pos` 变化，打印 `name` / index，与 **Ultra** 表核对。

- [ ] **确认 `ct_scale` 与 `current` 下标对齐方式**
  - **为何重要**：`plan.md` 写明 `tau_est` 用的是 `ct_scale[j]`，`j` 为 Deploy 中间向量下标，必须与映射后的关节一致。
  - **待确定**：
    - [ ] `ct_scale` 长度是否等于 `motor_num` 或仅腿子集
    - [ ] 向量顺序与 `getIndexById` 得到的 `index` 是否一致
    - [ ] 乘积量级是否接近 N·m（与关节额定对比数量级）
  - **建议**：对比官方或标定文档；若无，用已知负载工况做粗校验。

- [ ] **明确 `tg22_config.yaml` 中 `dt` 的含义**
  - **为何重要**：数值 `ddq` 用 `diff(speed)/dt`；若 `dt` 不是实际控制/采样周期，加速度特征全错。
  - **待确定**：
    - [ ] `dt` 是否等于 `/leg/status` 有效更新间隔
    - [ ] 若 Topic 频率与 `dt` 不一致，以哪一个为「真」时间步（通常以消息时间戳间隔为准）
  - **建议**：用 bag 统计相邻消息 `stamp` 差分的中位数，与 `dt` 对比。

- [ ] **测量 `/leg/status` 实际发布特性**
  - **为何重要**：`plan.md` 建议 20 Hz 统一网格；若实际 100 Hz 且抖动大，抽帧策略必须写清楚。
  - **待确定**：
    - [ ] 平均频率、最小/最大间隔、丢帧率
    - [ ] 是否适合「每 k 帧取 1 帧」还是必须按时间戳重采样
  - **建议**：10–60 s bag，`ros2 bag info` 或自写脚本统计间隔直方图。

---

## P1 — 特征语义与时间对齐

### P1 说明

P1 保证：**除温度电流外，其余输入在物理上与训练假设一致**；**多 Topic 拼在一起时同一时刻语义对齐**。否则模型在实机上会「离线很好、在线漂移」。

### P1 TODO List

- [ ] **确认 `one.name`（或等价 id 字段）与 `getIndexById` 的输入语义**
  - **为何重要**：映射链路的起点错了，后面全错。
  - **已由 TienKung_ROS 静态回答**：
    - [x] **`name` 类型为 `uint16`**，语义为 **`MotorName` 枚举**（见 `MotorName.msg`），**不是** UTF-8 字符串。
  - **仍须与 Deploy 对照**：
    - [ ] **Deploy** `getIndexById(one.name)` 若期望 CAN id，则须确认 **ROS2 消息与插件**是否与 **TienKung_ROS** 枚举一致；否则映射表以实机为准。
  - **建议**：实机一帧 `status` 打印 `name` 数值，对照 `MotorName.msg` 与 `bodyIdMap`。

- [ ] **确认 `one.pos` / `one.speed` 的单位与参考系**
  - **为何重要**：与 Lab 仿真或日志对齐、与 `|dq|` / 数值 `ddq` 一致。
  - **已由 TienKung_ROS 静态回答**：
    - [x] **`pos`**：`# rad`。
    - [x] **`speed`**：IDL 注释为 `rad`（**疑似笔误**）；与 `MotorDevice` 解码为角速度区间 **SPD_MIN/MAX** 一致，工程上按 **rad/s** 使用。
  - **待确定**：
    - [ ] 是否连续无跳变 / 是否包裹（依赖控制与记录方式）。
  - **建议**：与仿真同一动作对比曲线形状（尺度与符号）。

- [ ] **确认 `temperature` 与 `pos/speed/current` 在同一中间向量下标上对齐**
  - **为何重要**：避免出现「温度是腿 3、电流却是腿 5」的拼接错误。
  - **待确定**：
    - [ ] 插件写入 `temperature_midVec` 与 `pos_fed_midVec` 等是否同一套 index
  - **建议**：单关节扰动同时打印四者，确认只有一路同步变化。

- [ ] **确认 `/leg/status` 消息体与 `motor_num` 的关系**
  - **为何重要**：若一条消息含全身电机，采集端必须只取腿 12 路，不能按数组前 12 个元素截取。
  - **已由 TienKung_ROS 静态回答**：
    - [x] **`status` 为变长数组**，**不保证**「仅腿」或「顺序固定」；**不能**按前 12 元素截取。
  - **待确定**：
    - [ ] 实机单条 `/leg/status` 是否只聚合腿电机（由驱动节点配置决定）。
    - [ ] 若混有臂/腰，**按 `name`∈腿枚举** 过滤到 Ultra `T_leg`。
  - **建议**：打印单帧 `status` 长度与所有 `name`。

- [ ] **拿到 `Imu` 消息权威定义并对照插件读取字段**
  - **为何重要**：可选 9 维特征；欧拉角顺序、陀螺/加计坐标系错会导致无效甚至有害输入。
  - **已由 TienKung_ROS 静态回答（IDL）**：
    - [x] **`bodyctrl_msgs/Imu.msg`**：`geometry_msgs` 四元数/角速度/线加速度 + **`bodyctrl_msgs/Euler euler`**（`roll`/`pitch`/`yaw` 三字段 **`float64`**）。
    - [x] **Xsens 路径**：`XSensImuPlugin.cpp` 将设备欧拉从 **度转为弧度** 后写入 `msg->euler.*`。
  - **待确定**（依赖**实际发布的 IMU 插件**是否与 Xsens 一致）：
    - [ ] Deploy 侧 `OnXsensImuStatusMsg` 所用消息是否与上述一致；陀螺/加计坐标系与重力补偿。
  - **建议**：`ros2 interface show` + 静止/绕单轴旋转标定小实验。

- [ ] **测量 `/imu/status` 频率并与 `/leg/status` 对齐策略定稿**
  - **为何重要**：两路频率不同则必须在预处理中明确「谁向谁对齐、是否允许外推」。
  - **待确定**：
    - [ ] IMU 与 leg 的中位时间差
    - [ ] 采用最近邻、线性插值，还是只使用 leg 时间栅格上的 IMU 采样
  - **建议**：同一 bag 提取两者 stamp，画差分分布。

- [ ] **明确在线推理的触发周期与 ring buffer 深度**
  - **为何重要**：训练用长度 `L` 与步长 Δt 必须与在线一致，否则分布偏移。
  - **待确定**：
    - [ ] 推理是否每个控制周期调用，还是固定 50 ms / 100 ms
    - [ ] ring buffer 是否按 wall clock 还是按消息 stamp 推进
  - **建议**：在目标部署线程打时间戳日志，统计调用间隔。

---

## P2 — 训练与工程超参（可迭代）

### P2 说明

P2 不阻塞「是否正确」类问题，但影响 **Horizon、窗口、平滑、消融与延迟预算**；建议在基线跑通后按实验表推进。

### P2 TODO List

- [ ] **确定统一时间网格与落盘采样策略**
  - **为何重要**：`L`、Horizon 的「步」必须对应同一 Δt。
  - **待确定**：
    - [ ] 目标网格频率（如 20 Hz）如何从原始 `/leg/status` 得到
    - [ ] 降采样用抽帧、平均池化还是插值
  - **建议**：在文档中写死一种策略并版本化预处理脚本。

- [ ] **确定预测 Horizon 形态**
  - **为何重要**：`plan.md` 验收为约 15 s；需明确是单点还是多档监督。
  - **待确定**：
    - [ ] 仅预测 t+15 s 一个点
    - [ ] 或多档 {0.5, 1, …, 15} s 多任务头（类似 G1 文档中的多视距）
  - **建议**：与算力、标签噪声权衡；多档通常更稳但更重。

- [ ] **确认序列长度 `L` 是否足够**
  - **为何重要**：热惯性长时，`L` 过短会欠拟合慢升温。
  - **待确定**：
    - [ ] 在典型负载下，自相关或肉眼观察温升曲线，判断 5 s @ 20 Hz 是否够
  - **建议**：对比 `L ∈ {50, 100, 200}` 的验证集 MAE。

- [ ] **确定温度 EMA 的 `α`（若使用）**
  - **为何重要**：uint8 或低分辨率温度在 20 Hz 下台阶明显，EMA 改善梯度但引入滞后。
  - **待确定**：
    - [ ] 是否对输入温度做 EMA；`α` 取值（如 0.05）是否固定
    - [ ] 在线与离线是否同一套平滑
  - **建议**：小网格搜索 + 可视化预测曲线。

- [ ] **确定数值 `|ddq|` 的计算方式**
  - **为何重要**：差分对噪声敏感，方法不同特征分布不同。
  - **待确定**：
    - [ ] 前向差分 vs 中心差分
    - [ ] 是否先对 `speed` 做低通再差分
  - **建议**：消融「有/无 ddq」「两种差分」对 Val MAE 的影响。

- [ ] **消融：邻域温度特征是否加入**
  - **为何重要**：`plan.md` 允许仅 `T_leg` 相邻索引；需数据证明收益。
  - **待确定**：
    - [ ] 是否加入及 `K` 邻居数（如 2）
  - **建议**：与仅本体特征对比 Val MAE 与在线延迟。

- [ ] **消融：IMU 9 维是否加入**
  - **为何重要**：增加维度和同步复杂度。
  - **待确定**：
    - [ ] 基线不含 IMU；是否第二阶段加入
  - **建议**：跑跳、上下坡等 session 上对比。

- [ ] **实测端到端推理延迟分解**
  - **为何重要**：`plan.md` 要求单次前向 ≤ 5 ms（FP16）；预处理若占大头需优化或改 CPU 路径。
  - **待确定**：
    - [ ] 各阶段耗时：解析消息 → EMA/归一化 → 拷入 GPU → TensorRT → 反归一化
    - [ ] 目标硬件（机载 Orin / x86）上的 P50/P99
  - **建议**：用 CUDA event 或高精度计时包裹各段。

---

## 验证手段 TODO（按需勾选执行）

- [ ] 已执行：`ros2 interface show` 并保存 `MotorStatusMsg`、`Imu` 全文
- [ ] 已录制：≥30 s 的 `/leg/status` bag，并统计频率与 stamp 间隔
- [ ] 已录制：同时含 `/leg/status` 与 `/imu/status` 的 bag，用于对齐分析
- [ ] 已在采集节点打印：至少一帧完整 `status` 的 `name`、温度、电流、位置、速度与内部 index
- [ ] 已做：单关节运动学识别（确认 `T_leg` 映射）
- [ ] 已做：静止 / 缓动 / 高负载 三类短录，人工目检温度-电流-速度合理性

---

## 相关文档

- `Tienkung_thermal/docs/plan.md` — 白名单 Topic、特征清单、验收与推理预算
- `Tienkung_thermal/docs/task_memory.md` — Ultra 相对 G1 的调整要点与阶段计划
- `Tienkung_thermal/configs/leg_index_mapping.yaml` — `T_leg[0..11]` 关节名顺序
- `Tienkung_thermal/scripts/check/p0_check.py` — P0 实机核对脚本（用法见上文「P0 核对脚本」）
