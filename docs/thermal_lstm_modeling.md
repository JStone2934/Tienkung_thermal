# 天工 Ultra 腿部（12 关节）热动力学 LSTM 建模规格书

> **基准文档**: `docs/plan.md`  
> **适用范围**: `TienKung Ultra` 腿部 `12` 个关节  
> **监督信号**: `bodyctrl_msgs/msg/MotorStatus.temperature`，单标量 `float32`，单位 `°C`  
> **模型约束**: 因果 `LSTM`，禁用 `BiLSTM`，不引入 G1/29DOF/双温度通道正文方案  
> **建模范式**: **全关节联合输入/联合预测** — 同时输入 12 关节的 `(q, dq, T)` 序列，同时预测 12 关节的未来温度  
> **验收口径**: 未来 `15 s`，`12` 关节等权平均 `MAE <= 1.5°C`，单次前向推理 `<= 5 ms`  
> **工程网格**: `500 Hz`（步长 `2 ms`），与 `plan.md` §4 及 `dataset_leg_status_h5.md` 一致

---

## 1. 问题定义

### 1.1 建模目标

给定全部 `12` 个腿部关节在过去 `L` 个时间步的观测特征序列（每关节 `q, dq, T` 三维）

\[
\mathbf{X}_t = \left\{x_{t-L+1}, \ldots, x_t\right\}, \quad x_t \in \mathbb{R}^{12 \times 3}
\]

同时预测全部 `12` 个关节未来多个视距上的温度标量轨迹

\[
f_\theta\!\left(\mathbf{X}_t\right) \rightarrow \hat{\mathbf{Y}}_t \in \mathbb{R}^{12 \times H}
\]

其中

\[
\hat{\mathbf{Y}}_t =
\begin{bmatrix}
\hat{T}^{(0)}_{t+h_1} & \cdots & \hat{T}^{(0)}_{t+h_H} \\
\vdots & \ddots & \vdots \\
\hat{T}^{(11)}_{t+h_1} & \cdots & \hat{T}^{(11)}_{t+h_H}
\end{bmatrix}
\]

所有温度量均以 `°C` 表示，并直接对应 `MotorStatus.temperature` 的工程含义。

**与旧版单关节架构的区别**：旧版每次仅输入一个关节的特征并预测该关节温度；新版同时输入全部 12 关节的运动学与温度状态，使模型能学习关节间的热耦合与姿态-温度关联。

### 1.2 采样与预测视距

- 统一训练网格采用 **`500 Hz`**（步长 `2 ms`），与 `plan.md` §4 及 `dataset_leg_status_h5.md` 一致。
- 输入窗口长度固定为 **`L = 2500`**，对应约 `5 s` 历史。
- 预测 horizon 采用 `H = 9` 个视距点，对应步数 `h = [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]`。
- 在 `500 Hz` 下，上述 horizon 对应 `0.5 s, 1.0 s, 2.0 s, 3.0 s, 5.0 s, 7.0 s, 10.0 s, 12.0 s, 15.0 s`。
- 主验收仅看 `15.0 s` 对应视距与 `12` 关节等权平均 `MAE`。

### 1.3 工程约束

| 指标 | 要求 |
|:-----|:-----|
| 主验收指标 | `15 s` 视距、12 关节等权平均 `MAE <= 1.5°C` |
| 训练损失 | 全关节 `Huber + MAE`，支持 `w_0..w_11` 训练权重 |
| 推理因果性 | 仅使用 `t` 时刻及之前的数据 |
| 模型时延 | 单次前向 `<= 5 ms`（FP16） |
| 工程网格 | `500 Hz`（步长 `2 ms`），输入窗口 `L = 2500`（约 `5 s`） |
| 输入维度 | 每关节 `3` 维（`q, dq, T`），12 关节拼接后 `D = 36` |
| 输出维度 | `(B, 12, H)`，同时预测全部 12 关节 × `H` 个视距 |
| 数据边界 | 原始观测仅来自 `/leg/status`（`MotorStatusMsg`），以 `ros2ws` 中 `.msg` 定义为界 |
| 数据质量 | `MotorStatus.error ≠ 0` 的帧整帧丢弃，不进入训练或评估 |

### 1.4 因果性与接口边界

- 禁止使用未来帧、双向循环网络或任何离线平滑结果作为在线特征。
- `TienKung-Lab` 不提供可作为监督的电机温度真值；温度标签仅能来自 `data/bags/` 中 `/leg/status` 的实机录制（消息定义以 `Tienkung/ros2ws` 为准）。
- 所有特征必须可由 `plan.md` 白名单中的字段或其确定性后处理得到。

---

## 2. 系统边界与关节顺序

### 2.1 `T_leg[0..11]` 的唯一顺序

腿部温度向量 `T_leg` 的顺序唯一以 `Ultra` 为准，并与 `configs/leg_index_mapping.yaml` 对齐：

| 下标 | 关节名 |
|:----:|:-------|
| 0 | `hip_roll_l_joint` |
| 1 | `hip_yaw_l_joint` |
| 2 | `hip_pitch_l_joint` |
| 3 | `knee_pitch_l_joint` |
| 4 | `ankle_pitch_l_joint` |
| 5 | `ankle_roll_l_joint` |
| 6 | `hip_roll_r_joint` |
| 7 | `hip_yaw_r_joint` |
| 8 | `hip_pitch_r_joint` |
| 9 | `knee_pitch_r_joint` |
| 10 | `ankle_pitch_r_joint` |
| 11 | `ankle_roll_r_joint` |

bag 中 `/leg/status` 各电机的 **数组顺序**通常为 **`MotorName.msg` 的数值顺序**：左腿 **`51–56`**（`MOTOR_LEG_LEFT_1`…`6`）再接右腿 **`61–66`**（`MOTOR_LEG_RIGHT_1`…`6`）。这与 `Tienkung_thermal/data/leg_status_motor_samples.json` 等解码样例一致。**该顺序沿用历史 Deploy 腿中间向量排列**（髋部为 **R–P–Y**），与 Ultra 的 **R–Y–P** 不一致，因此：

- 禁止把 **`status` 数组的下标**（0…11）直接当作 `T_leg[i]`。
- 必须按 **`MotorStatus.name`（CAN ID）→ `T_leg[i]`** 的固定表重排；完整表见 **`plan.md` §1.2.1** 与 **`configs/leg_index_mapping.yaml`**（`can_id_to_t_leg`、`deploy_j_to_t_leg_i`），与代码 **`tienkung_thermal/bags/mapping.py`** 中 `CAN_TO_T_LEG` / `CAN_TO_DEPLOY_J` 一致。
- 消息字段定义以本机 **`ros2ws`** 为准（例如 `Tienkung/ros2ws/...` 或与录包环境一致的备份如 **`robot_control_backup_.../ros2ws/install/bodyctrl_msgs/...`**）；**`MotorName.msg` / `MotorStatus.msg` 与上述映射同时满足时**，样例 bag 与文档对齐。
- `ct_scale[j]`（`j` 为 Deploy 腿中间向量下标 `0…11`，与 CAN `51–56,61–66` 一一对应）来自 `ct_scale_profiles.yaml` 或录包同期机载快照；须先按 **`CAN_TO_DEPLOY_J`** 取 `j`，乘后再经 **`CAN_TO_T_LEG`** 置换到 Ultra 下标 `i`。

### 2.2 核心数据流

```mermaid
flowchart LR
  rosbag[data/bags rosbag2] --> legStatus[/leg/status MotorStatusMsg]
  rosbag --> imuStatus[/imu/status Optional]
  legStatus --> nameMapping[CAN_ID_to_T_leg_Ultra]
  imuStatus --> featureBuilder[FeatureBuilder]
  nameMapping --> featureBuilder
  featureBuilder --> lstmModel[CausalLSTM]
  lstmModel --> horizonPred[FutureTemperatureInCelsius]
  horizonPred --> offlineEval[MAE15sAcceptance]
  horizonPred --> onnxTrt[ONNXToTensorRT]
```

### 2.3 原始观测白名单

以 `ros2ws` 中 `MotorStatus.msg` 全字段为界（见 `plan.md` §1.3.2）：

| 逻辑量 | 来源 | 说明 |
|:-------|:-----|:-----|
| `q` | `one.pos` | 关节位置（rad） |
| `dq` | `one.speed` | 关节角速度（rad/s） |
| `T` | `one.temperature` | 温度标量，单位 `°C`（**监督标签**，同时作为输入特征） |
| `error` | `one.error` | 电机故障码（`uint32`）；**仅用于数据质量过滤**，`error ≠ 0` 整帧丢弃，不作为 LSTM 输入 |

**全关节联合建模下的输入**：每个时间步将 12 关节的 `(q, dq, T)` 按 `T_leg[0..11]` 顺序拼接为 `36` 维向量。

**不再作为基线输入的量**（可在后续消融中重新引入）：

- `current`（电机电流）
- `voltage`（电机端电压）
- `tau_est`（估计力矩）及其派生量 `tau_sq`、`dq_abs`、`ddq_abs`
- IMU 9 维上下文

### 2.4 禁止直接纳入基线模型的量

以下量不得写入基线模型输入，除非先被正式纳入 `plan.md` 白名单：

- 原生 `ddq` 话题（当前由 `speed` 数值差分替代）
- `MotorStatus1` 双温度通道（`motortemperature` + `mostemperature`，topic 名未确认）
- `PowerStatus` 板级热数据（`leg_a_temp` / `leg_b_temp` 等）
- BMS、主板温度、风扇状态
- 外置温湿度计
- 第三方机器人或旧版 G1 相关字段
- Lab 中不存在的虚构热标签

---

## 3. 物理先验与特征工程

### 3.1 全关节联合建模的物理直觉

在人形机器人行走/跑步等任务中，12 个腿部关节的运动状态高度耦合：

- **姿态-温度关联**：不同步态（站立、行走、跑步、转弯）下各关节的负载分布不同，导致温升模式差异显著。全关节输入使模型能从整体姿态推断各关节的热负载。
- **关节间热耦合**：相邻关节通过机械结构存在热传导；同侧关节在步态周期中存在协同运动模式。
- **对称性**：左右腿在正常步态下近似对称，模型可利用这一先验。

简化热平衡关系（指导特征选择）：

\[
C_i \frac{dT_i}{dt} \approx \alpha_i \dot{q}_i^2 + \beta_i f(q_0, \ldots, q_{11}) - k_i(T_i - T_{amb})
\]

其中 \(\dot{q}_i^2\) 与速度相关的损耗代理可由 `dq` 隐式学习，\(f(q_0, \ldots, q_{11})\) 表示全关节姿态对单关节负载的影响——这正是联合建模的核心优势。

### 3.2 每关节输入特征

对每个关节 `T_leg[i]`，在时间步 `t` 使用以下 `3` 个特征：

| 特征名 | 定义 | 维度 | 备注 |
|:-------|:-----|:----:|:-----|
| `q` | `one.pos` | 1 | 关节位置（rad）；反映姿态 |
| `dq` | `one.speed` | 1 | 关节角速度（rad/s）；与机械损耗/发热相关 |
| `T` | `one.temperature` | 1 | 当前温度 `°C`；热状态的直接观测 |

每关节 **`D_per_joint = 3`**，12 关节拼接后总输入维度 **`D = 36`**。

### 3.3 全关节特征拼接

在每个时间步 `t`，将 12 关节的特征按 `T_leg[0..11]` 顺序拼接：

\[
x_t = \left[q_0, dq_0, T_0, \; q_1, dq_1, T_1, \; \ldots, \; q_{11}, dq_{11}, T_{11}\right] \in \mathbb{R}^{36}
\]

拼接顺序为：**关节优先**（joint-major），即先排完一个关节的全部特征，再排下一个关节。

### 3.4 预处理约定

| 项目 | 规则 |
|:-----|:-----|
| 重采样 | 原始 `~1 kHz` 序列经**线性插值**统一到 **`500 Hz`**（步长 `2 ms`），与 `plan.md` §4 一致 |
| `error` 过滤 | 任一腿部电机 `error ≠ 0` 的原始帧**整帧丢弃**，不进入有效序列（`plan.md` §2.1.1） |
| 裁剪 | 仅允许对明显异常值做工程裁剪，需保留原值日志 |
| 归一化 | 使用训练集统计量做 `Z-score` |

### 3.5 张量组织

- 训练样本以 `(session, start_t)` 为单位切片（不再按关节拆分）。
- 单个样本输入形状为 `[L, D]`，其中 `D = 36`。
- 批量训练输入形状为 `[B, L, D]`。
- 标签形状为 `[B, 12, H]`，表示全部 12 关节未来多个视距的温度。

---

## 4. 模型架构

### 4.1 设计选择

本文采用"共享骨干 + 12 关节独立输出头"的因果 LSTM 结构，**全关节联合输入/联合预测**：

- 输入为全部 12 关节的 `(q, dq, T)` 拼接序列 `[B, L, 36]`。
- 共享骨干（投影层 + LSTM）从全关节运动学与温度状态中学习通用热动态规律与关节间耦合。
- 每个 `T_leg[i]` 使用独立线性输出头保留关节差异。
- 前向一次性输出全部 12 关节的预测 `(B, 12, H)`，无需 `joint_index` 参数。

该设计的优势：

- 模型能从全局姿态（12 关节的 `q`）推断各关节在不同工况下的热负载分布；
- 关节间的热耦合（如相邻关节热传导、对称步态下的协同温升）可被隐式学习；
- 在线部署时单次前向即可获得全部 12 关节的预测，无需循环调用。

### 4.2 网络拓扑

```text
Input: state_seq (B, 2500, 36)
        │                          D = 12 * 3 = 36
        ▼
Linear(36 -> d_proj)
LayerNorm
GELU
        │
        ▼
Causal LSTM
input_size = d_proj
hidden_size = d_hidden
num_layers = 2
batch_first = True
dropout = p_drop
        │
        ▼
Take last hidden state (B, d_hidden)
        │
        ▼
12 x JointHead[i]
Linear(d_hidden -> d_mid)
GELU
Linear(d_mid -> H)
        │
        ▼
Stack all heads: (B, 12, H=9)
Predicted future temperature in Celsius for all 12 joints
```

### 4.3 推荐超参数

| 超参数 | 推荐值 | 说明 |
|:-------|:------:|:-----|
| `L` | `2500` | 5 秒历史窗口（`500 Hz`） |
| `H` | `9` | 对应 `0.5 s` 到 `15 s` |
| `D` (input_dim) | `36` | `12 joints × 3 features (q, dq, T)` |
| `d_proj` | `32` | 输入投影维度 |
| `d_hidden` | `96` | LSTM 隐层维度 |
| `n_layers` | `2` | LSTM 层数 |
| `p_drop` | `0.10` | Dropout |
| `d_mid` | `64` | 输出头中间层维度 |
| `n_joints` | `12` | 关节数 |
| 参数规模 | `~210K` | 共享骨干 + 12 头 |

### 4.4 PyTorch 参考定义

```python
import torch
import torch.nn as nn


class UltraThermalLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,
        proj_dim: int = 32,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.10,
        mid_dim: int = 64,
        horizon: int = 9,
        n_joints: int = 12,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.horizon = horizon
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, mid_dim),
                    nn.GELU(),
                    nn.Linear(mid_dim, horizon),
                )
                for _ in range(n_joints)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)           # (B, L, d_proj)
        lstm_out, _ = self.lstm(x)       # (B, L, d_hidden)
        h_last = lstm_out[:, -1, :]      # (B, d_hidden)
        all_preds = torch.stack(
            [head(h_last) for head in self.heads], dim=1
        )  # (B, n_joints, H)
        return all_preds                 # (B, 12, H)
```

---

## 5. 损失函数与训练目标

### 5.1 训练损失

训练阶段对全关节预测采用 `Huber + MAE` 组合损失。对每个样本，预测与标签均为 `(12, H)`：

\[
\mathcal{L}_{per\_joint}(i)
= \lambda_h \cdot \operatorname{Huber}\!\left(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}\right)
+ \lambda_m \cdot \operatorname{MAE}\!\left(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}\right)
\]

全关节加权损失：

\[
\mathcal{L}_{train}
= \frac{\sum_{i=0}^{11} w_i \cdot \mathcal{L}_{per\_joint}(i)}{\sum_{i=0}^{11} w_i}
\]

推荐取值：

- `lambda_h = 0.5`
- `lambda_m = 0.5`
- `Huber delta = 1.0`

### 5.2 关节权重

- 默认 `w_0 ... w_11 = 1`（等权）。
- 训练可使用非等权重平衡重点关节（如膝关节、髋 pitch 等高负载关节）。
- 验收与 Gate 一律仅看 `12` 关节等权平均 `MAE`，不继承训练权重。

### 5.3 监控指标

每个 epoch 至少记录：

| 指标 | 用途 |
|:-----|:-----|
| `train_loss` | 训练收敛 |
| `val_mae_15s_equal_weight` | 主 Gate 指标（12 关节等权 MAE @ 15s） |
| `val_mae_per_joint_15s` | 分关节精度 |
| `max_ae` | 监控极端错误 |

---

## 6. 数据流水线

### 6.1 从 rosbag 到训练样本的处理流程

1. 从 rosbag 解码 `/leg/status`（`MotorStatusMsg`）。
2. 用每条 `MotorStatus` 的 `name`（CAN ID）映射到 Ultra `T_leg[0..11]`（§2.1 映射表）。
3. **丢弃整帧**：`len(status) ≠ 12`、CAN 未知、同槽位重复、或**任一关节 `error ≠ 0`**。
4. 在有效时间序列上以 **`numpy.interp` 线性插值**重采样到 **`500 Hz`**（步长 `2 ms`）。
5. 按 session 划分 Train / Val / Test（整段 session 不得拆分到不同集合）。
6. 用滑动窗口生成 `[L, 36] -> [12, H]` 样本。

### 6.2 HDF5 组织（与 `dataset_leg_status_h5.md` 对齐）

```text
{session_id}.h5
├── (根级属性)
│   ├── sample_rate_hz          # int, 500
│   ├── dt_grid_s               # float, 0.002
│   ├── source_rosbag           # str, rosbag2 目录绝对路径
│   ├── t_leg_order             # str, 12 个 Ultra 关节名（逗号分隔）
│   └── export_timestamp_utc    # str, ISO 8601
├── timestamps                  # float64, shape (N,)，秒，步长 0.002 s
├── joints/
│   ├── q                       # float32, shape (N, 12)
│   ├── dq                      # float32, shape (N, 12)
│   └── temperature             # float32, shape (N, 12)，°C，监督标签 + 输入特征
└── metadata/                   # HDF5 属性，用于质检与追溯
    ├── n_raw_messages_leg_status
    ├── n_valid_raw_frames
    ├── n_skipped_error_nonzero
    └── n_grid_frames           # = len(timestamps)
```

说明：

- `temperature` 既是输入特征也是监督标签（预测未来温度）。
- 全关节联合建模仅需 `q`、`dq`、`temperature` 三个字段。
- HDF5 中可能仍包含 `current`、`voltage`、`tau_est` 等历史字段，但当前模型不使用。

### 6.3 Dataset 参考实现

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

JOINT_FIELDS = ("q", "dq", "temperature")
N_JOINTS = 12
D_PER_JOINT = 3


class UltraThermalDataset(Dataset):
    def __init__(
        self,
        h5_paths: list[str],
        seq_len: int = 2500,
        horizon_steps: list[int] | None = None,
        stride: int = 50,
    ) -> None:
        if horizon_steps is None:
            horizon_steps = [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]
        self.seq_len = seq_len
        self.horizon_steps = horizon_steps
        self.max_horizon = max(horizon_steps)
        self.stride = max(1, stride)

        # Load all sessions into memory
        self._caches = []
        self._cum_start = []
        total = 0
        for path in h5_paths:
            with h5py.File(path, "r") as f:
                n_frames = f["timestamps"].shape[0]
                data = {}
                for field in JOINT_FIELDS:
                    data[field] = np.asarray(f[f"joints/{field}"], dtype=np.float32)
            valid_len = n_frames - seq_len - self.max_horizon
            if valid_len <= 0:
                continue
            n_windows = (valid_len + self.stride - 1) // self.stride
            self._cum_start.append(total)
            self._caches.append((data, n_windows))
            total += n_windows
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        import bisect
        si = bisect.bisect_right(self._cum_start, idx) - 1
        local = idx - self._cum_start[si]
        data, _ = self._caches[si]
        start_t = local * self.stride
        sl = slice(start_t, start_t + self.seq_len)

        # Build (L, 36): 12 joints x 3 features, joint-major order
        cols = []
        for j in range(N_JOINTS):
            for field in JOINT_FIELDS:
                cols.append(data[field][sl, j])
        x = np.stack(cols, axis=-1)  # (L, 36)

        # Build target (12, H): future temperature for all joints
        target_idx = start_t + self.seq_len
        target = np.stack(
            [
                np.array(
                    [data["temperature"][target_idx + h - 1, j] for h in self.horizon_steps],
                    dtype=np.float32,
                )
                for j in range(N_JOINTS)
            ]
        )  # (12, H)

        return torch.from_numpy(x), torch.from_numpy(target)
```

### 6.4 数据集划分

按采集 session 划分，禁止在同一 session 内拆分到不同集合：

| 划分 | 占比 | 约束 |
|:-----|:----:|:-----|
| Train | 70% | 覆盖多种工况与负载水平 |
| Val | 15% | 包含完整高负载与冷却段 |
| Test | 15% | 独立 session，训练中完全不可见 |

---

## 7. 训练协议与消融

### 7.1 优化器与训练设置

| 配置项 | 推荐值 |
|:-------|:-------|
| 优化器 | `AdamW` |
| 学习率 | `1e-3` |
| 权重衰减 | `1e-4` |
| 调度器 | `CosineAnnealingWarmRestarts(T_0=20, T_mult=2)` |
| Batch Size | `128` |
| 最大 Epoch | `200` |
| 梯度裁剪 | `max_norm = 1.0` |
| Early Stopping | `patience = 15`，监控 `val_mae_15s_equal_weight` |

### 7.2 训练伪代码

```python
model = UltraThermalLSTM(input_dim=36, horizon=9)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_gate = float("inf")
patience = 0

for epoch in range(200):
    model.train()
    for x, target in train_loader:
        # x: (B, L, 36), target: (B, 12, H)
        pred = model(x)  # (B, 12, H)
        loss = thermal_loss(pred, target, joint_weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()

    gate = evaluate_equal_weight_mae_15s(model, val_loader)
    if gate < best_gate:
        best_gate = gate
        patience = 0
        save_checkpoint(model, "best_ultra_thermal.pt")
    else:
        patience += 1
        if patience >= 15:
            break
```

### 7.3 建议的消融矩阵

| 实验 ID | 变量 | 基线 | 对比项 | 关注指标 |
|:--------|:-----|:-----|:-------|:---------|
| A1 | 每关节特征数 | `q, dq, T` (D=36) | 加入 `current, voltage` (D=60) | `15 s MAE` |
| A2 | 序列长度 `L` | `2500` | `1250`, `5000` | MAE 与时延 |
| A3 | `d_hidden` | `96` | `64`, `128`, `192` | MAE 与参数量 |
| A4 | 输出结构 | 12 独立头 | 全共享单头 (hidden->12*H) | MAE 与稳定性 |
| A5 | IMU 上下文 | 关闭 | 启用 (+9 维全局特征) | `15 s MAE` 与时延 |
| A6 | stride | `50` | `10`, `100` | MAE 与训练速度 |

消融结论应以主 Gate 指标为核心，不得仅凭训练损失决定最终方案。

---

## 8. 离线评估与验收

### 8.1 评估指标

| 指标 | 定义 | 说明 |
|:-----|:-----|:-----|
| `MAE@h_k` | 各 horizon 的平均绝对误差 | 报告视距退化趋势 |
| `MAE_15s_equal_weight` | `h = 7500` 时 12 关节等权平均 MAE | 主验收指标 |
| `MAE_high_load` | 高负载片段上的 MAE | 风险评估 |
| `MAE_cooling` | 冷却片段上的 MAE | 冷却动态评估 |
| `MaxAE` | 全测试集最大单点误差 | 安全边界观察 |
| `Latency_FP16_ms` | FP16 前向时间 | 部署门控 |

### 8.2 分关节误差热力图

误差热力图维度固定为 `12 x H`，行对应 `T_leg[0..11]`，列对应 9 个 horizon（步数 @ `500 Hz`）：

```text
              0.5s    1.0s    2.0s    3.0s    5.0s    7.0s    10.0s   12.0s   15.0s
              (250)   (500)   (1000)  (1500)  (2500)  (3500)  (5000)  (6000)  (7500)
hip_roll_l     ...
hip_yaw_l      ...
hip_pitch_l    ...
knee_pitch_l   ...
ankle_pitch_l  ...
ankle_roll_l   ...
hip_roll_r     ...
hip_yaw_r      ...
hip_pitch_r    ...
knee_pitch_r   ...
ankle_pitch_r  ...
ankle_roll_r   ...
```

分析重点：

- 某些关节是否在 `10 s` 之后明显失稳。
- 左右对称关节误差是否存在系统偏差。
- 高负载动作切换点是否集中产生大误差。

### 8.3 失败案例分析

对测试集 `Top-K` 大误差样本记录以下字段：

- `joint_name`
- `timestamp`
- `session_id`
- `phase / task`
- `target_temp_c`
- `pred_temp_c`
- `error_sign`

并重点排查：

- 是否为 `T_leg` 映射错误；
- 是否出现在温度快速上升或快速冷却区间；
- 是否与某个关节头部过拟合有关；
- 是否受到输入缺测或异常值影响。

---

## 9. 部署与在线推理

### 9.1 模型导出链路

```text
PyTorch (.pt)
    │
    ▼
ONNX (opset 17)
    │
    ▼
TensorRT FP16 (.engine)
```

导出时约定：

- 输入名：`state_seq`（无 `joint_index`）
- 输出名：`temp_c_horizon`
- 输入形状：`(B, L, 36)`
- 输出形状：`(B, 12, H)`
- ONNX 与 PyTorch 数值误差应控制在 `1e-4` 量级
- TensorRT 与 ONNX 的 `MAE` 偏差应远小于 `0.1°C`

```python
dummy_x = torch.randn(1, 2500, 36)

torch.onnx.export(
    model,
    (dummy_x,),
    "ultra_thermal_lstm.onnx",
    input_names=["state_seq"],
    output_names=["temp_c_horizon"],
    dynamic_axes={"state_seq": {0: "batch"}, "temp_c_horizon": {0: "batch"}},
    opset_version=17,
)
```

### 9.2 在线推理流水线

```text
/leg/status
        │
        ▼
Name-based mapping to T_leg[0..11]
        │
        ▼
Extract q, dq, T for all 12 joints at 500 Hz
        │
        ▼
Ring buffer (L = 2500, D = 36)
        │
        ▼
TensorRT FP16 inference
        │
        ▼
Future temperature horizons (12, H) in Celsius
        │
        ▼
Thermal guard / monitoring (per-joint)
```

在线阶段只能消费 `ros2ws` 中已定义的 `/leg/status` 字段。

### 9.3 热保护逻辑

```python
T_SOFT = 50.0
T_HARD = 60.0


def thermal_protection(pred_temp_c: np.ndarray) -> str:
    """pred_temp_c: (12, H) or (12,) — per-joint max predicted temperature."""
    t_max = float(pred_temp_c.max())
    if t_max >= T_HARD:
        return "HARD_LIMIT"
    if t_max >= T_SOFT:
        return "SOFT_LIMIT"
    return "NORMAL"
```

对关节 `i` 的力矩限幅可按如下方式衰减：

\[
\tau^{eff}_{max}(i) =
\tau^{rated}_{max}(i) \cdot \operatorname{clip}\!\left(
\frac{T_{hard} - \hat{T}^{(i)}_{max}}{T_{hard} - T_{soft}},
0,
1
\right)
\]

---

## 10. 与项目工程的接口

模型文档与工程代码的对应关系建议如下：

| 领域 | 建议文件 |
|:-----|:---------|
| 映射配置 | `configs/leg_index_mapping.yaml` |
| 数据采集 | `scripts/` 下采集或检查脚本 |
| 数据集定义 | `tienkung_thermal/data/` |
| 模型定义 | `tienkung_thermal/models/thermal_lstm.py` |
| 模型超参与代码落地里程碑（方案 A） | `configs/ultra_thermal_lstm.yaml`、`docs/ultra_thermal_lstm_implementation.md` |
| 训练与评估 | `tienkung_thermal/training/` |
| 导出与部署 | `tienkung_thermal/deployment/` |

该文档只定义建模口径，不覆盖具体 Python 包布局实现；**代码落地步骤、目录规划与方案 A 配置策略**见 `docs/ultra_thermal_lstm_implementation.md`。

---

## 11. 版本要求与一致性检查

本文件完成后应始终满足以下一致性约束：

- 正文只讨论 `Ultra` 腿部 `12` 关节。
- 温度监督只使用 `MotorStatus.temperature` 单标量 `°C`。
- 不出现 `temperature[0]` / `temperature[1]` 双通道正文依赖。
- 不出现 `29` 关节、G1、BMS、主板温、风扇等基线输入依赖。
- 输入特征为每关节 `q, dq, T`，12 关节拼接后 `D = 36`。
- 输出为 `(B, 12, H)`，同时预测全部 12 关节。
- 工程网格为 `500 Hz`，`L = 2500`，horizon 步数按 `500 Hz` 计算。
- `error ≠ 0` 帧整帧丢弃，不进入训练或评估。
- 所有 tensor 形状、horizon、验收标准均与 `docs/plan.md` 及 `docs/dataset_leg_status_h5.md` 保持一致。

---

## 附录 A: `T_leg[0..11]` 关节名称速查

| 下标 | 关节名 | 侧别 | 备注 |
|:----:|:-------|:----:|:-----|
| 0 | `hip_roll_l_joint` | 左 | 髋滚转 |
| 1 | `hip_yaw_l_joint` | 左 | 髋偏航 |
| 2 | `hip_pitch_l_joint` | 左 | 髋俯仰 |
| 3 | `knee_pitch_l_joint` | 左 | 膝俯仰 |
| 4 | `ankle_pitch_l_joint` | 左 | 踝俯仰 |
| 5 | `ankle_roll_l_joint` | 左 | 踝滚转 |
| 6 | `hip_roll_r_joint` | 右 | 髋滚转 |
| 7 | `hip_yaw_r_joint` | 右 | 髋偏航 |
| 8 | `hip_pitch_r_joint` | 右 | 髋俯仰 |
| 9 | `knee_pitch_r_joint` | 右 | 膝俯仰 |
| 10 | `ankle_pitch_r_joint` | 右 | 踝俯仰 |
| 11 | `ankle_roll_r_joint` | 右 | 踝滚转 |

## 附录 B: Horizon 时间映射

| Horizon 索引 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:-------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 未来时间（秒） | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 7.0 | 10.0 | 12.0 | 15.0 |
| 未来步数（500 Hz） | 250 | 500 | 1000 | 1500 | 2500 | 3500 | 5000 | 6000 | 7500 |

## 附录 C: 术语约定

- `temperature`: 温度标量，单位 `°C`（HDF5 字段名；即 `MotorStatus.temperature`）
- `q`: 关节位置，单位 `rad`（即 `MotorStatus.pos`）
- `dq`: 关节角速度，单位 `rad/s`（即 `MotorStatus.speed`）
- `error`: 电机故障码（`uint32`），`error ≠ 0` 帧整帧丢弃，仅用于数据质量过滤
- `equal_weight_mae`: 12 关节等权平均 MAE
- `D_per_joint`: 每关节输入特征维度（当前为 3：`q, dq, T`）
- `D`: 总输入维度（`n_joints × D_per_joint = 36`）

*文档结束。若与其它旧稿冲突，以 `docs/plan.md`、`docs/dataset_leg_status_h5.md` 与 `configs/leg_index_mapping.yaml` 为准。*
