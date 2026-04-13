# UltraThermalLSTM 训练：命令行与配置说明

本文档描述 **`scripts/train.py`** 的用法、命令行参数，以及与 **`configs/ultra_thermal_lstm.yaml`** 的对应关系。建模细节见 `thermal_lstm_modeling.md`；HDF5 格式见 `dataset_leg_status_h5.md`。

---

## 1. 环境与工作目录

- **Python**：建议 `>=3.10`，并安装训练依赖（见仓库根目录 `requirements.txt` 或 `pip install -e ".[train]"`）。
- **工作目录**：在仓库根 **`Tienkung_thermal/`** 下执行命令（下文路径均相对仓库根）。
- **导入路径**：`train.py` 会把仓库根加入 `sys.path`；若从其它目录调用，仍建议 `cd` 到仓库根再运行。
- **GPU**：`L=2500` 时在 CPU 上极慢；生产训练请使用 CUDA，并安装带 CUDA 的 PyTorch 发行版。

---

## 2. 训练入口：`scripts/train.py`

### 2.1 基本命令

```bash
# 默认：读取 configs/ultra_thermal_lstm.yaml，特征为 full（D=9，含派生量）
python scripts/train.py --config configs/ultra_thermal_lstm.yaml
```

### 2.2 命令行参数（CLI）

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `--config` | `configs/ultra_thermal_lstm.yaml` | YAML 配置文件路径。 |
| `--raw-only` | 关闭 | 仅使用原始量特征，**等价于** `features.use_derived: false`，输入维度 **D=5**（不叠加可选特征时）。与配置中 `use_derived: true` 同时存在时，**以 CLI 为准**（即传入 `--raw-only` 则强制 raw-only）。 |
| `--device` | 见下文 | 训练设备，如 `cuda`、`cuda:0`、`cpu`。若省略，则使用 YAML 中 `training.device`（默认 `cuda`）。 |
| `--checkpoint-dir` | `checkpoints` | 保存最佳模型的目录；最佳权重文件名为 **`best_ultra_thermal.pt`**。 |
| `--log-level` | `INFO` | 日志级别，如 `DEBUG`、`INFO`、`WARNING`。 |
| `--batch-size` | 使用 YAML | 正整数；**覆盖** `training.batch_size`。减小可降低 **GPU 显存**占用。 |
| `--seq-len` | 使用 YAML | 正整数；**覆盖** `sequence.seq_len`（输入序列长度 **L**）。对显存影响通常比 batch 更大，试验时可适当减小。 |
| `--num-workers` | `4` | DataLoader 子进程数；主要影响 **CPU 与主机内存**，对 GPU 显存影响很小。 |

**设备解析顺序**：`--device`（若提供）→ 否则 `training.device`。若指定 `cuda` 但当前无 GPU，PyTorch 会在 `trainer.py` 中回退到 **CPU**（仍建议显式使用 GPU 环境训练）。

### 2.3 常用示例

```bash
# Raw-only 消融（D=5）
python scripts/train.py --config configs/ultra_thermal_lstm.yaml --raw-only

# 指定第二块 GPU
python scripts/train.py --config configs/ultra_thermal_lstm.yaml --device cuda:1

# 指定 checkpoint 目录与更详细日志
python scripts/train.py --config configs/ultra_thermal_lstm.yaml \
  --checkpoint-dir runs/exp001 --log-level DEBUG

# 显存紧张：减小 batch，必要时再缩短序列长度做试验
python scripts/train.py --config configs/ultra_thermal_lstm.yaml --raw-only --device cuda \
  --batch-size 32 --seq-len 2500
```

---

## 3. YAML 配置与代码实际使用字段

训练脚本**不会**把整份 YAML 自动注入所有子模块；下表列出 **`train.py` / `trainer.py` 实际读取的键**。未列出的键可能供文档、导出或其它脚本使用，**当前训练循环不会读取**。

### 3.1 `model`（`UltraThermalLSTM` 构造）

| 键 | 默认（代码内） | 说明 |
|:---|:---------------|:-----|
| `proj_dim` | 32 | 输入线性投影维度。 |
| `hidden_dim` | 96 | LSTM hidden size。 |
| `num_layers` | 2 | LSTM 层数。 |
| `dropout` | 0.10 | `num_layers>1` 时作用于 LSTM 层间 dropout。 |
| `mid_dim` | 64 | 各关节 head 的中间层维度。 |
| `horizon` | 9 | 多视距预测头数 **H**（与 `horizon_steps` 长度一致）。 |
| `n_joints` | 12 | 关节数。 |
| `input_dim` | — | YAML 中有注释用途；**实际 `input_dim` 由特征开关动态计算**（见 §3.4），请勿仅改 YAML 中的 `input_dim` 而不改特征开关。 |

### 3.2 `sequence`

| 键 | 默认 | 说明 |
|:---|:-----|:-----|
| `seq_len` | 2500 | 输入序列长度 **L**（帧数，500 Hz 下 2500≈5 s）。 |

### 3.3 顶层 `horizon_steps`

- **类型**：长度为 9 的整数列表（步数，采样率 500 Hz）。
- **默认**（`train.py` 内若 YAML 未写）：`[250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]`。
- **与配置关系**：若 `configs/ultra_thermal_lstm.yaml` 中定义了 `horizon_steps`，训练脚本通过 `cfg.get("horizon_steps", …)` 读取；需与 `model.horizon` 长度一致。

### 3.4 `features`（决定输入维度 D）

| 键 | 说明 |
|:---|:-----|
| `use_derived` | `true`：原始 5 维 + 派生 4 维 → **D=9**（在未启用可选特征时）。`false`：仅原始量 → **D=5**。 |
| `optional_adjacent_temp` | `true` 时 **+2**（同侧邻关节温度特征）。 |
| `optional_imu` | `true` 时 **+9**（IMU：euler 3 + gyro 3 + accel 3）。 |

**CLI `--raw-only`**：为 `True` 时强制 `use_derived=False`，忽略 YAML 中的 `use_derived`。

维度计算公式（与 `train.py` 一致）：

- 基础：`5 + (4 if use_derived else 0)`
- 若 `optional_adjacent_temp`：`+2`
- 若 `optional_imu`：`+9`

### 3.5 `training` → `TrainConfig`

| 键 | 默认 | 说明 |
|:---|:-----|:-----|
| `lr` | 1e-3 | AdamW 学习率。 |
| `weight_decay` | 1e-4 | AdamW weight decay。 |
| `scheduler_T_0` | 20 | `CosineAnnealingWarmRestarts` 的 `T_0`。 |
| `scheduler_T_mult` | 2 | 同上，`T_mult`。 |
| `batch_size` | 128 | `DataLoader` batch size；可用 CLI `--batch-size` 覆盖。`pin_memory=True` 固定开启。 |
| `max_epochs` | 200 | 最大 epoch 数。 |
| `grad_clip_max_norm` | 1.0 | 梯度裁剪阈值。 |
| `early_stopping_patience` | 15 | 验证集监控指标连续不提升的容忍 epoch 数。 |
| `device` | `cuda` | 当未传 `--device` 时使用。 |

### 3.6 `loss` → `ThermalLoss` / `TrainConfig`

| 键 | 默认 | 说明 |
|:---|:-----|:-----|
| `huber_weight` | 0.5 | Huber 项权重。 |
| `mae_weight` | 0.5 | MAE 项权重。 |
| `huber_delta` | 1.0 | Huber 的 `delta`。 |
| `joint_weights` | 12×1.0 | 长度 12 的列表，按 `joint_index` 对样本加权。 |

### 3.7 `data`（数据路径与划分）

| 键 | 说明 |
|:---|:-----|
| `h5_dir` | 目录下所有 `*.h5` 可作为候选；若提供 `manifest_path` 则优先按 manifest 筛选与划分。 |
| `manifest_path` | CSV：需含列 **`hdf5_path`**；可选列 **`split`**，取值为 `train` / `val` / `test`（大小写不敏感）。 |

**划分逻辑**（`train.py` 内 `_collect_h5`）：

- 若 manifest **存在**且能读到行：按 `split` 列分配；**无 `split` 或为空** 的行进入「未分配」池，再按 **8:1:1** 随机划分（`random.seed(42)`）。
- 若 **无 manifest** 或路径无效：对 `h5_dir` 下所有 `*.h5` **随机打乱**（`seed=42`）后按 **8:1:1** 划分为 train / val / test。

测试集路径会被解析，但 **当前 `train.py` 仅构建 train 与 val 的 DataLoader**，不跑 test。

---

## 4. 训练循环行为摘要

- **优化器**：AdamW。  
- **学习率调度**：每个 epoch 结束后 `CosineAnnealingWarmRestarts.step()`。  
- **损失**：`ThermalLoss`（Huber + MAE，按关节权重对 batch 内样本加权）。  
- **早停监控指标**：验证集 **`val_mae_15s_equal_weight`**（`trainer.evaluate` 中定义为各关节在 **15 s 视距** 上 MAE 的等权平均；视距对应 `horizon_steps` **最后一个**元素）。  
- **YAML 中的 `training.early_stopping_monitor` / `acceptance.*`**：当前 **不参与** `trainer.py` 分支逻辑，仅作文档或后续扩展；早停仅以代码中的 `val_mae_15s_equal_weight` 为准。

---

## 5. 输出产物

- **路径**：`{checkpoint_dir}/best_ultra_thermal.pt`（默认 `checkpoints/best_ultra_thermal.pt`）。  
- **内容**：`epoch`、`model_state_dict`、`optimizer_state_dict`、`val_mae_15s`（保存时的门控验证 MAE）。

---

## 6. 相关测试命令（非训练，仅供开发）

```bash
# 建议在仓库根执行；需已安装 dev 依赖（含 pytest）
PYTHONPATH=. pytest tests/test_thermal_lstm.py -q
```

若系统同时加载了 ROS2 的 pytest 插件导致冲突，可尝试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_thermal_lstm.py -q
```

---

## 7. 另见

- `configs/ultra_thermal_lstm.yaml` — 完整默认超参与特征开关。  
- `docs/thermal_lstm_modeling.md` — 特征定义、horizon 与验收口径。  
- `docs/ultra_thermal_lstm_implementation.md` — 实现计划与模块边界。  
- `docs/dataset_leg_status_h5.md` — HDF5 字段与样本构造前提。
