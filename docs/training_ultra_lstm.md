# UltraThermalLSTM 训练：命令行与配置说明

本文档描述 **`scripts/train.py`** 的用法、命令行参数，以及与 **`configs/ultra_thermal_lstm.yaml`** 的对应关系。建模细节见 `thermal_lstm_modeling.md`；HDF5 格式见 `dataset_leg_status_h5.md`。

---

## 1. 环境与工作目录

- **Python**：建议 `>=3.10`，并安装训练依赖（见仓库根目录 `requirements.txt` 或 `pip install -e ".[train]"`）。
- **工作目录**：在仓库根 **`Tienkung_thermal/`** 下执行命令（下文路径均相对仓库根）。
- **GPU**：`L=2500` 时在 CPU 上极慢；生产训练请使用 CUDA。

---

## 2. 训练入口：`scripts/train.py`

### 2.1 基本命令

```bash
# 全关节联合建模（D=36，12 joints × 3 features）
python scripts/train.py --config configs/ultra_thermal_lstm.yaml
```

### 2.2 命令行参数（CLI）

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `--config` | `configs/ultra_thermal_lstm.yaml` | YAML 配置文件路径。 |
| `--device` | 见下文 | 训练设备，如 `cuda`、`cuda:0`、`cpu`。若省略，则使用 YAML 中 `training.device`（默认 `cuda`）。 |
| `--checkpoint-dir` | `checkpoints` | 保存最佳模型的目录；最佳权重文件名为 **`best_ultra_thermal.pt`**。 |
| `--log-level` | `INFO` | 日志级别，如 `DEBUG`、`INFO`、`WARNING`。 |
| `--batch-size` | 使用 YAML | 正整数；**覆盖** `training.batch_size`。 |
| `--seq-len` | 使用 YAML | 正整数；**覆盖** `sequence.seq_len`（输入序列长度 **L**）。 |
| `--num-workers` | `4` | DataLoader 子进程数。 |
| `--stride` | 使用 YAML | 滑窗步长（帧数），默认 50（0.1s@500Hz）。 |

### 2.3 常用示例

```bash
# 指定第二块 GPU
python scripts/train.py --config configs/ultra_thermal_lstm.yaml --device cuda:1

# 指定 checkpoint 目录与更详细日志
python scripts/train.py --config configs/ultra_thermal_lstm.yaml \
  --checkpoint-dir runs/exp001 --log-level DEBUG

# 调整 stride 和 batch size
python scripts/train.py --config configs/ultra_thermal_lstm.yaml \
  --stride 25 --batch-size 64
```

---

## 3. YAML 配置与代码实际使用字段

### 3.1 `model`（`UltraThermalLSTM` 构造）

| 键 | 默认（代码内） | 说明 |
|:---|:---------------|:-----|
| `input_dim` | 36 | 12 joints × 3 features (q, dq, T)。 |
| `proj_dim` | 32 | 输入线性投影维度。 |
| `hidden_dim` | 96 | LSTM hidden size。 |
| `num_layers` | 2 | LSTM 层数。 |
| `dropout` | 0.10 | `num_layers>1` 时作用于 LSTM 层间 dropout。 |
| `mid_dim` | 64 | 各关节 head 的中间层维度。 |
| `horizon` | 9 | 多视距预测头数 **H**（与 `horizon_steps` 长度一致）。 |
| `n_joints` | 12 | 关节数。 |

### 3.2 `sequence`

| 键 | 默认 | 说明 |
|:---|:-----|:-----|
| `seq_len` | 2500 | 输入序列长度 **L**（帧数，500 Hz 下 2500≈5 s）。 |
| `stride` | 50 | 滑窗步长（帧数），50 = 0.1s@500Hz。 |

### 3.3 顶层 `horizon_steps`

- **类型**：长度为 9 的整数列表（步数，采样率 500 Hz）。
- **默认**：`[250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]`。

### 3.4 `features`

全关节联合建模下，每关节使用 `q, dq, T` 三个特征，12 关节按 `T_leg[0..11]` 顺序 joint-major 拼接。

### 3.5 `training` → `TrainConfig`

| 键 | 默认 | 说明 |
|:---|:-----|:-----|
| `lr` | 1e-3 | AdamW 学习率。 |
| `weight_decay` | 1e-4 | AdamW weight decay。 |
| `scheduler_T_0` | 20 | `CosineAnnealingWarmRestarts` 的 `T_0`。 |
| `scheduler_T_mult` | 2 | 同上，`T_mult`。 |
| `batch_size` | 128 | `DataLoader` batch size；可用 CLI `--batch-size` 覆盖。 |
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
| `joint_weights` | 12×1.0 | 长度 12 的列表，按关节加权损失。 |

### 3.7 `data`（数据路径与划分）

| 键 | 说明 |
|:---|:-----|
| `h5_dir` | 目录下所有 `*.h5` 可作为候选。 |
| `manifest_path` | CSV：需含列 **`hdf5_path`**；可选列 **`split`**。 |

---

## 4. 训练循环行为摘要

- **输入**：`(B, L, 36)` — 12 关节 × 3 特征 (q, dq, T)。
- **输出**：`(B, 12, H)` — 全部 12 关节的多视距温度预测。
- **优化器**：AdamW。
- **学习率调度**：每个 epoch 结束后 `CosineAnnealingWarmRestarts.step()`。
- **损失**：`ThermalLoss`（Huber + MAE，按关节权重加权）。
- **早停监控指标**：验证集 **`val_mae_15s_equal_weight`**（12 关节在 15 s 视距上 MAE 的等权平均）。
- **进度日志**：每 5% batch 打印一次进度。

---

## 5. 输出产物

- **路径**：`{checkpoint_dir}/best_ultra_thermal.pt`（默认 `checkpoints/best_ultra_thermal.pt`）。
- **内容**：`epoch`、`model_state_dict`、`optimizer_state_dict`、`val_mae_15s`。

---

## 6. 相关测试命令

```bash
PYTHONPATH=. pytest tests/test_thermal_lstm.py -q
```

---

## 7. 另见

- `configs/ultra_thermal_lstm.yaml` — 完整默认超参。
- `docs/thermal_lstm_modeling.md` — 特征定义、horizon 与验收口径。
- `docs/ultra_thermal_lstm_implementation.md` — 实现计划与模块边界。
- `docs/dataset_leg_status_h5.md` — HDF5 字段与样本构造前提。
