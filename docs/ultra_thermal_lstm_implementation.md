# Ultra 热动力学 LSTM 代码落地实施计划（全关节联合建模）

> **基准建模文档**: `docs/thermal_lstm_modeling.md`  
> **工程总计划**: `docs/plan.md`  
> **配置方案**: 独立 `configs/ultra_thermal_lstm.yaml`，**不**以本文件覆盖或改写现有 `configs/thermal_predictor.yaml`（后者保留旧 MLP 约定，避免混用）。

---

## 1. 目标与验收口径

### 1.1 实现目标

在 `tienkung_thermal` 包内实现与 `thermal_lstm_modeling.md` 第 4 节一致的 **`UltraThermalLSTM`**（全关节联合输入/联合预测），供训练、评估与后续 ONNX/TensorRT 导出对齐。

### 1.2 行为与接口约定

| 项目 | 约定 |
|:-----|:-----|
| 输入 `state_seq` | 形状 `(B, L, 36)`，12 关节 × 3 特征 (q, dq, T)，joint-major 拼接 |
| 输出 | 形状 `(B, 12, H)`，`H=9`，全部 12 关节的多视距未来温度（°C） |
| 网络拓扑 | `Linear(36→d_proj) + LayerNorm + GELU` → `nn.LSTM`（`num_layers=2`，`batch_first=True`）→ 取最后时间步隐状态 → **12** 个独立小头，每头 `Linear(d_hidden→d_mid) + GELU + Linear(d_mid→H)`，全部输出 stack 为 `(B, 12, H)` |
| 默认超参 | `input_dim=36`，`d_proj=32`，`d_hidden=96`，`n_layers=2`，`dropout=0.10`，`d_mid=64`，`horizon=9`，`n_joints=12`（与建模文档表 4.3 一致） |

### 1.3 实现阶段可测验收

- **形状**: `forward(state_seq)` 输出 `(B, 12, H)`。
- **多输入维**: `D ∈ {36, 48, 60}` 时，构造参数 `input_dim=D` 前向无报错。
- **梯度冒烟**: 随机 batch 上 `loss.backward()` 成功。
- **全关节输出**: 12 个头输出不同值（独立头验证）。

### 1.4 延后里程碑

- ONNX 导出（`opset`、动态 batch、`input_names` 与建模文档 §9 对齐）。
- TensorRT FP16 引擎与时延门控。

---

## 2. 配置文件策略

| 文件 | 作用 |
|:-----|:-----|
| **`configs/ultra_thermal_lstm.yaml`** | Ultra LSTM 专用：`input_dim=36`、`seq_len`、`horizon_steps`、`stride` 等 |
| **`configs/thermal_predictor.yaml`**（保留） | 历史 MLP 与旧特征名约定；不在本专项中强行合并 |

---

## 3. 目录与文件规划

```text
Tienkung_thermal/
  configs/
    ultra_thermal_lstm.yaml
  tienkung_thermal/
    models/
      __init__.py
      thermal_lstm.py            # UltraThermalLSTM (input_dim=36, output (B,12,H))
    data/
      dataset.py                 # UltraThermalDataset: (x[L,36], target[12,H])
    training/
      trainer.py                 # 训练循环与 evaluate()
  scripts/
    train.py                     # 训练入口
  tests/
    test_thermal_lstm.py         # 形状、梯度冒烟
```

---

## 4. 类设计与 API 规格

### 4.1 类名

`UltraThermalLSTM`

### 4.2 构造函数参数

- `input_dim: int = 36` — 12 joints × 3 features (q, dq, T)
- `proj_dim: int = 32`
- `hidden_dim: int = 96`
- `num_layers: int = 2`
- `dropout: float = 0.10`
- `mid_dim: int = 64`
- `horizon: int = 9`
- `n_joints: int = 12`

### 4.3 `forward`

- 签名: `forward(x: Tensor) -> Tensor`
- `x`: `(B, L, D)`，`float`，`D=36`
- 返回: `(B, 12, H)`
- 逻辑：`input_proj` → `lstm` → `h_last = lstm_out[:, -1, :]` → 各 `head(h_last)` 堆叠为 `(B, 12, H)` 直接返回

---

## 5. 测试计划

| 用例 | 内容 |
|:-----|:-----|
| 形状 | `B=4, L=100, D=36`，输出 `(4, 12, 9)` |
| 多 `D` | `D=36, 48, 60`，`input_dim` 与 `x.shape[-1]` 一致 |
| 梯度 | `MSELoss(pred, target).backward()` 无异常 |
| 全关节独立 | 12 个头输出不同值 |

---

## 6. 风险与注意点

- **ONNX**: LSTM + LayerNorm + GELU 一般在较新 `opset` 下可导出。
- **配置混用**: LSTM 基线以 `ultra_thermal_lstm.yaml` + `thermal_lstm_modeling.md` 为准。
- **关节顺序**: `T_leg[0..11]` 必须以 `configs/leg_index_mapping.yaml` 与 `plan.md` 的 Ultra 顺序为准。

---

*文档结束。*
