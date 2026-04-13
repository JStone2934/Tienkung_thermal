# Ultra 热动力学 LSTM 代码落地实施计划（方案 A）

> **基准建模文档**: `docs/thermal_lstm_modeling.md`  
> **工程总计划**: `docs/plan.md`  
> **配置方案**: **方案 A** — 新增独立 `configs/ultra_thermal_lstm.yaml`，**不**以本文件覆盖或改写现有 `configs/thermal_predictor.yaml`（后者保留旧 MLP 约定，避免混用）。

---

## 1. 目标与验收口径

### 1.1 实现目标

在 `tienkung_thermal` 包内实现与 `thermal_lstm_modeling.md` 第 4 节一致的 **`UltraThermalLSTM`**，供训练、评估与后续 ONNX/TensorRT 导出对齐。

### 1.2 行为与接口约定

| 项目 | 约定 |
|:-----|:-----|
| 输入 `state_seq` | 形状 `(B, L, D)`，`L=100`（与建模文档一致；构造时可配置） |
| 输入 `joint_index` | 形状 `(B,)`，`dtype` 为整型，取值 `0..11`，对应 `T_leg[i]` |
| 输出 | 形状 `(B, H)`，`H=9`，多视距未来温度（°C 尺度与标签一致；模块本身不强制单位常量） |
| 网络拓扑 | `Linear(D→d_proj) + LayerNorm + GELU` → `nn.LSTM`（`num_layers=2`，`batch_first=True`）→ 取最后时间步隐状态 → **12** 个独立小头，每头 `Linear(d_hidden→d_mid) + GELU + Linear(d_mid→H)`，按 `joint_index` 选择输出 |
| 默认超参 | `d_proj=32`，`d_hidden=96`，`n_layers=2`，`dropout=0.10`，`d_mid=64`，`horizon=9`，`n_joints=12`（与建模文档表 4.3 一致） |

### 1.3 实现阶段可测验收

- **形状**: `forward(state_seq, joint_index)` 输出 `(B, H)`。
- **多输入维**: `D ∈ {7, 9, 16, 18}` 与建模文档 3.2–3.3 一致时，构造参数 `input_dim=D` 前向无报错。
- **梯度冒烟**: 随机 batch 上 `loss.backward()` 成功。
- **`joint_index` 边界**: 在文档与代码中明确约定——训练数据断言 `0..11`；推理侧建议与 Deploy 映射一致校验，越界策略（`assert` / 抛错 / `clamp`）在实现 PR 中写死一种并在此文档同步。

### 1.4 延后里程碑（不在本计划「首版」强制）

- ONNX 导出（`opset`、动态 batch、`input_names` 与建模文档 §9 对齐）。
- TensorRT FP16 引擎与时延门控。
- `torch.compile` 等可选优化。

---

## 2. 方案 A：配置文件策略

| 文件 | 作用 |
|:-----|:-----|
| **`configs/ultra_thermal_lstm.yaml`**（新建） | Ultra LSTM 专用：模型超参、`seq_len`、`horizon_steps`、可选 `sample_rate_hz` 等，与 `thermal_lstm_modeling.md` 对齐。 |
| **`configs/thermal_predictor.yaml`**（保留） | 历史 MLP 与旧特征名约定；**不**在本专项中强行合并，避免训练脚本误读维度。 |

**原则**: 训练/推理入口若同时支持多种模型，通过**显式配置路径**或 CLI 参数选择 `ultra_thermal_lstm.yaml`，禁止默认混用两套特征定义。

---

## 3. 目录与文件规划

```text
Tienkung_thermal/
  configs/
    ultra_thermal_lstm.yaml      # 新建（方案 A）
  tienkung_thermal/
    models/
      __init__.py                # 导出 UltraThermalLSTM
      thermal_lstm.py            # UltraThermalLSTM 定义
  tests/
    test_thermal_lstm.py         # 形状、多 D、梯度冒烟（若引入 pytest）
  docs/
    ultra_thermal_lstm_implementation.md  # 本文件
```

与建模文档 §10 建议路径一致：`tienkung_thermal/models/thermal_lstm.py`。

---

## 4. 类设计与 API 规格

### 4.1 类名

`UltraThermalLSTM`（与 `thermal_lstm_modeling.md` §4.4 参考命名一致）。

### 4.2 构造函数参数

全部可通过 YAML 或代码传入，默认值与建模文档表 4.3 一致：

- `input_dim: int` — 基线 `7`；启用邻域温度 `9`；基线+IMU `16`；基线+邻域+IMU `18`。
- `proj_dim: int = 32`
- `hidden_dim: int = 96`
- `num_layers: int = 2`
- `dropout: float = 0.10`
- `mid_dim: int = 64`
- `horizon: int = 9`
- `n_joints: int = 12`

### 4.3 子模块

1. **`input_proj`**: `nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())`
2. **`lstm`**: `nn.LSTM(input_size=proj_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)`
3. **`heads`**: `nn.ModuleList`，长度 `n_joints`，每个元素为 `nn.Sequential(Linear(hidden_dim, mid_dim), nn.GELU(), Linear(mid_dim, horizon))`

### 4.4 `forward`

- 签名: `forward(x: Tensor, joint_index: Tensor) -> Tensor`
- `x`: `(B, L, D)`，`float`
- `joint_index`: `(B,)`，`long`
- 逻辑与建模文档 §4.4 一致：`input_proj` → `lstm` → `h_last = lstm_out[:, -1, :]` → 各 `head(h_last)` 堆叠为 `(B, n_joints, H)` → `gather` 按 `joint_index` 取 `(B, H)`。

**说明**: `stack + gather` 便于与 ONNX 双输入导出一致；若后续需省算力，可增加「仅计算当前 `joint_index` 对应头」的路径，并以单元测试与主路径数值对齐（二选一作为唯一主实现即可）。

---

## 5. `configs/ultra_thermal_lstm.yaml` 内容规划

建议字段（与建模文档对齐，实现时可微调键名但需同步训练脚本）：

```yaml
model:
  class: ultra_thermal_lstm
  input_dim: 7
  proj_dim: 32
  hidden_dim: 96
  num_layers: 2
  dropout: 0.10
  mid_dim: 64
  horizon: 9
  n_joints: 12

sequence:
  seq_len: 100
  sample_rate_hz: 20

horizon_steps: [10, 20, 40, 60, 100, 140, 200, 240, 300]
```

**注意**: **模型类构造函数不强制读取磁盘路径**；YAML 由训练入口或工厂函数加载，避免 `import` 时依赖工作目录。

---

## 6. 可选工厂函数

- `build_ultra_thermal_lstm_from_yaml(path: str | Path) -> UltraThermalLSTM`：解析 `ultra_thermal_lstm.yaml` 的 `model:` 段并构建模块。  
- 可与训练 CLI 同 PR 交付，或作为紧随其后的 PR。

---

## 7. 依赖与开发环境

- **运行时**: `torch` 见 `pyproject.toml` 的 `optional-dependencies.train`（`pip install -e ".[train]"`）。
- **训练 CLI 与 YAML 字段说明**（命令、参数、`train.py` 实际读取的配置键）：见 **`docs/training_ultra_lstm.md`**。
- **测试**: 建议在 `optional-dependencies.dev` 中已含 `ruff`；若使用 `pytest`，在 `dev` 中增加 `pytest` 或在 README 中说明一次性安装命令。

---

## 8. 测试计划

| 用例 | 内容 |
|:-----|:-----|
| 形状 | `B=4, L=100, D=7`，随机 `joint_index ∈ [0,11]`，输出 `(4, 9)` |
| 多 `D` | `D=9, 16, 18`，`input_dim` 与 `x.shape[-1]` 一致 |
| 梯度 | `MSELoss(pred, target).backward()` 无异常 |
| 设备一致性 | `joint_index` 与预测张量同 device（`gather` 要求） |

---

## 9. 实施顺序（里程碑）

| 里程碑 | 内容 |
|:-------|:-----|
| **M1** | 新增 `tienkung_thermal/models/thermal_lstm.py`，实现 `UltraThermalLSTM`（含 docstring：张量形状与 `joint_index` 语义） |
| **M2** | 新增 `tienkung_thermal/models/__init__.py`，按需从包根 `__init__.py` 导出 |
| **M3** | 新增 `configs/ultra_thermal_lstm.yaml`（本文件 §5） |
| **M4** | 新增 `tests/test_thermal_lstm.py`（或等价自检脚本）并文档化 `pip install -e ".[train]"` |
| **M5** | 数据集、训练循环、评估与 ONNX（对齐 `thermal_lstm_modeling.md` §6–§9） |

---

## 10. 风险与注意点

- **ONNX**: LSTM + LayerNorm + GELU 一般在较新 `opset` 下可导出；若遇算子问题，在导出 PR 中单独处理，不阻塞 M1–M4。
- **配置混用**: 必须在 README 或本仓库 `docs/plan.md` 指向处说明：**LSTM 基线以 `ultra_thermal_lstm.yaml` + `thermal_lstm_modeling.md` 为准**。
- **关节顺序**: 所有 `joint_index` 与 `T_leg[0..11]` 必须以 `configs/leg_index_mapping.yaml` 与 `plan.md` 的 Ultra 顺序为准。

---

## 11. 文档维护

- 若建模文档 `thermal_lstm_modeling.md` 中张量形状、默认超参或 horizon 定义变更，应同步更新本文件 §1、§5、§9。
- 若实现上采用「单头路径」替代 `gather`，须更新 §4.4 并保留一种为权威描述。

---

*文档结束。*
