"""UltraThermalLSTM — 因果 LSTM 热预测模型。

架构与超参数约定见 docs/thermal_lstm_modeling.md §4；
配置文件见 configs/ultra_thermal_lstm.yaml。

输入:
    state_seq  (B, L, D)   — 历史特征序列
    joint_index (B,)       — T_leg[0..11] 关节编号
输出:
    (B, H)                 — 未来多视距温度预测 (°C)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UltraThermalLSTM(nn.Module):
    """共享骨干 + 12 关节独立输出头的因果 LSTM。"""

    def __init__(
        self,
        input_dim: int = 9,
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

    def forward(self, x: torch.Tensor, joint_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
        joint_index : (B,) long, 0..n_joints-1

        Returns
        -------
        (B, H) — 多视距未来温度预测
        """
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]  # (B, hidden_dim)
        all_preds = torch.stack(
            [head(h_last) for head in self.heads], dim=1
        )  # (B, n_joints, H)
        idx = joint_index.view(-1, 1, 1).expand(-1, 1, all_preds.size(-1))
        return all_preds.gather(dim=1, index=idx).squeeze(1)  # (B, H)
