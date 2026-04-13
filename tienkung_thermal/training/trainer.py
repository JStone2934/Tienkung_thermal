"""训练循环与评估——对齐 thermal_lstm_modeling.md §5、§7。

用法示例见 scripts/train.py。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

class ThermalLoss(nn.Module):
    """Huber + MAE 组合损失，支持关节级权重。"""

    def __init__(
        self,
        huber_weight: float = 0.5,
        mae_weight: float = 0.5,
        huber_delta: float = 1.0,
        joint_weights: list[float] | None = None,
        n_joints: int = 12,
    ) -> None:
        super().__init__()
        self.huber_weight = huber_weight
        self.mae_weight = mae_weight
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)
        if joint_weights is None:
            joint_weights = [1.0] * n_joints
        self.register_buffer(
            "jw", torch.tensor(joint_weights, dtype=torch.float32)
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        joint_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred, target : (B, H)
        joint_index  : (B,)
        """
        huber = self.huber(pred, target).mean(dim=-1)  # (B,)
        mae = (pred - target).abs().mean(dim=-1)  # (B,)
        per_sample = self.huber_weight * huber + self.mae_weight * mae
        w = self.jw[joint_index]  # (B,)
        return (per_sample * w).sum() / w.sum()


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizon_idx_15s: int = -1,
) -> dict[str, float]:
    """在验证/测试集上计算指标。

    Returns
    -------
    dict 含 val_loss, val_mae_15s_equal_weight, val_max_ae 等。
    """
    model.eval()
    total_ae_sum = 0.0
    total_ae_15s_per_joint = torch.zeros(12, device=device)
    count_per_joint = torch.zeros(12, device=device)
    max_ae = 0.0
    n_samples = 0

    for x, ji, y in loader:
        x, ji, y = x.to(device), ji.to(device), y.to(device)
        pred = model(x, ji)
        ae = (pred - y).abs()
        ae_15s = ae[:, horizon_idx_15s]  # (B,)
        for j in range(12):
            mask = ji == j
            if mask.any():
                total_ae_15s_per_joint[j] += ae_15s[mask].sum()
                count_per_joint[j] += mask.sum()
        batch_max = ae.max().item()
        if batch_max > max_ae:
            max_ae = batch_max
        total_ae_sum += ae.sum().item()
        n_samples += x.size(0)

    safe_count = count_per_joint.clamp(min=1)
    mae_per_joint = total_ae_15s_per_joint / safe_count  # (12,)
    active = count_per_joint > 0
    mae_15s_eq = mae_per_joint[active].mean().item() if active.any() else float("inf")

    return {
        "val_mae_15s_equal_weight": mae_15s_eq,
        "val_mae_per_joint_15s": mae_per_joint.cpu().tolist(),
        "val_max_ae": max_ae,
        "val_n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """与 ultra_thermal_lstm.yaml training / loss 段对齐的训练配置。"""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_T_0: int = 20
    scheduler_T_mult: int = 2
    batch_size: int = 128
    max_epochs: int = 200
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 15
    device: str = "cuda"
    huber_weight: float = 0.5
    mae_weight: float = 0.5
    huber_delta: float = 1.0
    joint_weights: list[float] = field(default_factory=lambda: [1.0] * 12)
    checkpoint_dir: str = "checkpoints"


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Path:
    """完整训练循环，返回最佳 checkpoint 路径。"""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = ThermalLoss(
        huber_weight=cfg.huber_weight,
        mae_weight=cfg.mae_weight,
        huber_delta=cfg.huber_delta,
        joint_weights=cfg.joint_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.scheduler_T_0, T_mult=cfg.scheduler_T_mult
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_ultra_thermal.pt"

    best_gate = float("inf")
    patience = 0

    total_train_batches = len(train_loader)
    log_every = max(1, total_train_batches // 20)  # ~5% 打印一次

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, ji, y in train_loader:
            x, ji, y = x.to(device), ji.to(device), y.to(device)
            pred = model(x, ji)
            loss = criterion(pred, y, ji)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % log_every == 0:
                pct = 100.0 * n_batches / total_train_batches
                avg_so_far = epoch_loss / n_batches
                elapsed_batch = time.time() - t0
                logger.info(
                    "  epoch %d [%5d/%d %5.1f%%] loss=%.4f  elapsed=%.0fs",
                    epoch, n_batches, total_train_batches, pct, avg_so_far, elapsed_batch,
                )

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        metrics = evaluate(model, val_loader, device)
        gate = metrics["val_mae_15s_equal_weight"]
        elapsed = time.time() - t0

        logger.info(
            "epoch %3d | train_loss %.4f | val_mae_15s %.4f°C | max_ae %.2f°C | %.1fs",
            epoch, avg_loss, gate, metrics["val_max_ae"], elapsed,
        )

        if gate < best_gate:
            best_gate = gate
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae_15s": gate,
                },
                best_path,
            )
            logger.info("  ✓ new best %.4f°C → %s", gate, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info("early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    logger.info("training done — best val_mae_15s = %.4f°C", best_gate)
    return best_path
