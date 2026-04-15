"""训练循环与评估——全关节联合建模版本。

对齐 thermal_lstm_modeling.md §5、§7。
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
    """Huber + MAE 组合损失，支持关节级权重。

    pred, target 均为 (B, 12, H)。
    """

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
    ) -> torch.Tensor:
        """
        pred, target : (B, 12, H)
        """
        huber = self.huber(pred, target).mean(dim=-1)  # (B, 12)
        mae = (pred - target).abs().mean(dim=-1)  # (B, 12)
        per_joint = self.huber_weight * huber + self.mae_weight * mae  # (B, 12)
        weighted = per_joint * self.jw.unsqueeze(0)  # (B, 12)
        return weighted.sum() / (self.jw.sum() * pred.size(0))


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
    dict 含 val_mae_15s_equal_weight, val_mae_per_joint_15s, val_max_ae 等。
    """
    model.eval()
    total_ae_15s_per_joint = torch.zeros(12, device=device)
    n_samples = 0
    max_ae = 0.0

    for x, target in loader:
        x, target = x.to(device), target.to(device)
        pred = model(x)  # (B, 12, H)
        ae = (pred - target).abs()  # (B, 12, H)
        ae_15s = ae[:, :, horizon_idx_15s]  # (B, 12)
        total_ae_15s_per_joint += ae_15s.sum(dim=0)  # (12,)
        batch_max = ae.max().item()
        if batch_max > max_ae:
            max_ae = batch_max
        n_samples += x.size(0)

    mae_per_joint = total_ae_15s_per_joint / max(n_samples, 1)  # (12,)
    mae_15s_eq = mae_per_joint.mean().item()

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
    tensorboard_dir: str | None = "runs"


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

    # TensorBoard
    writer = None
    if cfg.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
        logger.info("tensorboard → %s", cfg.tensorboard_dir)

    best_gate = float("inf")
    patience = 0

    total_train_batches = len(train_loader)
    log_every = max(1, total_train_batches // 20)

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)  # (B, 12, H)
            loss = criterion(pred, target)

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

        if writer:
            writer.add_scalar("loss/train", avg_loss, epoch)
            writer.add_scalar("mae/val_15s_equal_weight", gate, epoch)
            writer.add_scalar("mae/val_max_ae", metrics["val_max_ae"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            joint_names = [
                "hip_roll_l", "hip_yaw_l", "hip_pitch_l",
                "knee_pitch_l", "ankle_pitch_l", "ankle_roll_l",
                "hip_roll_r", "hip_yaw_r", "hip_pitch_r",
                "knee_pitch_r", "ankle_pitch_r", "ankle_roll_r",
            ]
            for i, name in enumerate(joint_names):
                writer.add_scalar(
                    f"mae_per_joint/{name}",
                    metrics["val_mae_per_joint_15s"][i],
                    epoch,
                )
            writer.flush()

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
    if writer:
        writer.close()
    return best_path
