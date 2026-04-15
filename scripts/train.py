#!/usr/bin/env python3
"""Ultra 腿部热 LSTM 训练入口（全关节联合建模）。

完整参数、YAML 字段与数据划分说明见 docs/training_ultra_lstm.md。

用法::

    python scripts/train.py --config configs/ultra_thermal_lstm.yaml

    # 指定 GPU
    python scripts/train.py --config configs/ultra_thermal_lstm.yaml --device cuda:1

    # 调整 stride 和 batch size
    python scripts/train.py --config configs/ultra_thermal_lstm.yaml --stride 25 --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_h5(h5_dir: str, manifest_path: str | None) -> tuple[list[str], list[str], list[str]]:
    """按 manifest split 列划分 h5 路径；无 split 列时按 8:1:1 随机划分。"""
    import csv
    import random

    h5_dir = Path(h5_dir)
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        train, val, test = [], [], []
        unassigned = []
        for r in rows:
            p = r.get("hdf5_path", "")
            if not p or not Path(p).exists():
                continue
            split = (r.get("split") or "").strip().lower()
            if split == "train":
                train.append(p)
            elif split == "val":
                val.append(p)
            elif split == "test":
                test.append(p)
            else:
                unassigned.append(p)
        if unassigned:
            random.seed(42)
            random.shuffle(unassigned)
            n = len(unassigned)
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
            val.extend(unassigned[:n_val])
            test.extend(unassigned[n_val : n_val + n_test])
            train.extend(unassigned[n_val + n_test :])
        return train, val, test

    all_h5 = sorted(h5_dir.glob("*.h5"))
    all_paths = [str(p) for p in all_h5]
    if not all_paths:
        raise FileNotFoundError(f"no .h5 files in {h5_dir}")
    random.seed(42)
    random.shuffle(all_paths)
    n = len(all_paths)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    return (
        all_paths[n_val + n_test :],
        all_paths[:n_val],
        all_paths[n_val : n_val + n_test],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UltraThermalLSTM (joint-coupled)")
    parser.add_argument("--config", default="configs/ultra_thermal_lstm.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="覆盖 YAML training.batch_size",
    )
    parser.add_argument(
        "--seq-len", type=int, default=None, metavar="L",
        help="覆盖 YAML sequence.seq_len",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, metavar="W",
        help="DataLoader 进程数，默认 4",
    )
    parser.add_argument(
        "--stride", type=int, default=None, metavar="S",
        help="滑窗步长（帧数），默认 50（0.1s@500Hz）",
    )
    parser.add_argument(
        "--tensorboard-dir", default="runs", metavar="DIR",
        help="TensorBoard 日志目录，默认 runs；设为 none 可禁用",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("train")

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    seq_cfg = cfg["sequence"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]

    import torch
    from torch.utils.data import DataLoader

    from tienkung_thermal.data.dataset import UltraThermalDataset, N_JOINTS, D_PER_JOINT
    from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM
    from tienkung_thermal.training.trainer import TrainConfig, train

    input_dim = model_cfg.get("input_dim", N_JOINTS * D_PER_JOINT)
    log.info("joint-coupled mode: n_joints=%d  d_per_joint=%d → D=%d", N_JOINTS, D_PER_JOINT, input_dim)

    train_h5, val_h5, test_h5 = _collect_h5(
        data_cfg.get("h5_dir", "data/processed/leg_status_500hz"),
        data_cfg.get("manifest_path"),
    )
    log.info("sessions: train=%d val=%d test=%d", len(train_h5), len(val_h5), len(test_h5))

    horizon_steps = cfg.get("horizon_steps", [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500])
    seq_len = args.seq_len if args.seq_len is not None else seq_cfg.get("seq_len", 2500)
    stride = args.stride if args.stride is not None else seq_cfg.get("stride", 50)
    ds_kwargs = dict(
        seq_len=seq_len,
        horizon_steps=horizon_steps,
        stride=stride,
    )

    train_ds = UltraThermalDataset(train_h5, **ds_kwargs)
    val_ds = UltraThermalDataset(val_h5, **ds_kwargs)
    log.info("samples: train=%d val=%d", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("train dataset is empty — check h5 files and seq_len/horizon")
        sys.exit(1)

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else train_cfg.get("batch_size", 128)
    )
    num_workers = args.num_workers if args.num_workers is not None else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    log.info("seq_len=%d  batch_size=%d  num_workers=%d  stride=%d", seq_len, batch_size, num_workers, stride)

    model = UltraThermalLSTM(
        input_dim=input_dim,
        proj_dim=model_cfg.get("proj_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 96),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.10),
        mid_dim=model_cfg.get("mid_dim", 64),
        horizon=model_cfg.get("horizon", 9),
        n_joints=model_cfg.get("n_joints", 12),
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: UltraThermalLSTM  params=%d  input_dim=%d", n_params, input_dim)

    device = args.device or train_cfg.get("device", "cuda")
    tcfg = TrainConfig(
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        scheduler_T_0=train_cfg.get("scheduler_T_0", 20),
        scheduler_T_mult=train_cfg.get("scheduler_T_mult", 2),
        batch_size=batch_size,
        max_epochs=train_cfg.get("max_epochs", 200),
        grad_clip_max_norm=train_cfg.get("grad_clip_max_norm", 1.0),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 15),
        device=device,
        huber_weight=loss_cfg.get("huber_weight", 0.5),
        mae_weight=loss_cfg.get("mae_weight", 0.5),
        huber_delta=loss_cfg.get("huber_delta", 1.0),
        joint_weights=loss_cfg.get("joint_weights", [1.0] * 12),
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir if args.tensorboard_dir.lower() != "none" else None,
    )

    best_ckpt = train(model, train_loader, val_loader, tcfg)
    log.info("best checkpoint: %s", best_ckpt)


if __name__ == "__main__":
    main()
