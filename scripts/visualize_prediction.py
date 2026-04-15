#!/usr/bin/env python3
"""可视化推理：在一个 session 上滑窗预测，画真实 vs 预测温度对比曲线。

用法::

    python scripts/visualize_prediction.py \
        --checkpoint checkpoints/best_ultra_thermal.pt \
        --h5 data/processed/leg_status_500hz_0413/rosbag2_2026_04_07-12_14_44.h5 \
        --horizon-idx 4 \
        --output prediction_vs_actual.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

JOINT_NAMES = [
    "hip_roll_l", "hip_yaw_l", "hip_pitch_l",
    "knee_pitch_l", "ankle_pitch_l", "ankle_roll_l",
    "hip_roll_r", "hip_yaw_r", "hip_pitch_r",
    "knee_pitch_r", "ankle_pitch_r", "ankle_roll_r",
]

HORIZON_LABELS = ["0.5s", "1s", "2s", "3s", "5s", "7s", "10s", "12s", "15s"]
HORIZON_STEPS = [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]
JOINT_FIELDS = ("q", "dq", "temperature")
N_JOINTS = 12
SEQ_LEN = 2500
SAMPLE_RATE = 500


def load_session(h5_path: str) -> dict[str, np.ndarray]:
    import h5py
    with h5py.File(h5_path, "r") as f:
        data = {}
        for field in JOINT_FIELDS:
            data[field] = np.asarray(f[f"joints/{field}"], dtype=np.float32)
        data["timestamps"] = np.asarray(f["timestamps"], dtype=np.float64)
    return data


def build_input(data: dict, start_t: int) -> np.ndarray:
    """构建单个输入窗口 (L, 36)。"""
    sl = slice(start_t, start_t + SEQ_LEN)
    cols = []
    for j in range(N_JOINTS):
        for field in JOINT_FIELDS:
            cols.append(data[field][sl, j])
    return np.stack(cols, axis=-1)  # (L, 36)


def main():
    parser = argparse.ArgumentParser(description="Visualize prediction vs actual temperature")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--h5", required=True, help="HDF5 session 文件路径")
    parser.add_argument(
        "--horizon-idx", type=int, default=4,
        help="要可视化的 horizon 索引 (0-8)，默认 4 对应 5s",
    )
    parser.add_argument("--stride", type=int, default=500, help="滑窗步长（帧），默认 500 = 1s")
    parser.add_argument("--output", default="prediction_vs_actual.png", help="输出图片路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--joints", default=None,
        help="要画的关节索引，逗号分隔，如 '2,3,8,9'；默认画全部 12 个",
    )
    args = parser.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UltraThermalLSTM()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"loaded checkpoint: epoch={ckpt.get('epoch')}, val_mae_15s={ckpt.get('val_mae_15s', '?'):.4f}°C")

    # 加载数据
    data = load_session(args.h5)
    n_frames = data["temperature"].shape[0]
    max_horizon = HORIZON_STEPS[args.horizon_idx]
    horizon_label = HORIZON_LABELS[args.horizon_idx]

    max_start = n_frames - SEQ_LEN - max(HORIZON_STEPS)
    if max_start <= 0:
        print(f"session 太短: {n_frames} 帧，需要至少 {SEQ_LEN + max(HORIZON_STEPS)} 帧")
        sys.exit(1)

    # 滑窗推理
    starts = list(range(0, max_start, args.stride))
    print(f"session: {n_frames} frames ({n_frames/SAMPLE_RATE:.1f}s), {len(starts)} windows, horizon={horizon_label}")

    pred_temps = []  # (N_windows, 12)
    actual_temps = []  # (N_windows, 12)
    time_points = []  # 每个预测对应的"目标时刻"

    with torch.no_grad():
        batch_inputs = []
        batch_starts = []
        BATCH = 64

        for start_t in starts:
            x = build_input(data, start_t)
            batch_inputs.append(x)
            batch_starts.append(start_t)

            if len(batch_inputs) == BATCH:
                x_batch = torch.from_numpy(np.stack(batch_inputs)).to(device)
                pred = model(x_batch)  # (B, 12, 9)
                pred_h = pred[:, :, args.horizon_idx].cpu().numpy()  # (B, 12)
                pred_temps.append(pred_h)

                for st in batch_starts:
                    target_t = st + SEQ_LEN + HORIZON_STEPS[args.horizon_idx] - 1
                    actual = data["temperature"][target_t, :]  # (12,)
                    actual_temps.append(actual)
                    time_points.append(data["timestamps"][target_t])

                batch_inputs.clear()
                batch_starts.clear()

        if batch_inputs:
            x_batch = torch.from_numpy(np.stack(batch_inputs)).to(device)
            pred = model(x_batch)
            pred_h = pred[:, :, args.horizon_idx].cpu().numpy()
            pred_temps.append(pred_h)

            for st in batch_starts:
                target_t = st + SEQ_LEN + HORIZON_STEPS[args.horizon_idx] - 1
                actual = data["temperature"][target_t, :]
                actual_temps.append(actual)
                time_points.append(data["timestamps"][target_t])

    pred_all = np.concatenate(pred_temps, axis=0)  # (N, 12)
    actual_all = np.stack(actual_temps)  # (N, 12)
    time_arr = np.array(time_points)
    time_arr = time_arr - time_arr[0]  # 相对时间（秒）

    # 选择要画的关节
    if args.joints:
        joint_indices = [int(x) for x in args.joints.split(",")]
    else:
        joint_indices = list(range(N_JOINTS))

    n_joints_plot = len(joint_indices)

    # 计算 MAE
    mae_per_joint = np.abs(pred_all - actual_all).mean(axis=0)
    mae_total = mae_per_joint[joint_indices].mean()

    # 画图
    n_cols = 2
    n_rows = (n_joints_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f"Temperature Prediction vs Actual  (horizon={horizon_label}, avg MAE={mae_total:.2f}°C)\n"
        f"{Path(args.h5).stem}",
        fontsize=13, y=1.01,
    )

    for idx, ji in enumerate(joint_indices):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.plot(time_arr, actual_all[:, ji], "b-", linewidth=0.8, alpha=0.8, label="actual")
        ax.plot(time_arr, pred_all[:, ji], "r-", linewidth=0.8, alpha=0.8, label=f"pred ({horizon_label})")
        ax.set_ylabel("°C")
        ax.set_title(f"{JOINT_NAMES[ji]}  (MAE={mae_per_joint[ji]:.2f}°C)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_joints_plot, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    axes[-1, 0].set_xlabel("time (s)")
    if n_cols > 1 and axes[-1, 1].get_visible():
        axes[-1, 1].set_xlabel("time (s)")

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nsaved → {args.output}")
    print(f"avg MAE ({horizon_label}): {mae_total:.2f}°C")
    for ji in joint_indices:
        print(f"  {JOINT_NAMES[ji]:20s} MAE={mae_per_joint[ji]:.2f}°C")


if __name__ == "__main__":
    main()
