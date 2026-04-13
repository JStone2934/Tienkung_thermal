"""归一化统计量：计算、保存、加载。

流程：
1. 遍历训练集 HDF5，对每个特征维度收集 Welford 在线统计量。
2. 对 log1p_fields（如 ddq_abs、tau_sq）先做 log1p 再统计。
3. 保存为 JSON（含 mean、std、log1p_fields 列表），供 Dataset 和推理复用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOG1P_FIELDS_DEFAULT = ("ddq_abs", "tau_sq")

RAW_FIELDS = ("q", "dq", "current", "temperature", "voltage")
DERIVED_FIELDS = ("tau_est", "tau_sq", "dq_abs", "ddq_abs")


def _ordered_feature_names(
    use_derived: bool,
    use_adjacent_temp: bool = False,
    use_imu: bool = False,
) -> list[str]:
    """返回与 Dataset.__getitem__ 拼接顺序一致的特征名列表。"""
    names = list(RAW_FIELDS)
    if use_derived:
        names.extend(DERIVED_FIELDS)
    if use_adjacent_temp:
        names.extend(["adj_temp_prev", "adj_temp_next"])
    if use_imu:
        names.extend([f"imu_{i}" for i in range(9)])
    return names


def compute_norm_stats(
    h5_paths: list[str] | list[Path],
    use_derived: bool = True,
    use_adjacent_temp: bool = False,
    use_imu: bool = False,
    log1p_fields: tuple[str, ...] = LOG1P_FIELDS_DEFAULT,
) -> dict:
    """用 Welford 在线算法遍历训练集 HDF5，返回 {mean, std, log1p_fields, feature_names}。

    统计粒度：所有关节合并（global），每个特征维度独立。
    """
    import h5py

    feat_names = _ordered_feature_names(use_derived, use_adjacent_temp, use_imu)
    D = len(feat_names)
    log1p_set = set(log1p_fields)
    log1p_indices = [i for i, n in enumerate(feat_names) if n in log1p_set]

    count = np.zeros(D, dtype=np.float64)
    mean = np.zeros(D, dtype=np.float64)
    m2 = np.zeros(D, dtype=np.float64)

    raw_fields = list(RAW_FIELDS)
    derived_fields = list(DERIVED_FIELDS) if use_derived else []

    for path in h5_paths:
        with h5py.File(str(path), "r") as f:
            n_frames = f["timestamps"].shape[0]
            n_joints = 12
            for j in range(n_joints):
                cols: list[np.ndarray] = []
                for field in raw_fields:
                    cols.append(np.asarray(f[f"joints/{field}"][:, j], dtype=np.float64))
                for field in derived_fields:
                    cols.append(np.asarray(f[f"joints/{field}"][:, j], dtype=np.float64))
                if use_adjacent_temp:
                    cols.append(np.zeros(n_frames, dtype=np.float64))
                    cols.append(np.zeros(n_frames, dtype=np.float64))
                if use_imu and "imu" in f:
                    imu = np.concatenate([
                        np.asarray(f["imu/euler"], dtype=np.float64),
                        np.asarray(f["imu/angular_velocity"], dtype=np.float64),
                        np.asarray(f["imu/linear_acceleration"], dtype=np.float64),
                    ], axis=-1)
                    for k in range(9):
                        cols.append(imu[:, k])

                data = np.stack(cols, axis=-1)  # (n_frames, D)
                for idx in log1p_indices:
                    data[:, idx] = np.log1p(np.abs(data[:, idx]))

                for i in range(D):
                    col = data[:, i]
                    valid = np.isfinite(col)
                    col = col[valid]
                    n = len(col)
                    if n == 0:
                        continue
                    for val in col:
                        count[i] += 1
                        delta = val - mean[i]
                        mean[i] += delta / count[i]
                        delta2 = val - mean[i]
                        m2[i] += delta * delta2

    std = np.sqrt(m2 / np.maximum(count, 1))
    std = np.maximum(std, 1e-8)

    logger.info("norm stats computed over %d files, feature dims=%d", len(h5_paths), D)
    for i, name in enumerate(feat_names):
        logger.info("  %-16s  mean=%10.4f  std=%10.4f  n=%d", name, mean[i], std[i], int(count[i]))

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "log1p_fields": list(log1p_fields),
        "feature_names": feat_names,
    }


def save_norm_stats(stats: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("norm stats saved → %s", p)
    return p


def load_norm_stats(path: str | Path) -> dict:
    with open(path) as f:
        stats = json.load(f)
    logger.info("norm stats loaded ← %s", path)
    return stats


def stats_to_tensors(stats: dict) -> dict[str, torch.Tensor]:
    """将 JSON dict 转为 Dataset 需要的 {mean: (D,), std: (D,)} float32 Tensor。"""
    return {
        "mean": torch.tensor(stats["mean"], dtype=torch.float32),
        "std": torch.tensor(stats["std"], dtype=torch.float32),
    }
