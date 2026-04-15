"""UltraThermalDataset — 全关节联合建模，从 leg_status_500hz HDF5 构建训练样本。

HDF5 格式见 docs/dataset_leg_status_h5.md；
特征定义见 docs/thermal_lstm_modeling.md §3.2。

每个样本为 (x, target)：
    x       : (L, 36)    — 12 关节 × 3 特征 (q, dq, T) 的历史窗口
    target  : (12, H)    — 12 关节的未来多视距温度 (°C)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

JOINT_FIELDS = ("q", "dq", "temperature")
N_JOINTS = 12
D_PER_JOINT = len(JOINT_FIELDS)


class _SessionCache:
    """单个 session 的内存缓存：将 HDF5 中需要的字段一次性读入 numpy 数组。"""

    __slots__ = ("path", "n_frames", "joints")

    def __init__(self, path: str) -> None:
        import h5py

        self.path = path
        with h5py.File(path, "r") as f:
            self.n_frames = f["timestamps"].shape[0]
            self.joints: dict[str, np.ndarray] = {}
            for field in JOINT_FIELDS:
                self.joints[field] = np.asarray(
                    f[f"joints/{field}"], dtype=np.float32
                )  # (N, 12)


class UltraThermalDataset(Dataset):
    """全关节联合滑动窗口数据集：单起始帧 = 一个样本。

    初始化时将所有 session 的关节数据一次性读入内存（float32），
    避免在 __getitem__ 中反复打开 HDF5。

    Parameters
    ----------
    h5_paths : 一个或多个 session HDF5 路径
    seq_len : 输入窗口长度（帧数）
    horizon_steps : 各视距对应的未来步数列表
    stride : 滑窗步长（帧数），默认 50（0.1s@500Hz）
    norm_stats : {"mean": Tensor, "std": Tensor} 或 None
    """

    def __init__(
        self,
        h5_paths: list[str] | list[Path],
        seq_len: int = 2500,
        horizon_steps: list[int] | None = None,
        stride: int = 50,
        norm_stats: dict | None = None,
    ) -> None:
        if horizon_steps is None:
            horizon_steps = [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]
        self.seq_len = seq_len
        self.horizon_steps = horizon_steps
        self.max_horizon = max(horizon_steps)
        self.stride = max(1, stride)
        self.norm_stats = norm_stats

        self._caches: list[_SessionCache] = []
        self._session_info: list[tuple[int, int]] = []  # (cache_idx, n_windows)
        self._cum_start: list[int] = []
        total = 0

        for path in h5_paths:
            cache = _SessionCache(str(path))
            cache_idx = len(self._caches)
            self._caches.append(cache)

            valid_len = cache.n_frames - seq_len - self.max_horizon
            if valid_len <= 0:
                continue
            n_windows = (valid_len + self.stride - 1) // self.stride
            self._cum_start.append(total)
            self._session_info.append((cache_idx, n_windows))
            total += n_windows

        self._total = total

    def __len__(self) -> int:
        return self._total

    @property
    def input_dim(self) -> int:
        return N_JOINTS * D_PER_JOINT

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        """将全局 idx 映射为 (cache_idx, start_t)。"""
        import bisect
        si = bisect.bisect_right(self._cum_start, idx) - 1
        local = idx - self._cum_start[si]
        cache_idx, _ = self._session_info[si]
        start_t = local * self.stride
        return cache_idx, start_t

    def __getitem__(self, idx: int):
        cache_idx, start_t = self._resolve_index(idx)
        cache = self._caches[cache_idx]
        sl = slice(start_t, start_t + self.seq_len)

        # (L, 36): joint-major order — [q0, dq0, T0, q1, dq1, T1, ...]
        cols: list[np.ndarray] = []
        for j in range(N_JOINTS):
            for field in JOINT_FIELDS:
                cols.append(cache.joints[field][sl, j])
        x = np.stack(cols, axis=-1)  # (L, 36)

        # (12, H): future temperature for all joints
        target_idx = start_t + self.seq_len
        target = np.stack(
            [
                np.array(
                    [cache.joints["temperature"][target_idx + h - 1, j]
                     for h in self.horizon_steps],
                    dtype=np.float32,
                )
                for j in range(N_JOINTS)
            ]
        )  # (12, H)

        x_t = torch.from_numpy(x)
        if self.norm_stats is not None:
            x_t = (x_t - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)

        return x_t, torch.from_numpy(target)
