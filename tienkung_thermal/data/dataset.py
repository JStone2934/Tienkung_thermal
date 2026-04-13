"""UltraThermalDataset — 从 leg_status_500hz HDF5 构建训练样本。

HDF5 格式见 docs/dataset_leg_status_h5.md；
特征定义见 docs/thermal_lstm_modeling.md §3.2。

每个样本为 (x, joint_index, target)：
    x       : (L, D)   — 历史特征窗口
    joint_index : int   — T_leg[i]
    target  : (H,)     — 未来多视距温度 (°C)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_LEFT = list(range(6))
_RIGHT = list(range(6, 12))


def _same_side_neighbors(joint_idx: int) -> tuple[int, int]:
    """返回同侧 (prev, next) 关节下标；边界镜像最近有效邻居。"""
    side = _LEFT if joint_idx < 6 else _RIGHT
    pos = side.index(joint_idx)
    prev = side[max(pos - 1, 0)]
    nxt = side[min(pos + 1, len(side) - 1)]
    return prev, nxt


class _SessionCache:
    """单个 session 的内存缓存：将 HDF5 中需要的字段一次性读入 numpy 数组。"""

    __slots__ = ("path", "n_frames", "joints", "imu")

    def __init__(
        self,
        path: str,
        fields: tuple[str, ...],
        load_imu: bool,
    ) -> None:
        import h5py

        self.path = path
        with h5py.File(path, "r") as f:
            self.n_frames = f["timestamps"].shape[0]
            self.joints: dict[str, np.ndarray] = {}
            for field in fields:
                self.joints[field] = np.asarray(
                    f[f"joints/{field}"], dtype=np.float32
                )
            if load_imu and "imu" in f:
                self.imu: np.ndarray | None = np.concatenate(
                    [
                        np.asarray(f["imu/euler"], dtype=np.float32),
                        np.asarray(f["imu/angular_velocity"], dtype=np.float32),
                        np.asarray(f["imu/linear_acceleration"], dtype=np.float32),
                    ],
                    axis=-1,
                )
            else:
                self.imu = None


class UltraThermalDataset(Dataset):
    """滑动窗口数据集：单关节 × 单起始帧 = 一个样本。

    初始化时将所有 session 的关节数据一次性读入内存（float32），
    避免在 __getitem__ 中反复打开 HDF5。

    Parameters
    ----------
    h5_paths : 一个或多个 session HDF5 路径
    seq_len : 输入窗口长度（帧数）
    horizon_steps : 各视距对应的未来步数列表
    use_derived : 是否拼接派生量（tau_est, tau_sq, dq_abs, ddq_abs）
    use_adjacent_temp : 是否拼接同侧邻域温度 (+2)
    use_imu : 是否拼接 IMU 9 维 (+9)
    norm_stats : {"mean": Tensor, "std": Tensor} 或 None
    """

    RAW_FIELDS = ("q", "dq", "current", "temperature", "voltage")
    DERIVED_FIELDS = ("tau_est", "tau_sq", "dq_abs", "ddq_abs")

    def __init__(
        self,
        h5_paths: list[str] | list[Path],
        seq_len: int = 2500,
        horizon_steps: list[int] | None = None,
        use_derived: bool = True,
        use_adjacent_temp: bool = False,
        use_imu: bool = False,
        norm_stats: dict | None = None,
        stride: int = 1,
    ) -> None:
        if horizon_steps is None:
            horizon_steps = [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500]
        self.seq_len = seq_len
        self.horizon_steps = horizon_steps
        self.max_horizon = max(horizon_steps)
        self.use_derived = use_derived
        self.use_adjacent_temp = use_adjacent_temp
        self.use_imu = use_imu
        self.norm_stats = norm_stats
        self.stride = max(1, stride)

        fields = list(self.RAW_FIELDS)
        if use_derived:
            fields.extend(self.DERIVED_FIELDS)

        self._caches: list[_SessionCache] = []
        self._session_info: list[tuple[int, int]] = []  # (cache_idx, n_windows)
        self._n_joints = 12
        self._cum_start: list[int] = []  # 每个 session 的起始 offset
        total = 0

        for path in h5_paths:
            path_str = str(path)
            cache = _SessionCache(path_str, tuple(fields), load_imu=use_imu)
            cache_idx = len(self._caches)
            self._caches.append(cache)

            valid_len = cache.n_frames - seq_len - self.max_horizon
            if valid_len <= 0:
                continue
            n_windows = (valid_len + self.stride - 1) // self.stride
            self._cum_start.append(total)
            self._session_info.append((cache_idx, n_windows))
            total += n_windows * self._n_joints

        self._total = total

    def __len__(self) -> int:
        return self._total

    @property
    def input_dim(self) -> int:
        """根据当前配置推算输入维度 D。"""
        d = len(self.RAW_FIELDS)
        if self.use_derived:
            d += len(self.DERIVED_FIELDS)
        if self.use_adjacent_temp:
            d += 2
        if self.use_imu:
            d += 9
        return d

    def _resolve_index(self, idx: int) -> tuple[int, int, int]:
        """将全局 idx 映射为 (cache_idx, joint_idx, start_t)。"""
        import bisect
        si = bisect.bisect_right(self._cum_start, idx) - 1
        local = idx - self._cum_start[si]
        cache_idx, n_windows = self._session_info[si]
        joint_idx = local // n_windows
        window_idx = local % n_windows
        start_t = window_idx * self.stride
        return cache_idx, joint_idx, start_t

    def __getitem__(self, idx: int):
        cache_idx, joint_idx, start_t = self._resolve_index(idx)
        cache = self._caches[cache_idx]
        sl = slice(start_t, start_t + self.seq_len)

        feature_cols: list[np.ndarray] = []
        for field in self.RAW_FIELDS:
            feature_cols.append(cache.joints[field][sl, joint_idx])

        if self.use_derived:
            for field in self.DERIVED_FIELDS:
                feature_cols.append(cache.joints[field][sl, joint_idx])

        if self.use_adjacent_temp:
            prev_idx, next_idx = _same_side_neighbors(joint_idx)
            feature_cols.append(cache.joints["temperature"][sl, prev_idx])
            feature_cols.append(cache.joints["temperature"][sl, next_idx])

        x = np.stack(feature_cols, axis=-1)

        if self.use_imu and cache.imu is not None:
            x = np.concatenate([x, cache.imu[sl]], axis=-1)

        target_idx = start_t + self.seq_len
        target = np.array(
            [cache.joints["temperature"][target_idx + h - 1, joint_idx] for h in self.horizon_steps],
            dtype=np.float32,
        )

        x_t = torch.from_numpy(x)
        if self.norm_stats is not None:
            x_t = (x_t - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)

        return x_t, torch.tensor(joint_idx, dtype=torch.long), torch.from_numpy(target)
