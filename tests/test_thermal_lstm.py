"""UltraThermalLSTM 冒烟测试——全关节联合建模版本。

运行: python -m pytest tests/test_thermal_lstm.py -v
或:   python tests/test_thermal_lstm.py
"""

from __future__ import annotations

import unittest

import torch

from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM


class TestUltraThermalLSTM(unittest.TestCase):

    def _make_model(self, input_dim: int = 36, **kw) -> UltraThermalLSTM:
        return UltraThermalLSTM(input_dim=input_dim, **kw)

    def test_output_shape_default(self):
        """D=36, B=4, L=100 → (4, 12, 9)"""
        model = self._make_model(36)
        x = torch.randn(4, 100, 36)
        out = model(x)
        self.assertEqual(out.shape, (4, 12, 9))

    def test_multiple_input_dims(self):
        """不同 input_dim 均可正常前向。"""
        for d in [36, 48, 60]:
            with self.subTest(D=d):
                model = self._make_model(d)
                x = torch.randn(2, 50, d)
                out = model(x)
                self.assertEqual(out.shape, (2, 12, 9))

    def test_gradient_flow(self):
        """loss.backward() 不报错且所有参数有梯度。"""
        model = self._make_model(36)
        x = torch.randn(4, 100, 36)
        target = torch.randn(4, 12, 9)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no gradient")

    def test_all_joints_different_output(self):
        """12 个关节头应产生不同输出。"""
        model = self._make_model(36)
        model.eval()
        x = torch.randn(1, 50, 36)
        with torch.no_grad():
            out = model(x)  # (1, 12, 9)
        # 至少部分关节输出应不同
        self.assertFalse(torch.allclose(out[0, 0], out[0, 6], atol=1e-6))

    def test_full_seq_len(self):
        """L=2500 (完整 5s@500Hz) 不报错。"""
        model = self._make_model(36)
        x = torch.randn(1, 2500, 36)
        out = model(x)
        self.assertEqual(out.shape, (1, 12, 9))

    def test_no_joint_index_required(self):
        """forward 只需要 x，不需要 joint_index。"""
        model = self._make_model(36)
        x = torch.randn(2, 50, 36)
        # Should work with just x
        out = model(x)
        self.assertEqual(out.shape, (2, 12, 9))

    def test_batch_size_one(self):
        """B=1 正常工作。"""
        model = self._make_model(36)
        x = torch.randn(1, 100, 36)
        out = model(x)
        self.assertEqual(out.shape, (1, 12, 9))


if __name__ == "__main__":
    unittest.main()
