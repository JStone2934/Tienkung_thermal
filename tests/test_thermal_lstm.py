"""UltraThermalLSTM 冒烟测试——形状、多输入维度、梯度、设备一致性。

运行: python -m pytest tests/test_thermal_lstm.py -v
或:   python tests/test_thermal_lstm.py
"""

from __future__ import annotations

import unittest

import torch

from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM


class TestUltraThermalLSTM(unittest.TestCase):

    def _make_model(self, input_dim: int = 9, **kw) -> UltraThermalLSTM:
        return UltraThermalLSTM(input_dim=input_dim, **kw)

    def test_output_shape_default(self):
        """D=9, B=4, L=100 → (4, 9)"""
        model = self._make_model(9)
        x = torch.randn(4, 100, 9)
        ji = torch.randint(0, 12, (4,))
        out = model(x, ji)
        self.assertEqual(out.shape, (4, 9))

    def test_output_shape_raw_only(self):
        """D=5 (raw-only) → (B, H)"""
        model = self._make_model(5)
        x = torch.randn(2, 100, 5)
        ji = torch.tensor([0, 11])
        out = model(x, ji)
        self.assertEqual(out.shape, (2, 9))

    def test_multiple_input_dims(self):
        """D ∈ {5, 7, 9, 11, 14, 16, 18, 20}"""
        for d in [5, 7, 9, 11, 14, 16, 18, 20]:
            with self.subTest(D=d):
                model = self._make_model(d)
                x = torch.randn(2, 50, d)
                ji = torch.randint(0, 12, (2,))
                out = model(x, ji)
                self.assertEqual(out.shape, (2, 9))

    def test_gradient_flow(self):
        """loss.backward() 不报错且所有参数有梯度。"""
        model = self._make_model(9)
        x = torch.randn(4, 100, 9)
        ji = torch.randint(0, 12, (4,))
        target = torch.randn(4, 9)
        pred = model(x, ji)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no gradient")

    def test_joint_index_boundary(self):
        """joint_index=0 和 joint_index=11 都正常工作。"""
        model = self._make_model(9)
        x = torch.randn(2, 50, 9)
        for j in [0, 11]:
            ji = torch.full((2,), j, dtype=torch.long)
            out = model(x, ji)
            self.assertEqual(out.shape, (2, 9))

    def test_different_joints_different_output(self):
        """不同关节 index 应产生不同输出（12 头独立）。"""
        model = self._make_model(9)
        model.eval()
        x = torch.randn(1, 50, 9)
        with torch.no_grad():
            out0 = model(x, torch.tensor([0]))
            out6 = model(x, torch.tensor([6]))
        self.assertFalse(torch.allclose(out0, out6, atol=1e-6))

    def test_full_seq_len(self):
        """L=2500 (完整 5s@500Hz) 不报错。"""
        model = self._make_model(9)
        x = torch.randn(1, 2500, 9)
        ji = torch.tensor([3])
        out = model(x, ji)
        self.assertEqual(out.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
