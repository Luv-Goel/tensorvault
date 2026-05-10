"""Tests for the core Tensor class and autograd engine."""

import math
import numpy as np
import pytest
from tensorvault import Tensor


class TestTensorCreation:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.data.tolist() == [1.0, 2.0, 3.0]

    def test_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = Tensor(arr)
        assert t.shape == (2, 2)
        np.testing.assert_array_equal(t.data, arr)

    def test_zeros(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        assert t.data.sum() == 0.0

    def test_ones(self):
        t = Tensor.ones(3)
        assert t.shape == (3,)
        assert t.data.tolist() == [1.0, 1.0, 1.0]

    def test_randn(self):
        t = Tensor.randn(100, 100)
        assert t.shape == (100, 100)
        assert -5 < t.data.mean() < 5  # sanity check

    def test_scalar(self):
        t = Tensor(42.0)
        assert t.shape == ()
        assert t.item() == 42.0


class TestTensorArithmetic:
    def test_add(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        assert c.data.tolist() == [4.0, 6.0]

    def test_sub(self):
        a = Tensor([5.0, 3.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        assert c.data.tolist() == [4.0, 1.0]

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        assert c.data.tolist() == [8.0, 15.0]

    def test_div(self):
        a = Tensor([10.0, 20.0])
        b = Tensor([2.0, 5.0])
        c = a / b
        assert c.data.tolist() == [5.0, 4.0]

    def test_matmul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.data, expected)

    def test_neg(self):
        a = Tensor([1.0, -2.0, 3.0])
        b = -a
        assert b.data.tolist() == [-1.0, 2.0, -3.0]

    def test_pow(self):
        a = Tensor([2.0, 3.0, 4.0])
        b = a ** 2
        assert b.data.tolist() == [4.0, 9.0, 16.0]

    def test_broadcast_add(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 20.0])
        c = a + b
        assert c.data.tolist() == [[11.0, 22.0], [13.0, 24.0]]


class TestAutograd:
    def test_simple_grad(self):
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.item() - 4.0) < 1e-6

    def test_chain_rule(self):
        x = Tensor([3.0], requires_grad=True)
        y = x ** 2
        z = y + 5
        z.backward()
        # dz/dx = 2x = 6
        assert x.grad is not None
        assert abs(x.grad.item() - 6.0) < 1e-6

    def test_multiple_ops(self):
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y + x ** 2
        z.backward()
        # dz/dx = y + 2x = 3 + 4 = 7
        # dz/dy = x = 2
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.item() - 7.0) < 1e-6
        assert abs(y.grad.item() - 2.0) < 1e-6

    def test_matmul_grad(self):
        x = Tensor([[1.0, 2.0]], requires_grad=True)  # 1×2
        w = Tensor([[3.0], [4.0]], requires_grad=True)  # 2×1
        z = x @ w  # 1×1
        z.backward()
        # dz/dx = w^T = [[3, 4]]
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, [[3.0, 4.0]])

    def test_no_grad_by_default(self):
        x = Tensor([1.0, 2.0])
        y = x + 1
        assert y.grad_fn is None

    def test_detach(self):
        x = Tensor([2.0], requires_grad=True)
        y = x.detach()
        assert y.requires_grad is False
        assert y.grad_fn is None

    def test_zero_grad(self):
        x = Tensor([2.0], requires_grad=True)
        (x ** 2).backward()
        assert x.grad is not None
        x.zero_grad()
        assert x.grad is None

    def test_reuse_intermediate(self):
        """Test that gradients flow through reused intermediate values."""
        x = Tensor([2.0], requires_grad=True)
        h = x ** 2  # 4
        y = h + h  # 8
        y.backward()
        # dy/dx = 2*2x = 4x = 8
        assert x.grad is not None
        assert abs(x.grad.item() - 8.0) < 1e-6


class TestLossFunctions:
    def test_mse(self):
        from tensorvault.nn import MSELoss
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.5, 3.5])
        loss = MSELoss()(pred, target)
        expected = ((0.5**2 + 0.5**2 + 0.5**2) / 3)
        assert abs(loss.item() - expected) < 1e-6

    def test_cross_entropy(self):
        from tensorvault.nn import CrossEntropyLoss
        pred = Tensor([[2.0, 1.0, 0.1],
                       [1.0, 3.0, 0.1]])
        target = Tensor([0, 1])
        loss = CrossEntropyLoss()(pred, target)
        assert loss.item() > 0  # loss should be positive

    def test_mse_backward(self):
        from tensorvault.nn import MSELoss
        pred = Tensor([1.0, 2.0], requires_grad=True)
        target = Tensor([3.0, 4.0])
        loss = MSELoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        # d/dx (1/2 * (x-t)^2) = (x-t)
        # d/dx (1/n * sum(x-t)^2) = 2*(x-t)/n
        n = 2
        expected = np.array([-4.0, -4.0]) / n
        np.testing.assert_array_almost_equal(pred.grad, expected)


class TestNNModules:
    def test_linear_forward(self):
        from tensorvault.nn import Linear
        layer = Linear(4, 3)
        x = Tensor.randn(10, 4)
        out = layer(x)
        assert out.shape == (10, 3)

    def test_linear_backward(self):
        from tensorvault.nn import Linear
        layer = Linear(4, 3)
        x = Tensor.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 4)

    def test_sequential(self):
        from tensorvault.nn import Sequential, Linear, ReLU
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )
        x = Tensor.randn(3, 10)
        out = model(x)
        assert out.shape == (3, 2)

    def test_relu(self):
        from tensorvault.nn import ReLU
        relu = ReLU()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        out = relu(x)
        assert out.data.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_relu_backward(self):
        from tensorvault.nn import ReLU
        relu = ReLU()
        x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
        out = relu(x)
        out.sum().backward()
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, [0.0, 0.0, 1.0])

    def test_dropout_training(self):
        from tensorvault.nn import Dropout
        dropout = Dropout(p=0.5)
        dropout.train()
        x = Tensor.ones(1000)
        out = dropout(x)
        # ~50% should be zero, ~50% should be 2.0 (scale factor)
        zeros = (out.data == 0).sum()
        assert 400 < zeros < 600  # rough check

    def test_dropout_eval(self):
        from tensorvault.nn import Dropout
        dropout = Dropout(p=0.5)
        dropout.eval()
        x = Tensor.ones(100)
        out = dropout(x)
        np.testing.assert_array_equal(out.data, np.ones(100))


class TestOptimizers:
    def test_sgd_step(self):
        from tensorvault.nn import Linear
        from tensorvault.optim import SGD
        model = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = model(x)
        out.sum().backward()
        lr = 0.01
        opt = SGD(model.parameters(), lr=lr)
        old_w = model.weight.data.copy()
        opt.step()
        # weight should have been updated
        assert not np.allclose(model.weight.data, old_w)

    def test_adam_step(self):
        from tensorvault.nn import Linear
        from tensorvault.optim import Adam
        model = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = model(x)
        out.sum().backward()
        opt = Adam(model.parameters(), lr=0.001)
        old_w = model.weight.data.copy()
        opt.step()
        assert not np.allclose(model.weight.data, old_w)

    def test_zero_grad(self):
        from tensorvault.nn import Linear
        from tensorvault.optim import SGD
        model = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = model(x)
        out.sum().backward()
        opt = SGD(model.parameters(), lr=0.01)
        opt.step()
        opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None or np.allclose(p.grad, 0)


class TestSerialization:
    def test_save_load(self, tmp_path):
        from tensorvault.nn import Sequential, Linear, ReLU
        model = Sequential(
            Linear(5, 3),
            ReLU(),
            Linear(3, 1),
        )
        path = str(tmp_path / "model.json")
        model.save(path)
        model2 = Sequential()
        model2.load(path)
        x = Tensor.randn(2, 5)
        out1 = model(x)
        out2 = model2(x)
        np.testing.assert_array_almost_equal(out1.data, out2.data)

    def test_model_parameters(self):
        from tensorvault.nn import Sequential, Linear
        model = Sequential(
            Linear(3, 2),
            Linear(2, 1),
        )
        params = list(model.parameters())
        assert len(params) >= 4  # weight + bias for each layer


class TestMathOps:
    def test_exp(self):
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        y = x.exp()
        np.testing.assert_array_almost_equal(y.data, np.exp([0, 1, 2]))
        y.sum().backward()
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, np.exp([0, 1, 2]))

    def test_log(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.log()
        np.testing.assert_array_almost_equal(y.data, np.log([1, 2, 3]))
        y.sum().backward()
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, 1.0 / np.array([1, 2, 3]))

    def test_sqrt(self):
        x = Tensor([4.0, 9.0, 16.0], requires_grad=True)
        y = x.sqrt()
        np.testing.assert_array_almost_equal(y.data, [2.0, 3.0, 4.0])

    def test_mean(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = x.mean()
        assert abs(y.item() - 2.5) < 1e-6
        y.backward()
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_sum(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        assert abs(y.item() - 6.0) < 1e-6
        y.backward()
        np.testing.assert_array_almost_equal(x.grad, [1.0, 1.0, 1.0])

    def test_softmax(self):
        from tensorvault.ops.activations import softmax
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = softmax(x)
        assert abs(y.data.sum() - 1.0) < 1e-6
