"""
Activation functions and element-wise nonlinearities.
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class ReLU(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.maximum(x.data, 0)
        return x._make_child(data, ReLU.backward, ctx, 'ReLU')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        return grad_output * Tensor(xp.where(x.data > 0, 1.0, 0.0), device=x.device)


class LeakyReLU(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, negative_slope: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.where(x.data > 0, x.data, negative_slope * x.data)
        result = x._make_child(data, LeakyReLU.backward, ctx, 'LeakyReLU')
        result._negative_slope = negative_slope
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        negative_slope = getattr(grad_output, '_negative_slope', 0.01)
        return grad_output * Tensor(xp.where(x.data > 0, 1.0, negative_slope), device=x.device)


class Sigmoid(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        xp = x.xp
        data = 1.0 / (1.0 + xp.exp(-x.data))
        result = x._make_child(data, Sigmoid.backward, ctx, 'Sigmoid')
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (s,) = ctx.get_saved_tensors()
        return grad_output * s * (1 - s)


class Tanh(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        xp = x.xp
        data = xp.tanh(x.data)
        result = x._make_child(data, Tanh.backward, ctx, 'Tanh')
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (t,) = ctx.get_saved_tensors()
        return grad_output * (1 - t ** 2)


class GELU(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = 0.5 * x.data * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x.data + 0.044715 * xp.power(x.data, 3))))
        return x._make_child(data, GELU.backward, ctx, 'GELU')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        sqrt_2_over_pi = xp.sqrt(2 / xp.pi)
        x3 = x.data ** 3
        tanh_arg = sqrt_2_over_pi * (x.data + 0.044715 * x3)
        tanh_val = xp.tanh(tanh_arg)
        sech2 = 1 - tanh_val ** 2
        grad = 0.5 * (1 + tanh_val) + 0.5 * x.data * sech2 * sqrt_2_over_pi * (1 + 0.134145 * x.data ** 2)
        return grad_output * Tensor(grad, device=x.device)


class Softmax(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, axis: int = -1) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        axis = axis % x.ndim
        x_max = xp.max(x.data, axis=axis, keepdims=True)
        exp_x = xp.exp(x.data - x_max)
        data = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
        result = x._make_child(data, Softmax.backward, ctx, 'Softmax')
        result._softmax_axis = axis
        # Save softmax output for backward
        ctx.saved_tensors.append(result)
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        saved = ctx.get_saved_tensors()
        x = saved[0]  # original input
        s = saved[1]  # softmax output
        axis = getattr(s, '_softmax_axis', -1) % s.ndim
        xp = s.xp
        # Jacobian: s_i * (delta_ij - s_j)
        # Efficient: s * (grad - sum(grad * s, axis))
        s_data = s.data
        g_data = grad_output.data
        sum_term = xp.sum(g_data * s_data, axis=axis, keepdims=True)
        grad = s_data * (g_data - sum_term)
        return Tensor(grad, device=s.device)


class LogSoftmax(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, axis: int = -1) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        axis = axis % x.ndim
        x_max = xp.max(x.data, axis=axis, keepdims=True)
        log_sum_exp = xp.log(xp.sum(xp.exp(x.data - x_max), axis=axis, keepdims=True))
        data = x.data - x_max - log_sum_exp
        result = x._make_child(data, LogSoftmax.backward, ctx, 'LogSoftmax')
        result._logsoftmax_axis = axis
        ctx.saved_tensors.append(result)
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        saved = ctx.get_saved_tensors()
        ls = saved[1]
        axis = getattr(ls, '_logsoftmax_axis', -1) % ls.ndim
        xp = ls.xp
        # Gradient of log_softmax: grad - exp(log_softmax) * sum(grad)
        s = xp.exp(ls.data)
        grad = grad_output.data - s * xp.sum(grad_output.data, axis=axis, keepdims=True)
        return Tensor(grad, device=ls.device)
