"""
Math operations: Sqrt, Exp, Log, Sin, Cos, Pow
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class Sqrt(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.sqrt(x.data)
        return x._make_child(data, Sqrt.backward, ctx, 'Sqrt')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        return grad_output / (2 * Tensor.sqrt(x.detach()) + 1e-8)


class Exp(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.exp(x.data)
        result = x._make_child(data, Exp.backward, ctx, 'Exp')
        # Save the result for backward reuse
        ctx.saved_tensors.append(result)
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        saved = ctx.get_saved_tensors()
        return grad_output * saved[1]  # saved[1] is the result of exp


class Log(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.log(x.data + 1e-8)
        return x._make_child(data, Log.backward, ctx, 'Log')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return grad_output / (x.detach() + 1e-8)


class Sin(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.sin(x.data)
        return x._make_child(data, Sin.backward, ctx, 'Sin')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return grad_output * Tensor.cos(x.detach())


class Cos(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.cos(x.data)
        return x._make_child(data, Cos.backward, ctx, 'Cos')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return -grad_output * Tensor.sin(x.detach())


class Pow(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        xp = a.xp
        data = xp.power(a.data, b.data)
        return a._make_child(data, Pow.backward, ctx, 'Pow')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        xp = a.xp
        grad_a = grad_output * b.detach() * (a.detach() ** (b.detach() - 1))
        grad_b = grad_output * (a.detach() ** b.detach()) * Tensor.log(a.detach())
        return grad_a, grad_b
