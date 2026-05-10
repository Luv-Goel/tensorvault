"""
Shape manipulation operations: Reshape, Transpose, Flatten
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class Reshape(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, shape: tuple) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = x.data.reshape(shape)
        return x._make_child(data, Reshape.backward, ctx, 'Reshape')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return grad_output.reshape(x.shape)


class Transpose(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, axes: tuple) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.transpose(x.data, axes)
        # Store inverse axes for backward
        inv_axes = tuple(axes.index(i) for i in range(len(axes)))
        result = x._make_child(data, Transpose.backward, ctx, 'Transpose')
        result._transpose_inv = inv_axes
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        inv_axes = getattr(grad_output, '_transpose_inv', tuple(range(grad_output.ndim)))
        return grad_output.transpose(inv_axes)


class Flatten(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        shape = x.shape
        end_dim = end_dim if end_dim >= 0 else len(shape) + end_dim
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]
        new_shape = shape[:start_dim] + (flattened_size,) + shape[end_dim + 1:]
        data = x.data.reshape(new_shape)
        result = x._make_child(data, Flatten.backward, ctx, 'Flatten')
        result._orig_shape = shape
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return grad_output.reshape(x.shape)
