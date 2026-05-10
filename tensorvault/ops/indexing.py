"""
Indexing operations: Slice, Gather
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class Slice(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, key) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = x.data[key]
        result = x._make_child(data, Slice.backward, ctx, 'Slice')
        result._slice_key = key
        result._slice_input_shape = x.shape
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        key = getattr(grad_output, '_slice_key', None)
        input_shape = getattr(grad_output, '_slice_input_shape', x.shape)
        grad = xp.zeros(input_shape, dtype=x.data.dtype)
        if key is not None:
            # Handle different key types
            if isinstance(key, tuple):
                # Normalize key
                norm_key = []
                for k in key:
                    if isinstance(k, slice):
                        norm_key.append(k)
                    elif isinstance(k, int):
                        norm_key.append(k)
                    elif isinstance(k, Tensor):
                        norm_key.append(k.data)
                    else:
                        norm_key.append(k)
                grad[tuple(norm_key)] = grad_output.data
            else:
                grad[key] = grad_output.data
        else:
            grad = grad_output.data.reshape(input_shape)
        return Tensor(grad, device=x.device)


class Gather(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, dim: int, index) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        if isinstance(index, Tensor):
            index = index.data
        data = xp.take_along_axis(x.data, index, axis=dim)
        result = x._make_child(data, Gather.backward, ctx, 'Gather')
        result._gather_dim = dim
        result._gather_index = index
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        xp = x.xp
        dim = getattr(grad_output, '_gather_dim', 0)
        index = getattr(grad_output, '_gather_index', None)
        grad = xp.zeros_like(x.data)
        if index is not None:
            xp.put_along_axis(grad, index, grad_output.data, axis=dim)
        return Tensor(grad, device=x.device)
