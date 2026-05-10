"""
Manipulation operations: Cat, Stack, Pad
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class Cat(_Function):
    @staticmethod
    def forward(ctx: _Context, tensors: list, dim: int = 0) -> Tensor:
        ctx.save_for_backward(*tensors)
        xp = tensors[0].xp
        data = xp.concatenate([t.data for t in tensors], axis=dim)
        result = tensors[0]._make_child(data, Cat.backward, ctx, 'Cat')
        result._cat_dim = dim
        result._cat_shapes = [t.shape for t in tensors]
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> list:
        saved = ctx.get_saved_tensors()
        dim = getattr(grad_output, '_cat_dim', 0)
        shapes = getattr(grad_output, '_cat_shapes', [t.shape for t in saved])
        xp = grad_output.xp
        grads = []
        start = 0
        for shape in shapes:
            end = start + shape[dim]
            slices = [slice(None)] * grad_output.ndim
            slices[dim] = slice(start, end)
            grad_slice = grad_output.data[tuple(slices)]
            grads.append(Tensor(grad_slice.copy(), device=grad_output.device))
            start = end
        return tuple(grads)


class Stack(_Function):
    @staticmethod
    def forward(ctx: _Context, tensors: list, dim: int = 0) -> Tensor:
        ctx.save_for_backward(*tensors)
        xp = tensors[0].xp
        # Expand dims and concat
        expanded = [t.data.reshape(list(t.shape[:dim]) + [1] + list(t.shape[dim:])) for t in tensors]
        data = xp.concatenate(expanded, axis=dim)
        result = tensors[0]._make_child(data, Stack.backward, ctx, 'Stack')
        result._stack_dim = dim
        result._stack_shapes = [t.shape for t in tensors]
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> list:
        saved = ctx.get_saved_tensors()
        dim = getattr(grad_output, '_stack_dim', 0)
        shapes = getattr(grad_output, '_stack_shapes', [t.shape for t in saved])
        grads = []
        for i, shape in enumerate(shapes):
            slices = [slice(None)] * grad_output.ndim
            slices[dim] = slice(i, i + 1)
            grad_slice = grad_output.data[tuple(slices)]
            # Squeeze the stacked dim
            grad_slice = grad_slice.reshape(shape)
            grads.append(Tensor(grad_slice.copy(), device=grad_output.device))
        return tuple(grads)


class Pad(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor, pad_width, mode='constant', constant_values=0) -> Tensor:
        ctx.save_for_backward(x)
        xp = x.xp
        data = xp.pad(x.data, pad_width, mode=mode, constant_values=constant_values)
        result = x._make_child(data, Pad.backward, ctx, 'Pad')
        result._pad_width = pad_width
        return result

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        pad_width = getattr(grad_output, '_pad_width', None)
        if pad_width is None:
            return grad_output
        xp = grad_output.xp
        # Unpad the gradient
        slices = []
        for p in pad_width:
            if isinstance(p, int):
                slices.append(slice(p, -p if p > 0 else None))
            elif isinstance(p, (tuple, list)):
                slices.append(slice(p[0], -p[1] if p[1] > 0 else None))
        return Tensor(grad_output.data[tuple(slices)].copy(), device=grad_output.device)
