"""
Basic arithmetic operations: Neg, Abs, Add, Sub, Mul, Div, MatMul.
"""

from ..tensor import Tensor, _Context, _Function
import numpy as np


class Neg(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data = -x.data
        return x._make_child(data, Neg.backward, ctx, 'Neg')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        return -grad_output


class Abs(_Function):
    @staticmethod
    def forward(ctx: _Context, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        data = abs(x.data)
        return x._make_child(data, Abs.backward, ctx, 'Abs')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> Tensor:
        (x,) = ctx.get_saved_tensors()
        return grad_output * Tensor.sign(x.data)


class Add(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        data = a.data + b.data
        return a._make_child(data, Add.backward, ctx, 'Add')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        grad_a = grad_output
        grad_b = grad_output
        # Handle broadcasting: sum over broadcast dimensions
        if a.shape != grad_a.shape:
            axis = tuple(range(-grad_a.ndim + a.ndim, 0)) if a.ndim < grad_a.ndim else None
            if axis:
                grad_a = grad_a.sum(axis=axis, keepdims=False)
            # Sum over broadcast dimensions
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
            for i in range(a.ndim):
                if a.shape[i] == 1 and grad_a.shape[i] != 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            # Squeeze remaining singletons
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
        if b.shape != grad_b.shape:
            axis = tuple(range(-grad_b.ndim + b.ndim, 0)) if b.ndim < grad_b.ndim else None
            if axis:
                grad_b = grad_b.sum(axis=axis, keepdims=False)
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
            for i in range(b.ndim):
                if b.shape[i] == 1 and grad_b.shape[i] != 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
        return grad_a, grad_b


class Sub(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        data = a.data - b.data
        return a._make_child(data, Sub.backward, ctx, 'Sub')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        grad_a = grad_output
        grad_b = -grad_output
        # Handle broadcasting
        if a.shape != grad_a.shape:
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
            for i in range(a.ndim):
                if a.shape[i] == 1 and grad_a.shape[i] != 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
        if b.shape != grad_b.shape:
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
            for i in range(b.ndim):
                if b.shape[i] == 1 and grad_b.shape[i] != 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
        return grad_a, grad_b


class Mul(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        data = a.data * b.data
        return a._make_child(data, Mul.backward, ctx, 'Mul')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        grad_a = grad_output * b.detach()
        grad_b = grad_output * a.detach()
        # Handle broadcasting
        if a.shape != grad_a.shape:
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
            for i in range(a.ndim):
                if a.shape[i] == 1 and grad_a.shape[i] != 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
        if b.shape != grad_b.shape:
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
            for i in range(b.ndim):
                if b.shape[i] == 1 and grad_b.shape[i] != 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
        return grad_a, grad_b


class Div(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        data = a.data / b.data
        return a._make_child(data, Div.backward, ctx, 'Div')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        xp = a.xp
        grad_a = grad_output / b.detach()
        grad_b = -grad_output * a.detach() / (b.detach() ** 2)
        # Handle broadcasting
        if a.shape != grad_a.shape:
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
            for i in range(a.ndim):
                if a.shape[i] == 1 and grad_a.shape[i] != 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            while grad_a.ndim > a.ndim:
                grad_a = grad_a.sum(axis=0, keepdims=False)
        if b.shape != grad_b.shape:
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
            for i in range(b.ndim):
                if b.shape[i] == 1 and grad_b.shape[i] != 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            while grad_b.ndim > b.ndim:
                grad_b = grad_b.sum(axis=0, keepdims=False)
        return grad_a, grad_b


class MatMul(_Function):
    @staticmethod
    def forward(ctx: _Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        xp = a.xp
        data = xp.matmul(a.data, b.data)
        return a._make_child(data, MatMul.backward, ctx, 'MatMul')

    @staticmethod
    def backward(ctx: _Context, grad_output: Tensor) -> tuple:
        a, b = ctx.get_saved_tensors()
        xp = a.xp
        grad_a = grad_output @ b.detach().T
        grad_b = a.detach().T @ grad_output
        return grad_a, grad_b
