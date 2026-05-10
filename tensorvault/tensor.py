"""
Core Tensor class with automatic differentiation (reverse-mode autograd).

Supports CPU (NumPy) and GPU (CuPy) backends. Builds a computational graph
on-the-fly for gradient computation.
"""

import numpy as np
from typing import List, Optional, Tuple, Union, Callable, Self, Any

# Try optional GPU backend
try:
    import cupy as cp
    _gpu_available = True
except ImportError:
    cp = np
    _gpu_available = False


def _get_array_module(arr):
    """Return cupy or numpy depending on the array type."""
    if _gpu_available and isinstance(arr, cp.ndarray):
        return cp
    return np


def _to_numpy(arr):
    """Convert to numpy array regardless of backend."""
    xp = _get_array_module(arr)
    if xp is np:
        return arr
    return cp.asnumpy(arr)


def _to_backend(arr, xp):
    """Convert array to the given backend module."""
    if xp is np:
        return _to_numpy(arr) if hasattr(arr, 'get') else np.array(arr)
    return cp.array(arr)


# ─── Graph global state ──────────────────────────────────────────────
_GRAD_ENABLED = True


def set_grad_enabled(enabled: bool):
    global _GRAD_ENABLED
    _GRAD_ENABLED = enabled


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


# ─── Gradient function registry ──────────────────────────────────────

class _Context:
    """Stores saved tensors needed for backward computation."""
    def __init__(self):
        self.saved_tensors: List['Tensor'] = []

    def save_for_backward(self, *tensors: 'Tensor'):
        self.saved_tensors.extend(tensors)

    def get_saved_tensors(self):
        return self.saved_tensors


class _Function:
    """Base class for differentiable operations."""
    @staticmethod
    def forward(ctx: _Context, *inputs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: _Context, grad_output: Any) -> Tuple[Any, ...]:
        raise NotImplementedError


# ─── Tensor ──────────────────────────────────────────────────────────

class Tensor:
    """Multi-dimensional array with automatic differentiation.

    Args:
        data: numpy array, cupy array, list, scalar, or another Tensor.
        requires_grad: If True, gradients are computed for this tensor.
        device: 'cpu' or 'cuda'. Auto-detected if None.
        dtype: Data type (default: float32).
    """
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        device: Optional[str] = None,
        dtype=None,
    ):
        if isinstance(data, Tensor):
            data = data.data.copy()

        if isinstance(data, (int, float, complex, np.integer, np.floating)):
            data = np.array(data)

        # Determine backend
        if _gpu_available and isinstance(data, cp.ndarray):
            xp = cp
            self._device = 'cuda'
        else:
            if isinstance(data, np.ndarray):
                xp = np
            else:
                data = np.array(data)
                xp = np
            self._device = 'cpu'

        if device is not None:
            self._device = device
            if device == 'cuda' and _gpu_available:
                data = cp.array(data)
                xp = cp
            else:
                data = _to_numpy(data) if _gpu_available and isinstance(data, cp.ndarray) else np.array(data)
                xp = np

        if dtype is not None:
            data = data.astype(dtype)
        elif isinstance(data, np.ndarray) and data.dtype.kind not in ('f',):
            data = data.astype(np.float32)
        elif _gpu_available and isinstance(data, cp.ndarray) and data.dtype.kind not in ('f',):
            data = data.astype(cp.float32)

        self.data = data
        self.requires_grad = requires_grad and _GRAD_ENABLED
        self.grad: Optional[np.ndarray] = None
        self._ctx: Optional[_Context] = None
        self._backward_fn: Optional[Callable] = None
        self._is_leaf = True
        self._grad_fn_name: Optional[str] = None

    # ── Properties ───────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def grad_fn(self) -> Optional[str]:
        return self._grad_fn_name

    @property
    def xp(self):
        """Return the array module (numpy or cupy) for this tensor."""
        return _get_array_module(self.data)

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def zeros(cls, *shape, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.zeros(shape, dtype=dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def ones(cls, *shape, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.ones(shape, dtype=dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def randn(cls, *shape, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.random.randn(*shape).astype(dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def rand(cls, *shape, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.random.rand(*shape).astype(dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def arange(cls, start, stop=None, step=1, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.arange(start, stop, step).astype(dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def eye(cls, n, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.eye(n, dtype=dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def empty(cls, *shape, requires_grad=False, device='cpu', dtype=None):
        xp = cp if device == 'cuda' and _gpu_available else np
        data = xp.empty(shape, dtype=dtype or xp.float32)
        return cls(data, requires_grad=requires_grad, device=device)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad=False, device='cpu'):
        if device == 'cuda' and _gpu_available:
            arr = cp.array(arr)
        return cls(arr, requires_grad=requires_grad, device=device)

    @classmethod
    def tensor(cls, data, requires_grad=False, device='cpu', dtype=None):
        return cls(data, requires_grad=requires_grad, device=device, dtype=dtype)

    # ── Internal helpers ─────────────────────────────────────────

    def _make_child(self, data, backward_fn, ctx: _Context, grad_fn_name: str) -> 'Tensor':
        """Create a new tensor that tracks gradients through this op."""
        requires_grad = self.requires_grad or (
            hasattr(self, '_child_requires_grad') and self._child_requires_grad
        )
        # Check all saved tensors
        for t in ctx.saved_tensors:
            if isinstance(t, Tensor) and t.requires_grad:
                requires_grad = True

        child = Tensor(
            data,
            requires_grad=requires_grad,
            device=self.device,
        )
        if requires_grad:
            child._is_leaf = False
            child._backward_fn = backward_fn
            child._ctx = ctx
            child._grad_fn_name = grad_fn_name
        return child

    def _zero_grad(self):
        self.grad = None

    def _is_cuda(self) -> bool:
        return self._device == 'cuda' and _gpu_available

    # ── Convert to numpy ─────────────────────────────────────────

    def numpy(self) -> np.ndarray:
        """Return a numpy copy of the data."""
        return _to_numpy(self.data).copy()

    def item(self):
        """Return scalar value (for 0-d tensors)."""
        return _to_numpy(self.data).item()

    def to(self, device: str) -> 'Tensor':
        """Move tensor to device ('cpu' or 'cuda')."""
        if device == self._device:
            return self
        if device == 'cuda' and _gpu_available:
            new_data = cp.array(_to_numpy(self.data))
        else:
            new_data = _to_numpy(self.data)
        t = Tensor(new_data, requires_grad=self.requires_grad, device=device)
        if self.grad is not None:
            t.grad = _to_numpy(self.grad) if device == 'cpu' else cp.array(self.grad)
        return t

    def cpu(self) -> 'Tensor':
        return self.to('cpu')

    def cuda(self) -> 'Tensor':
        return self.to('cuda')

    # ── Copy / detach ────────────────────────────────────────────

    def detach(self) -> 'Tensor':
        """Return a new tensor with the same data but no gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    def clone(self) -> 'Tensor':
        """Return a new tensor with the same data and gradient tracking."""
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self.device)
        if self.requires_grad:
            t._is_leaf = False
            t._backward_fn = self._backward_fn
            t._ctx = self._ctx
            t._grad_fn_name = self._grad_fn_name
        return t

    # ── Gradient computation ─────────────────────────────────────

    def backward(self, gradient: Optional['Tensor'] = None):
        """Compute gradients via reverse-mode automatic differentiation.

        Args:
            gradient: Initial gradient (for non-scalar outputs). If None,
                     assumes the tensor is a scalar.
        """
        if not self.requires_grad and self._is_leaf:
            return

        if gradient is None:
            if self.shape == ():
                gradient = Tensor(1.0, device=self.device)
            else:
                raise RuntimeError(
                    "backward() can only be called on scalar tensors "
                    "without a gradient argument. "
                    f"Got shape {self.shape}"
                )

        # Topological order
        topo: List[Tensor] = []
        visited = set()
        self._build_topo(topo, visited)
        topo.reverse()

        # Initialize gradients to zero
        for t in topo:
            if t.requires_grad:
                if t._is_leaf:
                    t.grad = None
                else:
                    t.grad = None

        # Set the initial gradient
        self.grad = gradient.data.copy() if isinstance(gradient, Tensor) else gradient.copy()

        # Backprop
        for t in topo:
            if t._is_leaf or t._backward_fn is None:
                continue
            if t.grad is None:
                continue

            grad_tensor = Tensor(t.grad, requires_grad=False, device=t.device)
            try:
                grad_inputs = t._backward_fn(t._ctx, grad_tensor)
            except Exception as e:
                raise RuntimeError(
                    f"Error in backward of {t._grad_fn_name}: {e}"
                ) from e

            if not isinstance(grad_inputs, (tuple, list)):
                grad_inputs = (grad_inputs,)

            saved = t._ctx.get_saved_tensors() if t._ctx else []
            for i, st in enumerate(saved):
                if isinstance(st, Tensor) and st.requires_grad:
                    gi = grad_inputs[i] if i < len(grad_inputs) else None
                    if gi is not None:
                        g_data = gi.data if isinstance(gi, Tensor) else gi
                        if st.grad is None:
                            st.grad = g_data.copy()
                        else:
                            xp = _get_array_module(g_data)
                            st.grad = xp.add(st.grad, g_data)

        # Clean up intermediate gradients
        for t in topo:
            if not t._is_leaf and t.requires_grad:
                t.grad = None

    def _build_topo(self, topo: List['Tensor'], visited: set):
        if id(self) in visited:
            return
        visited.add(id(self))
        if self._backward_fn is not None and self._ctx is not None:
            for st in self._ctx.get_saved_tensors():
                if isinstance(st, Tensor):
                    st._build_topo(topo, visited)
        topo.append(self)

    # ── Zero grad ────────────────────────────────────────────────

    def zero_grad(self):
        self.grad = None

    # ── String representations ───────────────────────────────────

    def __repr__(self) -> str:
        xp = self.xp
        grad_str = f", grad={self.grad.shape}" if self.grad is not None else ""
        req_str = ", requires_grad=True" if self.requires_grad else ""
        dev_str = f", device='{self.device}'" if self._is_cuda() else ""
        data_str = str(_to_numpy(self.data).round(4))
        if len(data_str) > 80:
            data_str = data_str[:77] + "..."
        return f"Tensor({data_str}{req_str}{dev_str}{grad_str})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.data)

    # ── Boolean conversion ───────────────────────────────────────

    def __bool__(self) -> bool:
        if self.size == 1:
            return bool(self.item())
        raise ValueError("The truth value of a Tensor with more than one element is ambiguous")

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    # ================================================================
    #  OPERATIONS
    # ================================================================

    # ── Unary ops ────────────────────────────────────────────────

    def __neg__(self) -> 'Tensor':
        from .ops.basic import Neg
        return Neg.apply(self)

    def __abs__(self) -> 'Tensor':
        return self.abs()

    def abs(self) -> 'Tensor':
        from .ops.basic import Abs
        return Abs.apply(self)

    def sqrt(self) -> 'Tensor':
        from .ops.math import Sqrt
        return Sqrt.apply(self)

    def exp(self) -> 'Tensor':
        from .ops.math import Exp
        return Exp.apply(self)

    def log(self) -> 'Tensor':
        from .ops.math import Log
        return Log.apply(self)

    def sin(self) -> 'Tensor':
        from .ops.math import Sin
        return Sin.apply(self)

    def cos(self) -> 'Tensor':
        from .ops.math import Cos
        return Cos.apply(self)

    def tanh(self) -> 'Tensor':
        from .ops.activations import Tanh
        return Tanh.apply(self)

    def sigmoid(self) -> 'Tensor':
        from .ops.activations import Sigmoid
        return Sigmoid.apply(self)

    def relu(self) -> 'Tensor':
        from .ops.activations import ReLU
        return ReLU.apply(self)

    # ── Binary ops ───────────────────────────────────────────────

    def __add__(self, other) -> 'Tensor':
        from .ops.basic import Add
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return Add.apply(self, other)

    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other) -> 'Tensor':
        from .ops.basic import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return Sub.apply(self, other)

    def __rsub__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return other.__sub__(self)

    def __mul__(self, other) -> 'Tensor':
        from .ops.basic import Mul
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return Mul.apply(self, other)

    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Tensor':
        from .ops.basic import Div
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return Div.apply(self, other)

    def __rtruediv__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return other.__truediv__(self)

    def __pow__(self, other) -> 'Tensor':
        from .ops.math import Pow
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return Pow.apply(self, other)

    def __rpow__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return other.__pow__(self)

    def __matmul__(self, other) -> 'Tensor':
        from .ops.basic import MatMul
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return MatMul.apply(self, other)

    def __iadd__(self, other) -> 'Tensor':
        result = self.__add__(other)
        self.data = result.data
        return self

    def __isub__(self, other) -> 'Tensor':
        result = self.__sub__(other)
        self.data = result.data
        return self

    def __imul__(self, other) -> 'Tensor':
        result = self.__mul__(other)
        self.data = result.data
        return self

    def __idiv__(self, other) -> 'Tensor':
        result = self.__truediv__(other)
        self.data = result.data
        return self

    # ── Comparison (no grad) ─────────────────────────────────────

    def __eq__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data == other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    def __ne__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data != other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    def __lt__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data < other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    def __le__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data <= other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    def __gt__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data > other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    def __ge__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        data = (self.data >= other.data)
        return Tensor(data, requires_grad=False, device=self.device)

    # ── Shape ops ────────────────────────────────────────────────

    def reshape(self, *shape) -> 'Tensor':
        from .ops.shape import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape)

    def transpose(self, *axes) -> 'Tensor':
        from .ops.shape import Transpose
        if len(axes) == 0:
            axes = (1, 0)
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Transpose.apply(self, axes)

    @property
    def T(self) -> 'Tensor':
        if self.ndim <= 1:
            return self
        return self.transpose()

    def flatten(self, start_dim=0, end_dim=-1) -> 'Tensor':
        from .ops.shape import Flatten
        return Flatten.apply(self, start_dim, end_dim)

    # ── Reduction ops ────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        from .ops.reductions import Sum
        return Sum.apply(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        from .ops.reductions import Mean
        return Mean.apply(self, axis, keepdims)

    def max(self, axis=None, keepdims=False) -> 'Tensor':
        from .ops.reductions import Max
        return Max.apply(self, axis, keepdims)

    def min(self, axis=None, keepdims=False) -> 'Tensor':
        from .ops.reductions import Min
        return Min.apply(self, axis, keepdims)

    # ── Softmax ──────────────────────────────────────────────────

    def softmax(self, axis=-1) -> 'Tensor':
        from .ops.activations import Softmax
        return Softmax.apply(self, axis)

    def log_softmax(self, axis=-1) -> 'Tensor':
        from .ops.activations import LogSoftmax
        return LogSoftmax.apply(self, axis)

    # ── Indexing ─────────────────────────────────────────────────

    def __getitem__(self, key):
        from .ops.indexing import Slice
        return Slice.apply(self, key)

    # ── Utility ──────────────────────────────────────────────────

    def fill_(self, value):
        """Fill tensor in-place (no gradient tracking)."""
        xp = self.xp
        self.data = xp.full_like(self.data, value)
        return self

    def zero_(self):
        return self.fill_(0.0)

    def copy_(self, other: 'Tensor'):
        """Copy data from another tensor in-place."""
        self.data = other.data.copy()
        return self

    # ── Gather / Scatter ─────────────────────────────────────────

    def gather(self, dim, index):
        from .ops.indexing import Gather
        return Gather.apply(self, dim, index)

    # ── Cat / Stack ──────────────────────────────────────────────

    @staticmethod
    def cat(tensors: List['Tensor'], dim=0) -> 'Tensor':
        from .ops.manipulation import Cat
        return Cat.apply(tensors, dim)

    @staticmethod
    def stack(tensors: List['Tensor'], dim=0) -> 'Tensor':
        from .ops.manipulation import Stack
        return Stack.apply(tensors, dim)

    # ── Pad ──────────────────────────────────────────────────────

    def pad(self, pad_width, mode='constant', constant_values=0):
        from .ops.manipulation import Pad
        return Pad.apply(self, pad_width, mode, constant_values)


# ── Convenience alias ────────────────────────────────────────────────

def tensor(data, requires_grad=False, device='cpu', dtype=None) -> Tensor:
    return Tensor.tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)
