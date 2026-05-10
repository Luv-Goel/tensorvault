"""
Normalization layers: BatchNorm, LayerNorm, Dropout.
"""

from ..tensor import Tensor
from .module import Module, Parameter
import numpy as np


class BatchNorm(Module):
    """Batch Normalization layer.

    Args:
        num_features: Number of features (channels).
        eps: Small constant for numerical stability.
        momentum: Momentum for running statistics.
        affine: If True, has learnable affine parameters.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Parameter(Tensor.ones(num_features))
            self.bias = Parameter(Tensor.zeros(num_features))
        else:
            self.weight = None
            self.bias = None

        self.register_buffer('running_mean', Tensor.zeros(num_features))
        self.register_buffer('running_var', Tensor.ones(num_features))
        self.num_batches_tracked = 0

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Compute mean and var along batch and spatial dims
            reduce_dims = tuple(range(0, x.ndim - 1)) if x.ndim > 2 else (0,)
            
            mean = x.mean(axis=reduce_dims, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=reduce_dims, keepdims=True)

            # Update running statistics
            xp = x.xp
            self.num_batches_tracked += 1
            m = self.momentum
            running_mean_data = mean.data.ravel()
            running_var_data = var.data.ravel()
            
            if hasattr(self.running_mean, 'data'):
                self.running_mean.data = (1 - m) * self.running_mean.data + m * running_mean_data
                self.running_var.data = (1 - m) * self.running_var.data + m * running_var_data

            x_norm = (x - mean) / (var + self.eps).sqrt()
        else:
            # Use running statistics
            xp = x.xp
            # Reshape running stats to broadcast
            shape = [1] * x.ndim
            shape[1] = self.num_features
            rm = self.running_mean.data.reshape(shape)
            rv = self.running_var.data.reshape(shape)
            x_norm = (x - Tensor(rm, device=x.device)) / Tensor(rv + self.eps, device=x.device).sqrt()

        if self.affine:
            shape = [1] * x.ndim
            shape[1] = self.num_features
            w = self.weight.data.reshape(shape)
            b = self.bias.data.reshape(shape)
            return Tensor(w, device=x.device) * x_norm + Tensor(b, device=x.device)
        
        return x_norm

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"


class LayerNorm(Module):
    """Layer Normalization.

    Args:
        normalized_shape: Shape of the normalized dimensions.
        eps: Small constant for numerical stability.
        affine: If True, has learnable affine parameters.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(Tensor.ones(*self.normalized_shape))
            self.bias = Parameter(Tensor.zeros(*self.normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last len(normalized_shape) dims
        reduce_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        
        mean = x.mean(axis=reduce_dims, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=reduce_dims, keepdims=True)
        
        x_norm = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            shape = [1] * x.ndim
            for i, d in enumerate(reduce_dims):
                shape[d] = self.normalized_shape[i]
            w = self.weight.data.reshape(shape)
            b = self.bias.data.reshape(shape)
            return Tensor(w, device=x.device) * x_norm + Tensor(b, device=x.device)

        return x_norm

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, affine={self.affine}"


class Dropout(Module):
    """Randomly zeroes elements during training.

    Args:
        p: Probability of an element to be zeroed.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        xp = x.xp
        mask = xp.random.binomial(1, 1 - self.p, size=x.shape).astype(x.data.dtype)
        # Scale to maintain expected value
        mask = mask / (1 - self.p)
        
        result = x * Tensor(mask, device=x.device)
        return result

    def extra_repr(self) -> str:
        return f"p={self.p}"
