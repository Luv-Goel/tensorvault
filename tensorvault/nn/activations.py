"""
Activation function modules (wrapping op functions as Module subclasses).
"""

from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    """Applies the rectified linear unit function."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Applies the sigmoid function."""
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Applies the hyperbolic tangent function."""
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class LeakyReLU(Module):
    """Applies the leaky rectified linear unit function.

    Args:
        negative_slope: Controls the angle of the negative slope.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        xp = x.xp
        data = xp.where(x.data > 0, x.data, self.negative_slope * x.data)
        return x._make_child(data, None, x._ctx, 'LeakyReLU') if hasattr(x, '_make_child') else Tensor(data, device=x.device)


class GELU(Module):
    """Applies the Gaussian Error Linear Units function."""
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu() if hasattr(x, 'gelu') else Tensor._gelu(x)


class Softmax(Module):
    """Applies the Softmax function.

    Args:
        dim: Dimension along which softmax is computed.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.dim)
