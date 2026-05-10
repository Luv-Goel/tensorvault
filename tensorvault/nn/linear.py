"""
Linear (fully connected) layer.
"""

from ..tensor import Tensor
from .module import Module, Parameter
from . import init


class Linear(Module):
    """Applies a linear transformation: y = x @ W.T + b.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
