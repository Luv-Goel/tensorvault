"""
tensorvault — A from-scratch deep learning framework with automatic differentiation.

Built for understanding: like micro-PyTorch, with GPU support and neural network modules.
"""

__version__ = "0.2.0"

from .tensor import Tensor
from . import ops
from . import nn
from . import optim
from . import data
from . import utils
