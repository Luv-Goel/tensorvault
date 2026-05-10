"""
Parameter initialization functions.
"""

from ..tensor import Tensor
from .module import Parameter
import numpy as np


def _calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity."""
    gains = {
        'sigmoid': 1,
        'tanh': 5/3,
        'relu': np.sqrt(2),
        'leaky_relu': np.sqrt(2 / (1 + param**2)) if param else np.sqrt(2),
        'linear': 1,
        'identity': 1,
    }
    return gains.get(nonlinearity, 1)


def xavier_uniform_(tensor: Parameter, gain: float = 1.0):
    """Fill tensor with values from a uniform distribution using Xavier init."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    fan_in, fan_out = _fan_in_fan_out(t.shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    limit = np.sqrt(3.0) * std
    t.data = xp.random.uniform(-limit, limit, size=t.shape).astype(t.data.dtype)
    return tensor


def xavier_normal_(tensor: Parameter, gain: float = 1.0):
    """Fill tensor with values from a normal distribution using Xavier init."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    fan_in, fan_out = _fan_in_fan_out(t.shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    t.data = xp.random.normal(0, std, size=t.shape).astype(t.data.dtype)
    return tensor


def kaiming_uniform_(tensor: Parameter, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """Fill tensor with values using Kaiming He uniform init."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    fan = _fan_in_fan_out(t.shape)[0 if mode == 'fan_in' else 1]
    gain = _calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    limit = np.sqrt(3.0) * std
    t.data = xp.random.uniform(-limit, limit, size=t.shape).astype(t.data.dtype)
    return tensor


def kaiming_normal_(tensor: Parameter, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """Fill tensor with values using Kaiming He normal init."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    fan = _fan_in_fan_out(t.shape)[0 if mode == 'fan_in' else 1]
    gain = _calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    t.data = xp.random.normal(0, std, size=t.shape).astype(t.data.dtype)
    return tensor


def zeros_(tensor: Parameter):
    """Fill tensor with zeros."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    t.data = xp.zeros_like(t.data)
    return tensor


def ones_(tensor: Parameter):
    """Fill tensor with ones."""
    if isinstance(tensor, Parameter):
        t = tensor
    else:
        t = tensor
    xp = t.xp
    t.data = xp.ones_like(t.data)
    return tensor


def _fan_in_fan_out(shape):
    """Compute fan_in and fan_out for initialization."""
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    # Conv weights: (out_channels, in_channels, *kernel_size)
    fan_in = shape[1] * np.prod(shape[2:])
    fan_out = shape[0] * np.prod(shape[2:])
    return fan_in, fan_out
