"""
2D Convolution layer.
"""

from ..tensor import Tensor
from .module import Module, Parameter
import numpy as np


class Conv2D(Module):
    """2D convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides.
        bias: If True, adds a learnable bias.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        k = 1.0 / (in_channels * kernel_size * kernel_size)
        self.weight = Parameter(
            Tensor.randn(out_channels, in_channels, kernel_size, kernel_size) * k ** 0.5
        )
        if bias:
            self.bias = Parameter(Tensor.zeros(out_channels))
        else:
            self.bias = None

    def _pad_input(self, x: Tensor) -> Tensor:
        if self.padding == 0:
            return x
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        return x.pad(pad_width, constant_values=0)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch_size, in_channels, height, width)"""
        xp = x.xp
        batch_size, in_c, height, width = x.shape
        out_c, _, k_h, k_w = self.weight.shape
        stride = self.stride

        x_padded = self._pad_input(x)
        p_h, p_w = x_padded.shape[2], x_padded.shape[3]

        # Output dimensions
        out_h = (p_h - k_h) // stride + 1
        out_w = (p_w - k_w) // stride + 1

        # Im2col approach
        # Create sliding window view using unfold
        # Weight: (out_c, in_c, k_h, k_w) -> (out_c, in_c * k_h * k_w)
        W_flat = self.weight.data.reshape(out_c, -1)  # (out_c, in_c * k_h * k_w)

        # Extract patches: (batch, in_c, out_h, out_w, k_h, k_w) -> (batch, out_h, out_w, in_c * k_h * k_w)
        patches = xp.zeros((batch_size, out_h, out_w, in_c * k_h * k_w), dtype=x.data.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded.data[:, :, h_start:h_start + k_h, w_start:w_start + k_w]
                patches[:, i, j, :] = patch.reshape(batch_size, -1)

        # Compute output: (batch, out_h, out_w, out_c)
        out_data = xp.tensordot(patches, W_flat.T, axes=([3], [0]))  # (batch, out_h, out_w, out_c)

        # Rearrange to (batch, out_c, out_h, out_w)
        out_data = xp.transpose(out_data, (0, 3, 1, 2))

        # Create result tensor
        result = Tensor(out_data, requires_grad=x.requires_grad, device=x.device)

        if self.bias is not None:
            result = result + self.bias.reshape(1, -1, 1, 1)

        # We need to handle gradients manually since our autograd doesn't support Conv2D natively
        # For simplicity, we implement a manual backward in a wrapper
        if x.requires_grad or self.weight.requires_grad:
            result = Conv2DFunction.apply(x, self.weight, self.bias, self.padding, self.stride, result)
        else:
            result = Tensor(out_data, requires_grad=False, device=x.device)
            if self.bias is not None:
                result = result + self.bias.reshape(1, -1, 1, 1)

        return result


class Conv2DFunction:
    """Manual Conv2D with autograd support using im2col."""
    
    @staticmethod
    def apply(x, weight, bias, padding, stride, output=None):
        from ..tensor import Tensor, _Context
        from ..ops.basic import Add
        
        xp = x.xp
        batch_size, in_c, height, width = x.shape
        out_c, _, k_h, k_w = weight.shape
        out_h = (height + 2 * padding - k_h) // stride + 1
        out_w = (width + 2 * padding - k_w) // stride + 1

        ctx = _Context()
        ctx.save_for_backward(x, weight)

        # Manually compute forward if not provided
        if output is None:
            x_padded = xp.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
            W_flat = weight.data.reshape(out_c, -1)
            patches = xp.zeros((batch_size, out_h, out_w, in_c * k_h * k_w), dtype=x.data.dtype)
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x_padded[:, :, h_start:h_start + k_h, w_start:w_start + k_w]
                    patches[:, i, j, :] = patch.reshape(batch_size, -1)
            out_data = xp.tensordot(patches, W_flat.T, axes=([3], [0]))
            out_data = xp.transpose(out_data, (0, 3, 1, 2))
            result = Tensor(out_data, requires_grad=True, device=x.device)
            if bias is not None:
                result = result + bias.reshape(1, -1, 1, 1)
        else:
            result = output

        result._is_leaf = False
        result._backward_fn = Conv2DFunction.backward
        result._ctx = ctx
        result._grad_fn_name = 'Conv2D'
        result._conv_padding = padding
        result._conv_stride = stride
        result._conv_kernel_size = k_h
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.get_saved_tensors()
        # Extract saved conv params from grad_output
        padding = getattr(grad_output, '_conv_padding', 0)
        stride = getattr(grad_output, '_conv_stride', 1)
        k_h = getattr(grad_output, '_conv_kernel_size', 3)
        
        xp = x.xp
        batch_size, in_c, height, width = x.shape
        out_c, _, k_h, k_w = weight.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        # Pad input for backward
        x_padded = xp.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        W_flat = weight.data.reshape(out_c, -1)
        g = grad_output.data  # (batch, out_c, out_h, out_w)

        # Gradient w.r.t. weight: im2col patches @ grad_output
        patches = xp.zeros((batch_size, out_h, out_w, in_c * k_h * k_w), dtype=x.data.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start + k_h, w_start:w_start + k_w]
                patches[:, i, j, :] = patch.reshape(batch_size, -1)

        # grad_weight: (out_c, in_c * k_h * k_w)
        g_transposed = xp.transpose(g, (0, 2, 3, 1))  # (batch, out_h, out_w, out_c)
        grad_weight = xp.tensordot(g_transposed, patches, axes=([0, 1, 2], [0, 1, 2]))
        # grad_weight shape: (out_c, in_c * k_h * k_w) -> (out_c, in_c, k_h, k_w)
        grad_weight = grad_weight.reshape(out_c, in_c, k_h, k_w)

        # Gradient w.r.t. input: col2im
        # g @ W  then fold back
        W_flat_T = W_flat.T  # (in_c * k_h * k_w, out_c)
        grad_im2col = xp.tensordot(g_transposed, W_flat_T, axes=([3], [1]))  # (batch, out_h, out_w, in_c * k_h * k_w)

        # Fold back to input
        grad_x = xp.zeros_like(x_padded)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch_grad = grad_im2col[:, i, j, :].reshape(batch_size, in_c, k_h, k_w)
                grad_x[:, :, h_start:h_start + k_h, w_start:w_start + k_w] += patch_grad

        # Remove padding
        if padding > 0:
            grad_x = grad_x[:, :, padding:-padding, padding:-padding]

        grad_x = Tensor(grad_x, device=x.device)
        grad_weight = Tensor(grad_weight, device=weight.device)

        return grad_x, grad_weight, None if weight.shape[0] > 0 else None
