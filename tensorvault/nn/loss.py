"""
Loss functions: MSE, CrossEntropy, BCE, NLLLoss.
"""

from ..tensor import Tensor
from .module import Module
import numpy as np


class MSELoss(Module):
    """Mean Squared Error loss.

    Args:
        reduction: 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        loss = diff ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CrossEntropyLoss(Module):
    """Cross Entropy loss (combines LogSoftmax + NLLLoss).

    Args:
        reduction: 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (batch, num_classes), target: (batch,) of class indices."""
        # LogSoftmax
        xp = pred.xp
        pred_data = pred.data
        # Subtract max for numerical stability
        pred_max = xp.max(pred_data, axis=1, keepdims=True)
        exp_pred = xp.exp(pred_data - pred_max)
        log_sum_exp = xp.log(xp.sum(exp_pred, axis=1, keepdims=True))
        log_probs = pred_data - pred_max - log_sum_exp  # (batch, num_classes)

        # Gather the log probabilities at target indices
        batch_size = pred.shape[0]
        target_data = target.data if isinstance(target, Tensor) else target
        target_data = target_data.astype(xp.int64)
        
        # One-hot like gather
        batch_indices = xp.arange(batch_size)
        nll = -log_probs[batch_indices, target_data]  # (batch,)

        if self.reduction == 'mean':
            return Tensor(nll.mean(), device=pred.device)
        elif self.reduction == 'sum':
            return Tensor(nll.sum(), device=pred.device)
        return Tensor(nll, device=pred.device)


class BCELoss(Module):
    """Binary Cross Entropy loss.

    Args:
        reduction: 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        xp = pred.xp
        eps = 1e-8
        pred_clamped = xp.clip(pred.data, eps, 1 - eps)
        loss = -(target.data * xp.log(pred_clamped) + (1 - target.data) * xp.log(1 - pred_clamped))
        loss_t = Tensor(loss, device=pred.device)
        if self.reduction == 'mean':
            return loss_t.mean()
        elif self.reduction == 'sum':
            return loss_t.sum()
        return loss_t


class NLLLoss(Module):
    """Negative Log Likelihood loss.

    Args:
        reduction: 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (batch, num_classes) - log probabilities, target: (batch,) of class indices."""
        xp = pred.xp
        batch_size = pred.shape[0]
        target_data = target.data if isinstance(target, Tensor) else target
        target_data = target_data.astype(xp.int64)
        
        batch_indices = xp.arange(batch_size)
        nll = -pred.data[batch_indices, target_data]

        if self.reduction == 'mean':
            return Tensor(nll.mean(), device=pred.device)
        elif self.reduction == 'sum':
            return Tensor(nll.sum(), device=pred.device)
        return Tensor(nll, device=pred.device)
