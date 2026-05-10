"""
Optimizers: SGD, Adam, AdamW.
"""

from ..tensor import Tensor
from ..nn.module import Module, Parameter
from typing import Iterable, Callable, Optional


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: Iterable[Parameter], defaults: dict):
        self.params = list(params)
        self.defaults = defaults
        self.state = {}  # Optimizer state per parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[id(p)] = {}

    @property
    def param_groups(self):
        return [{'params': self.params, **self.defaults}]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()

    def step(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.defaults})"


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay (L2 penalty).
        nesterov: Enables Nesterov momentum.
    """
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                xp = p.xp
                grad = p.grad

                # Weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                # Momentum
                if momentum != 0:
                    param_state = self.state[id(p)]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = xp.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf = momentum * buf + grad
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                    param_state['momentum_buffer'] = buf

                p.data = p.data - lr * grad


class Adam(Optimizer):
    """Adam optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for computing running averages.
        eps: Term added for numerical stability.
        weight_decay: Weight decay (L2 penalty).
    """
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        import numpy as np
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                xp = p.xp
                grad = p.grad

                # Weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                param_state = self.state[id(p)]
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = xp.zeros_like(p.data)
                    param_state['exp_avg_sq'] = xp.zeros_like(p.data)

                param_state['step'] += 1
                step = param_state['step']

                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # Update biased second moment estimate
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Update parameters
                denom = (xp.sqrt(exp_avg_sq) / np.sqrt(bias_correction2) + eps)
                step_size = lr / bias_correction1
                p.data = p.data - step_size * exp_avg / denom

                # Save state
                param_state['exp_avg'] = exp_avg
                param_state['exp_avg_sq'] = exp_avg_sq


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for computing running averages.
        eps: Term added for numerical stability.
        weight_decay: Weight decay factor.
    """
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        import numpy as np
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                xp = p.xp
                grad = p.grad

                param_state = self.state[id(p)]
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = xp.zeros_like(p.data)
                    param_state['exp_avg_sq'] = xp.zeros_like(p.data)

                param_state['step'] += 1
                step = param_state['step']

                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']

                # Decoupled weight decay
                p.data = p.data - lr * weight_decay * p.data

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # Update biased second moment estimate
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Update parameters
                denom = (xp.sqrt(exp_avg_sq) / bias_correction2 ** 0.5 + eps)
                step_size = lr / bias_correction1
                p.data = p.data - step_size * exp_avg / denom

                # Save state
                param_state['exp_avg'] = exp_avg
                param_state['exp_avg_sq'] = exp_avg_sq
