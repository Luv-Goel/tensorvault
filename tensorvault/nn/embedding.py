"""
Embedding layer.
"""

from ..tensor import Tensor
from .module import Module, Parameter


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings: Size of the dictionary.
        embedding_dim: Dimension of each embedding vector.
        padding_idx: If specified, the entries at padding_idx do not contribute to gradient.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(Tensor.randn(num_embeddings, embedding_dim) * 0.1)

    def forward(self, indices) -> Tensor:
        """indices: Tensor of token indices with shape (*,)."""
        if isinstance(indices, Tensor):
            indices_np = indices.data
        else:
            indices_np = indices

        xp = self.weight.xp
        # Gather embeddings
        if hasattr(xp, 'take'):
            data = xp.take(self.weight.data, indices_np.astype(xp.int64), axis=0)
        else:
            data = self.weight.data[indices_np.astype(xp.int64)]

        result = Tensor(data, requires_grad=self.weight.requires_grad, device=self.weight.device)

        if self.weight.requires_grad:
            result._is_leaf = False
            result._backward_fn = self._backward
            result._ctx = (self, indices_np)
            result._grad_fn_name = 'Embedding'

        return result

    @staticmethod
    def _backward(ctx, grad_output):
        module, indices = ctx
        xp = grad_output.xp
        grad_weight = xp.zeros_like(module.weight.data)
        if module.padding_idx is not None:
            mask = indices != module.padding_idx
            valid_indices = indices[mask]
            if len(valid_indices) > 0:
                xp.add.at(grad_weight, valid_indices.astype(xp.int64), grad_output.data[mask])
        else:
            xp.add.at(grad_weight, indices.astype(xp.int64), grad_output.data)
        return Tensor(grad_weight, device=module.weight.device)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
