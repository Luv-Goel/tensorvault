"""
Container modules: Sequential, ModuleList.
"""

from .module import Module
from ..tensor import Tensor
from typing import Iterable, Iterator, Union


class Sequential(Module):
    """A sequential container of modules.

    Modules are called in the order they are passed.
    """
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


class ModuleList(Module):
    """A list of modules."""
    def __init__(self, modules: Iterable[Module] = None):
        super().__init__()
        if modules is not None:
            for i, module in enumerate(modules):
                self._modules[str(i)] = module

    def append(self, module: Module):
        self._modules[str(len(self._modules))] = module

    def __getitem__(self, idx) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())
