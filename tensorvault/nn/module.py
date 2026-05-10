"""
Base Module class and Parameter type.
"""

from ..tensor import Tensor
from typing import List, Dict, Optional, Iterator, Tuple, Any


class Parameter(Tensor):
    """A Tensor that is registered as a module parameter."""
    def __init__(self, data=None, requires_grad=True, device='cpu', dtype=None):
        if data is None:
            data = Tensor.zeros(1)
        if isinstance(data, Tensor):
            if data.requires_grad != requires_grad:
                data.requires_grad = requires_grad
            super().__init__(data.data.copy(), requires_grad=requires_grad, device=data.device)
        else:
            super().__init__(data, requires_grad=requires_grad, device=device, dtype=dtype)


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._modules: Dict[str, 'Module'] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, Tensor] = {}
        self.training = True

    def __setattr__(self, name: str, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Tensor):
            # Regular tensors are not parameters by default
            pass
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        if name in self._buffers:
            return self._buffers[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def register_parameter(self, name: str, param: Optional[Parameter]):
        if param is not None:
            self._parameters[name] = param
        elif name in self._parameters:
            del self._parameters[name]

    def register_buffer(self, name: str, tensor: Tensor):
        self._buffers[name] = tensor

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)

    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield name, param
        if recurse:
            for mod_name, module in self._modules.items():
                for p_name, param in module.named_parameters(recurse=True):
                    yield f"{mod_name}.{p_name}", param

    def modules(self) -> Iterator['Module']:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def children(self) -> Iterator['Module']:
        yield from self._modules.values()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

    def eval(self):
        self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        for name, param in self._parameters.items():
            destination[prefix + name] = param if keep_vars else param.data.copy()
        for name, buf in self._buffers.items():
            destination[prefix + name] = buf.data.copy() if hasattr(buf, 'data') else buf
        for mod_name, module in self._modules.items():
            module.state_dict(destination, prefix + mod_name + '.', keep_vars)
        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        own_state = self.state_dict(keep_vars=True)
        for name, param in own_state.items():
            if name in state_dict:
                if isinstance(param, Tensor):
                    param.data = state_dict[name].data if isinstance(state_dict[name], Tensor) else state_dict[name].copy()
                else:
                    own_state[name] = state_dict[name]
            elif strict:
                raise KeyError(f"Missing key in state_dict: '{name}'")
        # Load any remaining keys not in own_state
        for name, param in state_dict.items():
            if name not in own_state:
                if strict:
                    raise KeyError(f"Unexpected key in state_dict: '{name}'")

    def __repr__(self):
        return self._repr(0)

    def _repr(self, indent: int = 0) -> str:
        prefix = '  ' * indent
        lines = [f"{prefix}{type(self).__name__}("]
        for name, module in self._modules.items():
            lines.append(module._repr(indent + 1))
        lines.append(f"{prefix})")
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ''
