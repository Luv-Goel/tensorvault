"""
RNN and LSTM layers.
"""

from ..tensor import Tensor
from .module import Module, Parameter
import numpy as np


class RNN(Module):
    """A simple Elman RNN layer.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden features.
        num_layers: Number of recurrent layers.
        bias: If True, uses bias.
        dropout: Dropout probability (not implemented for RNN yet).
        nonlinearity: 'tanh' or 'relu'.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        nonlinearity: str = 'tanh',
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.weight_ih_l = ParameterList()
        self.weight_hh_l = ParameterList()
        self.bias_ih_l = ParameterList()
        self.bias_hh_l = ParameterList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.weight_ih_l.append(Parameter(Tensor.randn(hidden_size, layer_input_size) * 0.01))
            self.weight_hh_l.append(Parameter(Tensor.randn(hidden_size, hidden_size) * 0.01))
            if bias:
                self.bias_ih_l.append(Parameter(Tensor.zeros(hidden_size)))
                self.bias_hh_l.append(Parameter(Tensor.zeros(hidden_size)))
            else:
                self.bias_ih_l.append(None)
                self.bias_hh_l.append(None)

    def forward(self, x: Tensor, h0: Tensor = None) -> tuple:
        """x: (seq_len, batch, input_size)"""
        seq_len, batch_size, _ = x.shape
        if h0 is None:
            h0 = Tensor.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        output = []
        h = h0
        for t in range(seq_len):
            xt = x[t]
            new_h = []
            for layer in range(self.num_layers):
                h_layer = h[layer]
                ih = self.weight_ih_l[layer] @ xt + (self.bias_ih_l[layer] if self.bias_ih_l[layer] is not None else 0)
                hh = self.weight_hh_l[layer] @ h_layer + (self.bias_hh_l[layer] if self.bias_hh_l[layer] is not None else 0)
                if self.nonlinearity == 'tanh':
                    ht = (ih + hh).tanh()
                else:
                    ht = (ih + hh).relu()
                new_h.append(ht)
                xt = ht
            h = Tensor.stack(new_h)
            output.append(xt)

        output = Tensor.stack(output)
        return output, h


class LSTM(Module):
    """LSTM layer.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden features.
        num_layers: Number of recurrent layers.
        bias: If True, uses bias.
        dropout: Dropout probability (not implemented for LSTM yet).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.weight_ih_l = ParameterList()
        self.weight_hh_l = ParameterList()
        self.bias_ih_l = ParameterList()
        self.bias_hh_l = ParameterList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            # Gates: input, forget, cell, output
            self.weight_ih_l.append(Parameter(Tensor.randn(4 * hidden_size, layer_input_size) * 0.01))
            self.weight_hh_l.append(Parameter(Tensor.randn(4 * hidden_size, hidden_size) * 0.01))
            if bias:
                self.bias_ih_l.append(Parameter(Tensor.zeros(4 * hidden_size)))
                self.bias_hh_l.append(Parameter(Tensor.zeros(4 * hidden_size)))
            else:
                self.bias_ih_l.append(None)
                self.bias_hh_l.append(None)

    def forward(self, x: Tensor, state=None) -> tuple:
        """x: (seq_len, batch, input_size)"""
        seq_len, batch_size, _ = x.shape
        if state is None:
            h0 = Tensor.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = Tensor.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h0, c0 = state

        h = h0
        c = c0
        output = []

        for t in range(seq_len):
            xt = x[t]
            new_h = []
            new_c = []
            for layer in range(self.num_layers):
                h_layer = h[layer]
                c_layer = c[layer]

                gates_ih = self.weight_ih_l[layer] @ xt
                gates_hh = self.weight_hh_l[layer] @ h_layer

                if self.bias_ih_l[layer] is not None:
                    gates_ih = gates_ih + self.bias_ih_l[layer]
                if self.bias_hh_l[layer] is not None:
                    gates_hh = gates_hh + self.bias_hh_l[layer]

                gates = gates_ih + gates_hh
                # Split gates
                hs = self.hidden_size
                i_gate = gates[:hs].sigmoid()
                f_gate = gates[hs:2*hs].sigmoid()
                g_gate = gates[2*hs:3*hs].tanh()
                o_gate = gates[3*hs:4*hs].sigmoid()

                c_new = f_gate * c_layer + i_gate * g_gate
                h_new = o_gate * c_new.tanh()

                new_c.append(c_new)
                new_h.append(h_new)
                xt = h_new

            h = Tensor.stack(new_h)
            c = Tensor.stack(new_c)
            output.append(xt)

        output = Tensor.stack(output)
        return output, (h, c)


class ParameterList(Module):
    """A simple list of parameters (helper for RNN/LSTM)."""
    def __init__(self):
        super().__init__()
        self._params = []

    def append(self, p):
        self._params.append(p)

    def __getitem__(self, idx):
        return self._params[idx]

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    def parameters(self, recurse=True):
        for p in self._params:
            if isinstance(p, Parameter):
                yield p
