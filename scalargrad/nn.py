from __future__ import annotations
import random
from scalargrad.engine import Value
from typing import Callable, Union


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


def get_activation(
    activation: Union[None, str, Callable[[Value], Value]],
) -> Callable[[Value], Value]:
    if activation is None:
        return lambda x: x.tanh()
    if isinstance(activation, str):
        act = activation.lower()
        if act == "tanh":
            return lambda x: x.tanh()
        elif act == "relu":
            return lambda x: x.relu()
        elif act == "sigmoid":
            return lambda x: x.sigmoid()
        elif act == "none":
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    if callable(activation):
        return activation


class Neuron(Module):
    def __init__(self, n_in, activation=None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = get_activation(activation)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = self.activation(act)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layers(Module):
    def __init__(self, n_in, n_out, activation=None):
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, n_out, activation=None):
        sizes = [n_in] + n_out
        self.layers = [
            Layers(sizes[i], sizes[i + 1], activation) for i in range(len(n_out))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
