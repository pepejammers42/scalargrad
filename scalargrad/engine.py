import numpy as np


class Value:
    def __init__(self, data, _childs=(), _op=""):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        else:
            data = data.astype(np.float64)

        self.data = data
        self._childs = set(_childs)
        self.op = _op
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return Value(other) * (self**-1)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int/float powers"
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        t = 1 / (1 + np.exp(-self.data))
        out = Value(t, (self,), "sigmoid")

        def _backward():
            self.grad += (t * (1 - t)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0).astype(self.data.dtype) * out.grad

        out._backward = _backward
        return out

    def backprop(self):
        # Topological sort: build a list of nodes in the graph.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._childs:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
