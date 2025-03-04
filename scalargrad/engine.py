from math import exp


class Value:
    def __init__(self, data, _childs=(), _op="") -> None:
        self.data = data
        self._childs = set(_childs)
        self.op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"data={self.data}"

    def __add__(self, addend):
        addend = addend if isinstance(addend, Value) else Value(addend)
        out = Value(self.data + addend.data, (self, addend), "+")

        def _backward():
            self.grad += out.grad
            addend.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, multiplier):
        multiplier = multiplier if isinstance(multiplier, Value) else Value(multiplier)
        out = Value(self.data * multiplier.data, (self, multiplier), "*")

        def _backward():
            self.grad += multiplier.data * out.grad
            multiplier.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data**power, (self,), "**")

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def backprop(self):
        # topological sort
        sorted = []
        vis = set()

        def dfs(v):
            if v not in vis:
                vis.add(v)
                for child in v._childs:
                    dfs(child)
                sorted.append(v)

        dfs(self)

        self.grad = 1.0

        for n in reversed(sorted):
            n._backward()

    # def exp(self):
    #     pass

    def tanh(self):
        x = self.data
        t = (exp(2**x) - 1) / (exp(2**x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        pass

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
