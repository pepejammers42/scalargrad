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
        out = Value(self.data + addend.data, (self, addend), "+")
        return out

    def __mul__(self, multiplier):
        out = Value(self.data * multiplier.data, (self, multiplier), "*")
        return out

    def __pow__(self, power):
        out = Value(self.data**power, (self,), "**")

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "relu")
        return out

    def backprop(self):
        pass

    def exp(self):
        pass
