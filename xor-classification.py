# using the scalargrad engine!

from scalargrad.nn import MLP
from scalargrad.engine import Value


training_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

mlp = MLP(2, [2, 1])
parameters = mlp.parameters()

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    loss = Value(0)

    for x, target in training_data:
        x_vals = [Value(xi) for xi in x]
        pred = mlp(x_vals)[0]
        diff = pred - target
        loss = loss + diff * diff

    for p in parameters:
        p.grad = 0

    loss.backprop()

    for p in parameters:
        p.data -= learning_rate * p.grad

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss.data:.3f}")

print("\nTrained XOR results:")
for x, target in training_data:
    x_vals = [Value(xi) for xi in x]
    pred = mlp(x_vals)[0]

    pred_label = 1 if pred.data > 0.5 else 0

    print(
        f"Input {x} -> Raw Output: {pred.data:.3f}, "
        + f"Predicted: {pred_label}, Target: {target}"
    )
