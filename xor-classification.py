# using the scalargrad engine!

from scalargrad.nn import MLP
from scalargrad.engine import Value


training_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

mlp = MLP(2, [2, 2], "sigmoid")
parameters = mlp.parameters()

epochs = 10000
learning_rate = 0.1


def softmax(logits):
    counts = [logit.exp() for logit in logits]
    denominator = sum(counts)
    out = [c / denominator for c in counts]
    return out


def cross_entropy(preds, target):
    return -(preds[target].log())


for epoch in range(epochs):
    loss = Value(0)

    for x, target in training_data:
        x_vals = [Value(xi) for xi in x]
        logits = mlp(x_vals)
        probs = softmax(logits)
        loss += cross_entropy(probs, target)

    for p in parameters:
        p.grad.fill(0)

    loss.backprop()

    for p in parameters:
        p.data -= learning_rate * p.grad

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss.data:.3f}")
        print("Network parameters:")
        for layer_idx, layer in enumerate(mlp.layers):
            print(f"  Layer {layer_idx + 1}:")
            for neuron_idx, neuron in enumerate(layer.neurons):
                weights_str = ", ".join(f"{w.data:.3f}" for w in neuron.w)
                print(
                    f"    Neuron {neuron_idx + 1}: weights=({weights_str}), bias={neuron.b.data:.3f}"
                )
        print("-" * 40)

print("\nTrained XOR results:")
for x, target in training_data:
    x_vals = [Value(xi) for xi in x]
    logits = mlp(x_vals)
    probs = softmax(logits)
    pred_label = 0 if probs[0].data > probs[1].data else 1

    print(
        f"Input {x} -> Probabilities: {[p.data for p in probs]}, "
        f"Predicted: {pred_label}, Target: {target}"
    )
