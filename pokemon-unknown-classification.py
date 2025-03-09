from scalargrad.nn import MLP
from scalargrad.engine import Value
import numpy as np
import os
import gc
from urllib.request import urlretrieve


# imports are taken from STA414-2025 A1. Thanks Professor Erdogdu! (I didn't ask him for permission)
def download(url, filename):
    if not os.path.exists("data"):
        os.makedirs("data")
    out_file = os.path.join("data", filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def unown_mnist():
    base_url = "https://github.com/lopeLH/unown-mnist/raw/refs/heads/main/"
    X_test_url = "X_test.npy"
    X_train_url = "X_train.npy"
    y_train_url = "Y_train.npy"
    y_test_url = "Y_test.npy"

    for filename in [X_train_url, X_test_url, y_train_url, y_test_url]:
        download(base_url + filename, filename)

    X_train = np.load("./data/X_train.npy")
    X_test = np.load("./data/X_test.npy")
    Y_train = np.load("./data/Y_train.npy")
    Y_test = np.load("./data/Y_test.npy")

    return X_train, Y_train, X_test, Y_test


def load_unown_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = unown_mnist()
    num_unique_labels = len(np.unique(train_labels))
    train_images = (partial_flatten(train_images) / 255.0).astype(float)
    test_images = (partial_flatten(test_images) / 255.0).astype(float)
    train_images_binarized = (train_images > 0.5).astype(float)
    test_images_binarized = (test_images > 0.5).astype(float)
    train_labels = one_hot(train_labels, num_unique_labels)
    test_labels = one_hot(test_labels, num_unique_labels)
    N_data = train_images.shape[0]

    return (
        N_data,
        train_images,
        train_images_binarized,
        train_labels,
        test_images,
        test_images_binarized,
        test_labels,
    )


(
    N_data,
    train_images,
    train_images_binarized,
    train_labels,
    test_images,
    test_images_binarized,
    test_labels,
) = load_unown_mnist()

N_small = 100
train_images = train_images[:N_small]
train_labels = train_labels[:N_small]
N_data = train_images.shape[0]
test_images = test_images[100 : 100 + N_small]
test_labels = test_labels[100 : 100 + N_small]


print(f"Training examples: {train_images.shape[0]}")
print(f"Input dimension: {train_images.shape[1]}")
print(f"Number of classes: {len(train_labels[0])}")
print(f"Training examples: {train_images.shape[0]}")
print(f"Test examples: {test_images.shape[0]}")


def softmax(logits):
    exps = [logit.exp() for logit in logits]
    denominator = sum(exps)
    return [e / denominator for e in exps]


def cross_entropy(preds, target_index):
    return -(preds[target_index].log())


input_dim = train_images.shape[1]
output_dim = len(train_labels[0])
hidden_size = 28

mlp = MLP(input_dim, [hidden_size, output_dim], "relu")
parameters = mlp.parameters()

epochs = 5
learning_rate = 0.1
batch_size = 32
num_batches = N_data // batch_size  # number of mini-batches

for epoch in range(epochs):
    total_loss = Value(0)

    for i in range(num_batches):
        batch_loss = Value(0)
        for j in range(batch_size):
            index = i * batch_size + j
            x = train_images[index]
            target_index = int(np.argmax(train_labels[index]))
            x_vals = [Value(xi) for xi in x]
            logits = mlp(x_vals)
            probs = softmax(logits)
            sample_loss = cross_entropy(probs, target_index)
            batch_loss += sample_loss
        batch_loss = batch_loss * (1 / batch_size)

        for p in parameters:
            p.grad.fill(0)
        batch_loss.backprop()
        for p in parameters:
            p.data -= learning_rate * p.grad

        total_loss += batch_loss

        del batch_loss
        gc.collect()

    avg_loss = total_loss.data / num_batches
    print(f"Epoch {epoch}, Average Loss = {avg_loss:.3f}")
    del total_loss
    gc.collect()

correct = 0
for i in range(len(test_images)):
    x = test_images[i]
    target_index = int(np.argmax(test_labels[i]))
    x_vals = [Value(xi) for xi in x]
    logits = mlp(x_vals)
    probs = softmax(logits)
    pred_label = np.argmax([p.data for p in probs])
    if pred_label == target_index:
        correct += 1

accuracy = correct / len(test_images)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
