import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model


# ── Utilities ──────────────────────────────────────────────────────────────

def sigmoid(x):
    """Sigmoid activation: 1 / (1 + e^-x)"""
    return 1 / (1 + np.exp(-x))


def load_planar_dataset():
    """Generate the 2D flower dataset — 400 examples, 2 features, binary labels."""
    np.random.seed(1)
    m, N, a = 400, 200, 4
    X = np.zeros((m, 2))
    Y = np.zeros((m, 1), dtype='uint8')
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X.T, Y.T


def plot_decision_boundary(model, X, Y):
    """Plot the model's decision boundary over the dataset."""
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral)


# ── Model ──────────────────────────────────────────────────────────────────

def layer_sizes(X, Y):
    """Return (n_x, n_h, n_y) — input, hidden, and output layer sizes."""
    return X.shape[0], 4, Y.shape[0]


def initialize_parameters(n_x, n_h, n_y):
    """Initialize weights randomly (×0.01) and biases to zero."""
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros((n_y, 1)),
    }


def forward_propagation(X, parameters):
    """Run the forward pass: tanh on hidden layer, sigmoid on output. Returns (A2, cache)."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    A1 = np.tanh(np.dot(W1, X) + b1)
    A2 = sigmoid(np.dot(W2, A1) + b2)
    cache = {"A1": A1, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    """Compute binary cross-entropy loss."""
    m = Y.shape[1]
    cost = -(1 / m) * np.sum(np.multiply(np.log(A2), Y) +
                              np.multiply(np.log(1 - A2), (1 - Y)))
    return float(np.squeeze(cost))


def backward_propagation(parameters, cache, X, Y):
    """Compute gradients for W1, b1, W2, b2 via backprop."""
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = parameters["W2"]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(parameters, grads, learning_rate=1.2):
    """Update parameters using gradient descent: W = W - α·dW"""
    return {
        "W1": parameters["W1"] - learning_rate * grads["dW1"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b2": parameters["b2"] - learning_rate * grads["db2"],
    }


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """Train the network by running the forward/backward loop. Returns learned parameters."""
    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")

    return parameters


def predict(parameters, X):
    """Predict class labels (0 or 1) by thresholding A2 at 0.5."""
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, Y = load_planar_dataset()

    # Baseline
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T.ravel())
    lr_acc = float((np.dot(Y, clf.predict(X.T)) +
                    np.dot(1 - Y, 1 - clf.predict(X.T))) / Y.size * 100)
    print(f"Logistic Regression Accuracy: {lr_acc:.0f}%")

    # Neural network
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    predictions = predict(parameters, X)
    nn_acc = float((np.dot(Y, predictions.T) +
                    np.dot(1 - Y, 1 - predictions.T)) / Y.size * 100)
    print(f"Neural Network Accuracy: {nn_acc:.0f}%")

    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Neural Network — Decision Boundary (n_h = 4)")
    plt.show()
