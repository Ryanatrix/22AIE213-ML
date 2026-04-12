import numpy as np
import pandas as pd


def A1_summation(x, w, b):
    return np.dot(x, w) + b


def A1_step(x):
    return 1 if x >= 0 else 0


def A1_bipolar(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def A1_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def A1_relu(x):
    return max(0, x)


def A1_leaky_relu(x):
    return x if x > 0 else 0.01 * x


def A1_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def A2_train_perceptron(X, y, activation, lr=0.05, max_epochs=1000):
    w = np.array([0.2, -0.75])
    b = 10
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            net = A1_summation(X[i], w, b)
            out = activation(net)
            err = y[i] - out
            total_error += err**2
            w = w + lr * err * X[i]
            b = b + lr * err

        errors.append(total_error)

        if total_error <= 0.002:
            break

    return w, b, errors, epoch + 1


def A3_compare_activations(X, y):
    activations = {
        "Step": A1_step,
        "Bipolar": A1_bipolar,
        "Sigmoid": A1_sigmoid,
        "ReLU": A1_relu
    }

    results = {}

    for name, func in activations.items():
        _, _, _, epochs = A2_train_perceptron(X, y, func)
        results[name] = epochs

    return results


def A4_learning_rates(X, y, activation):
    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    results = {}

    for lr in rates:
        _, _, _, epochs = A2_train_perceptron(X, y, activation, lr=lr)
        results[lr] = epochs

    return results


def A5_xor_experiment():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    _, _, errors, epochs = A2_train_perceptron(X, y, A1_step)
    return errors, epochs


def A6_dataset():
    data = {
        "Candies":[20,16,27,19,24,22,15,18,21,16],
        "Mangoes":[6,3,6,1,4,1,4,4,1,2],
        "Milk":[2,6,2,2,2,5,2,2,4,4],
        "Payment":[386,289,393,110,280,167,271,274,148,198],
        "Label":[1,1,1,0,1,0,1,1,0,0]
    }

    df = pd.DataFrame(data)
    X = df.iloc[:,:4].values
    y = df["Label"].values

    return X, y


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


def A8_neural_network(X, y, lr=0.05, epochs=1000):
    np.random.seed(42)

    input_size = X.shape[1]
    hidden_size = 2
    output_size = 1

    W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))

    for _ in range(epochs):
        h = sigmoid(np.dot(X, W1))
        out = sigmoid(np.dot(h, W2))

        error = y.reshape(-1,1) - out

        d_out = error * sigmoid_derivative(out)
        d_hidden = d_out.dot(W2.T) * sigmoid_derivative(h)

        W2 += lr * h.T.dot(d_out)
        W1 += lr * X.T.dot(d_hidden)

    return W1, W2


def main():
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])

    w, b, errors, epochs = A2_train_perceptron(X_and, y_and, A1_step)
    print("A2 AND Epochs:", epochs)

    act_results = A3_compare_activations(X_and, y_and)
    print("A3:", act_results)

    lr_results = A4_learning_rates(X_and, y_and, A1_step)
    print("A4:", lr_results)

    xor_errors, xor_epochs = A5_xor_experiment()
    print("A5 XOR Epochs:", xor_epochs)

    X_real, y_real = A6_dataset()
    _, _, _, epochs = A2_train_perceptron(X_real, y_real, A1_sigmoid)
    print("A6 Dataset Epochs:", epochs)

    W1, W2 = A8_neural_network(X_and, y_and)
    print("A8 W1:", W1)
    print("A8 W2:", W2)


if __name__ == "__main__":
    main()
