import logging
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

from network import FFNeuralNetwork
from activation_functions import *
from loss_functions import *
from metric_functions import *



def get_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    raw_x = np.asarray(mnist.data)
    raw_y = np.asarray(mnist.target).reshape(-1, 1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(raw_x)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(raw_y)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, test_size=10000, random_state=42)
    return x_train, x_test, y_train, y_test



def train_mnist(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> FFNeuralNetwork:
    
    model = FFNeuralNetwork.initialise_with_random_small_parameters(
        layer_sizes=[784, 256, 128, 10],
        metric_function=categorical_accuracy_metric,
        activation_function=relu_activation,
        final_activation_function=softmax_activation,
        loss_function=cross_entropy_loss_function,
        weights_std=0.05
    )

    model.fit(
        features=x_train,
        labels=y_train,
        epochs=2,
        batch_size=32,
        learning_rate=0.01,
        momentum_gamma=0.9
    )
    test_preds = model.predict(x_test)
    final_acc = categorical_accuracy_metric(test_preds.T, y_test)
    print(f"\nFinal MNIST Test Accuracy: {final_acc:.4f}")
    return model


def run_experiment() -> FFNeuralNetwork:
    x_train, x_test, y_train, y_test = get_mnist_data()
    return train_mnist(x_train, x_test, y_train, y_test)


def visualise_predictions(model: FFNeuralNetwork, image: np.ndarray) -> None:
    # Flatten image for the model and reshape back for visualisation
    x_sample = image.reshape(1, -1)
    preds = model.predict(x_sample)
    pred_label = int(np.argmax(preds, axis=0)[0])

    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.show()
    print(pred_label, f"(Predicted probabilities: {preds.flatten()})")


if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    filename=Path(__file__).parent / "logs" / "mnist_training.log",
    filemode='w',
)
    
    run_experiment()
