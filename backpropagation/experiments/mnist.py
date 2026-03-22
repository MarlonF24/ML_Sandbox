import sys, os, logging
import numpy as np
from typing import cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network import FFNeuralNetwork
from metric_functions import categorical_accuracy_metric

logging.basicConfig(
    level=logging.INFO,
    filename="mnist_training.log",
    filemode='w',
)

def run_experiment():
    print("Fetching MNIST dataset (this may take a minute)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    
    # MNIST images are 28x28 = 784 pixels
    raw_x = cast(np.ndarray, mnist.data)
    raw_y = cast(np.ndarray, mnist.target).reshape(-1, 1)

    print(f"Data shape: {raw_x.shape}")

    # Normalization (Crucial for MNIST)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(raw_x)
    
    # One-Hot Encoding for 10 classes (0-9)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(raw_y)

    # Splits (60k train, 10k test as per standard MNIST convention)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_onehot, test_size=10000, random_state=42
    )

    # Model initialization
    # 784 input neurons, 128/64 hidden, 10 output
    model = FFNeuralNetwork.initialise_with_random_small_parameters(
        layer_sizes=[784, 128, 64, 10],
        metric_function=categorical_accuracy_metric,
        weights_std=0.01
    )

    print("Starting training on MNIST (28x28 images)...")
    # Reduced epochs for first run as MNIST is much larger
    model.fit(
        features=cast(np.ndarray, x_train), 
        labels=cast(np.ndarray, y_train), 
        epochs=5, 
        batch_size=64, 
        learning_rate=0.1, 
        momentum_gamma=0.9
    )

    test_preds = model.predict(cast(np.ndarray, x_test))
    final_acc = categorical_accuracy_metric(test_preds.T, cast(np.ndarray, y_test))
    print(f"\nFinal MNIST Test Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    run_experiment()
