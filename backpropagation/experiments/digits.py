import sys, os, logging
import numpy as np
from typing import cast
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network import FFNeuralNetwork, categorical_accuracy_metric

logging.basicConfig(
    level=logging.INFO,
    filename="digits_training.log",
    filemode='w',
    )

def run_experiment():
    # Data loading and preparation
    digits_data = load_digits()
    raw_x = cast(np.ndarray, digits_data.data) # type: ignore
    raw_y = cast(np.ndarray, digits_data.target).reshape(-1, 1) # type: ignore

    # Normalization and Encoding
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(raw_x)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(raw_y)

    # Splits
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_onehot, test_size=0.2, random_state=42
    )

    # Model initialization
    model = FFNeuralNetwork.initialise_with_random_small_parameters(
        layer_sizes=[64, 32, 10],
        metric_function=categorical_accuracy_metric,
        weights_std=0.1
    )

    print("Starting training on scikit-learn digits (8x8 images)...")
    model.fit(
        features=cast(np.ndarray, x_train), 
        labels=cast(np.ndarray, y_train), 
        epochs=10, 
        batch_size=32, 
        learning_rate=0.2, 
        momentum_gamma=0.9
    )

    test_preds = model.predict(cast(np.ndarray, x_test))
    final_acc = categorical_accuracy_metric(test_preds.T, cast(np.ndarray, y_test))
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    run_experiment()
