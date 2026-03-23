import numpy as np
import logging

from network import FFNeuralNetwork
from metric_functions import get_binary_accuracy_metric
from activation_functions import get_stable_sigmoid_activation
from loss_functions import squareLossFunction

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    xor_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    xor_labels = np.array([[0], [1], [1], [0]], dtype=np.float64)

    model = FFNeuralNetwork.initialise_with_random_small_parameters(
        layer_sizes=[2, 4, 1], 
        activation_function=get_stable_sigmoid_activation(),
        loss_function=squareLossFunction,
        metric_function=get_binary_accuracy_metric(threshold=0.5),
        weights_std=1.0, 
        biases_std=1.0
    )
    model.fit(xor_features, xor_labels, epochs=600, batch_size=4, learning_rate=0.5, momentum_gamma=0.9)

    print("\nPredictions on XOR dataset:")
    print(model.predict(xor_features))