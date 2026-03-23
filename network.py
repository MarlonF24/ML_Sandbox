from __future__ import annotations


from activation_functions import *
from loss_functions import *
from metric_functions import *

from typing import Sequence, Generator
from collections import deque
from dataclasses import dataclass, field

from jaxtyping import Float64, Int64, jaxtyped # type: ignore
from beartype import beartype

import numpy as np, logging



npt = np.typing

@jaxtyped(typechecker=beartype)
@dataclass
class ProcessingLayer:
    weights: Float64[np.ndarray, "curr_layer_size prev_layer_size"]
    biases: Float64[np.ndarray, "curr_layer_size"]
    activation_function: ActivationFunction = relu_activation
    trainable: bool = True

    @classmethod
    def initialise_with_random_parameters(cls, 
                                          curr_layer_size: int, 
                                          prev_layer_size: int,
                                          weights_mean: float = 0,
                                          weights_std: float = 0.1,
                                          biases_mean: float = 0,
                                          biases_std: float = 0.1,
                                          activation_function: ActivationFunction = relu_activation
                                          ) -> "ProcessingLayer":
        return cls(
            weights=np.random.normal(weights_mean, weights_std, (curr_layer_size, prev_layer_size),),
            biases=np.random.normal(biases_mean, biases_std, curr_layer_size),
            activation_function=activation_function
        )


@jaxtyped(typechecker=beartype)
@dataclass
class FFLayerData:
    net_inputs: Float64[np.ndarray, "layer_size batch_size"] | None # None for input layer, as no net inputs
    activations: Float64[np.ndarray, "layer_size batch_size"]


@jaxtyped(typechecker=beartype)
@dataclass
class LayerGradients:
    weight_gradients: Float64[np.ndarray, "curr_layer_size prev_layer_size"]
    bias_gradients: Float64[np.ndarray, "layer_size"]


@dataclass
@jaxtyped(typechecker=beartype)
class FFNeuralNetwork:
    """
        processing_layers: List of ProcessingLayer objects
        activation_function: Which function to use to normalise the netinputs of nodes
        loss_function: Which function to compute the loss with.
        metric_function: Optional human-readable evaluation metric (e.g. Accuracy). 
                            Must take (predictions, labels) and return a scalar.
    """
    
    processing_layers: list[ProcessingLayer]
    loss_function: LossFunction = squareLossFunction
    metric_function: MetricFunction | None = None


    @classmethod
    def initialise_with_random_small_parameters(cls, 
                                                layer_sizes: list[int],
                                                activation_function: ActivationFunction = relu_activation,
                                                final_activation_function: ActivationFunction | None = None,
                                                loss_function: LossFunction = squareLossFunction,
                                                metric_function: MetricFunction | None = None, 
                                                weights_std: float = 0.1,
                                                biases_std: float = 0.1
                                                ) -> "FFNeuralNetwork":#
        
        processing_layers: list[ProcessingLayer] = [ # expluding the input layer, as no weights or biases
            ProcessingLayer.initialise_with_random_parameters(
                curr_layer_size=layer_size,
                prev_layer_size=previous_layer_size,
                weights_std=weights_std,
                biases_std=biases_std,
                activation_function=activation_function
            )
            for previous_layer_size, layer_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        if final_activation_function is not None:
            processing_layers[-1].activation_function = final_activation_function

        return cls(processing_layers=processing_layers, loss_function=loss_function, metric_function=metric_function) 
    
    @property
    def trainable_processing_layers(self) -> filter[ProcessingLayer]:
        return filter(lambda layer: layer.trainable, self.processing_layers)


    @jaxtyped(typechecker=beartype)
    def fit(self, 
            features: Float64[np.ndarray, "dataset_size num_features"], 
            labels: Float64[np.ndarray, "dataset_size label_size"], 
            epochs: int, 
            batch_size: int,
            learning_rate: float = 0.02,
            momentum_gamma: float = 0.0
            ):

        trainer = Trainer(model=self)

        trainer.train(features, labels, epochs, batch_size, learning_rate, momentum_gamma)
        

    
    @jaxtyped(typechecker=beartype)
    def predict(self, instance_batch: Float64[np.ndarray, "batch_size features"]):
        return self.feed_forward(instance_batch)[-1].activations


    @jaxtyped(typechecker=beartype)
    def feed_forward(self, instance_batch: Float64[np.ndarray, "batch_size features"]) -> list[FFLayerData]:
        """
        Do FF for a batch_size of instances

        Args:
            instance_batch: 2D array where each row is an instance's features

        Returns:
            FF_Data_list: list of FFLayerData objects containing the net inputs and activations for each layer
        """
        curr_layer_activation = instance_batch.T

        FF_Data_list: list[FFLayerData] = [FFLayerData(net_inputs=None, activations=curr_layer_activation)]

        for next_layer in self.processing_layers:
            
            # (next_layer_size, batch_size) = (next_layer_size, curr_layer_size) @ (curr_layer_size, batch_size) + (next_layer_size, batch_size)
            curr_net_inputs = (next_layer.weights @ curr_layer_activation) + next_layer.biases[:, np.newaxis]
            
            curr_layer_activation = next_layer.activation_function(curr_net_inputs)
            FF_Data_list.append(FFLayerData(net_inputs=curr_net_inputs, activations=curr_layer_activation))


        return FF_Data_list




Hyperparam_Generator = Generator[float, None, None]

@jaxtyped(typechecker=beartype)
@dataclass
class Trainer:
    model: FFNeuralNetwork
    default_learning_rate: float = 0.02
    default_momentum_gamma: float = 0.0
    learning_rate: float | Hyperparam_Generator = field(init=False)
    momentum_gamma: float | Hyperparam_Generator = field(init=False)
    momentum_gradients: Sequence[LayerGradients] | None = field(default=None, init=False)

    def __post_init__(self):
        self.learning_rate = self.default_learning_rate
        self.momentum_gamma = self.default_momentum_gamma


    @jaxtyped(typechecker=beartype)
    def train(self, 
            features: Float64[np.ndarray, "dataset_size num_features"], 
            labels: Float64[np.ndarray, "dataset_size label_size"], 
            epochs: int, 
            batch_size: int,
            run_learning_rate: float | Hyperparam_Generator = None,
            run_momentum_gamma: float | Hyperparam_Generator = None
            ):
        
        run_learning_rate = run_learning_rate if run_learning_rate is not None else self.default_learning_rate
        run_momentum_gamma = run_momentum_gamma if run_momentum_gamma is not None else self.default_momentum_gamma


        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            
            self.learning_rate = next(run_learning_rate) if isinstance(run_learning_rate, Generator) else run_learning_rate
            self.momentum_gamma = next(run_momentum_gamma) if isinstance(run_momentum_gamma, Generator) else run_momentum_gamma

            dataset_size: int = features.shape[0]
            indices: Int64[np.ndarray, "dataset_size"] = np.arange(dataset_size)

            np.random.shuffle(indices)
            
            for batch_cutoff in range(0, dataset_size, batch_size):
                batch_idx = indices[batch_cutoff:batch_cutoff + batch_size]
                X_batch = features[batch_idx]
                y_batch = labels[batch_idx]

                ff_data = self.model.feed_forward(X_batch)
                self.backward_phase(ff_data, y_batch, )
    

    @jaxtyped(typechecker=beartype)
    def _compute_initial_error_terms(self, output_layer_ff_data: FFLayerData, labels: Float64[np.ndarray, "batch_size label_size"]) -> Float64[np.ndarray, "output_layer_size batch_size"]:
        loss_function = self.model.loss_function

        output_labels = output_layer_ff_data.activations.T

        # (batch_size), (batch_size, output_layer_size) = loss((batch_size, output_layer_size), (batch_size, output_layer_size))
        loss = loss_function(output_labels, labels)

        metric_info = ""
        if self.model.metric_function:
            m_val = self.model.metric_function(output_labels, labels)
            metric_info = f" | {self.model.metric_function.name}: {m_val:.4f}"

        logging.info(f"Batch Loss ({loss_function.name}): {loss.mean():.4f}{metric_info}")


        loss_derivatives = loss_function.derivative(output_labels, labels)

        assert output_layer_ff_data.net_inputs is not None

        # (output_layer_size, batch_size) = (batch_size, output_layer_size).T
        return loss_derivatives.T * self.model.processing_layers[-1].activation_function.derivative(output_layer_ff_data.net_inputs, output_layer_ff_data.activations)



    @jaxtyped(typechecker=beartype)
    def backward_phase(self, 
                       ff_data: list[FFLayerData], 
                       labels: Float64[np.ndarray, "batch_size label_size"],
                       ):
        """
        Do backward phase after FF for a batch_size of instances.

        Args:
            ff_data: list of FFLayerData returned by feed_forward
            labels: 2D array, where each row is the label vector of an instance (vectors must match size of output layer)
        """

        
        output_layer_ff_data = ff_data[-1]

        curr_layer_error_terms = self._compute_initial_error_terms(output_layer_ff_data, labels)

        curr_layer = self.model.processing_layers[-1]

        gradients_list: deque[LayerGradients] = deque()

        iterator = zip(reversed(ff_data[:-1]), reversed([None] + self.model.processing_layers[:-1]))
        

        for prev_layer_ff_data, prev_layer in iterator:
            
            if curr_layer.trainable:
                gradients = self.get_gradients_for_layer(curr_layer_error_terms, prev_layer_ff_data.activations)
                
                gradients_list.appendleft(gradients)

            if prev_layer_ff_data.net_inputs is not None: # if not input layer
                
                assert prev_layer is not None

                curr_layer_error_terms = self.backpropagate(curr_layer_error_terms, curr_layer.weights, prev_layer.activation_function, prev_layer_ff_data)

                curr_layer = prev_layer 
        

        self.gradient_descent(gradients_list)


    
    @jaxtyped(typechecker=beartype)
    def backpropagate(self, 
                      curr_layer_error_terms: Float64[np.ndarray, "curr_layer_size batch_size"] | None, 
                      curr_layer_weights: Float64[np.ndarray, "curr_layer_size prev_layer_size"],
                      prev_layer_activation_funcion: ActivationFunction,
                      prev_ff_layer_data: FFLayerData
                      ) -> Float64[np.ndarray, "prev_layer_size batch_size"]:
        

        
        # NOTE: as said, some activation functions calculate the derivative with their own output 
        # (prev_layer_size, batch_size) = g'((prev_layer_size, batch_size))
        assert prev_ff_layer_data.net_inputs is not None
        dO_j__dh_j = prev_layer_activation_funcion.derivative(prev_ff_layer_data.net_inputs, prev_ff_layer_data.activations) 

        # (prev_layer_size, batch_size) = (prev_layer_size, curr_layer_size) @ (curr_layer_size, batch_size)
        dL__dO_j = curr_layer_weights.T @ curr_layer_error_terms 

        prev_layer_error_terms = dL__dO_j * dO_j__dh_j 

        return prev_layer_error_terms
    

    
    @jaxtyped(typechecker=beartype)
    def get_gradients_for_layer(self, 
                                error_terms: Float64[np.ndarray, "curr_layer_size batch_size"], 
                                prev_layer_activations: Float64[np.ndarray, "prev_layer_size batch_size"]
                                ) -> LayerGradients:

        # avg gradients across batches:


        # (curr_layer_size, prev_layer_size) = (curr_layer_size, batch_size) @ (batch_size, prev_layer_size)
        weight_gradients = (error_terms @ prev_layer_activations.T) / error_terms.shape[1]
        
        # (curr_layer_size) = np.mean((curr_layer_size, batch_size), axis=1)
        bias_gradients = np.mean(error_terms, axis=1)
        
        return LayerGradients(weight_gradients, bias_gradients)
    

    
    @jaxtyped(typechecker=beartype)
    def gradient_descent(self, gradients_list: Sequence[LayerGradients]) -> None:

        if self.momentum_gamma != 0 and self.momentum_gradients is not None:

            for processing_layer, gradients, momentum_gradients in zip(self.model.trainable_processing_layers, gradients_list, self.momentum_gradients):

                processing_layer.weights -= self.learning_rate * gradients.weight_gradients + self.momentum_gamma * momentum_gradients.weight_gradients
                processing_layer.biases -= self.learning_rate * gradients.bias_gradients + self.momentum_gamma * momentum_gradients.bias_gradients 
        else:

            for processing_layer, gradients in zip(self.model.trainable_processing_layers, gradients_list):

                processing_layer.weights -= self.learning_rate * gradients.weight_gradients
                processing_layer.biases -= self.learning_rate * gradients.bias_gradients

        self.momentum_gradients = gradients_list

    


        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    xor_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    xor_labels = np.array([[0], [1], [1], [0]], dtype=np.float64)

    model = FFNeuralNetwork.initialise_with_random_small_parameters(
        layer_sizes=[2, 4, 1], 
        metric_function=get_binary_accuracy_metric(threshold=0.5), 
        weights_std=1.0, 
        biases_std=1.0
    )
    model.fit(xor_features, xor_labels, epochs=5000, batch_size=4, learning_rate=0.1, momentum_gamma=0.9)

    print("\nPredictions on XOR dataset:")
    print(model.predict(xor_features))

