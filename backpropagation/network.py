
from activation_functions import *
from loss_functions import *
from metric_functions import *

from typing import Sequence
from collections import deque
from dataclasses import dataclass

from jaxtyping import Float64, Int32, jaxtyped # type: ignore
from beartype import beartype

import numpy as np, logging



npt = np.typing


@dataclass
class ProcessingLayer:
    weights: Float64[np.ndarray, "curr_layer_size prev_layer_size"]
    biases: Float64[np.ndarray, "curr_layer_size"]



@dataclass
class FFLayerData:
    net_inputs: Float64[np.ndarray, "layer_size batch_size"] | None # None for input layer, as no net inputs
    activations: Float64[np.ndarray, "layer_size batch_size"]


@dataclass
class LayerGradients:
    weight_gradients: Float64[np.ndarray, "curr_layer_size prev_layer_size batch_size"]
    bias_gradients: Float64[np.ndarray, "layer_size batch_size"]



@jaxtyped(typechecker=beartype)
class FFNeuralNetwork:

        
    
    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 layer_sizes: list[int] | Int32[np.ndarray, "num_layers"], 
                 activation_function: ActivationFunction = get_stable_sigmoid_activation(),
                 loss_function: LossFunction = get_square_loss_function(),
                 metric_function: MetricFunction | None = None,
                 learning_rate: float = 0.02) -> None:
        """
        Args:
            layer_sizes: List/vector of layer sizes
            activation_function: Which function to use to normalise the netinputs of nodes
            loss_function: Which function to compute the loss with.
            metric_function: Optional human-readable evaluation metric (e.g. Accuracy). 
                             Must take (predictions, labels) and return a scalar.
            learning_rate: Step size for gradient descent.
        """


        self.layer_sizes = layer_sizes

        self.processing_layers: list[ProcessingLayer] = [ # expluding the input layer, as no weights or biases
            ProcessingLayer(
                weights=np.random.normal(0, 0.1, (layer_size, previous_layer_size)),
                biases=np.zeros(layer_size)
            )
            for previous_layer_size, layer_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.activation_function: ActivationFunction = activation_function
        self.loss_function: LossFunction = loss_function
        self.metric_function: MetricFunction | None = metric_function
        self.learning_rate: float = learning_rate



    
    @jaxtyped(typechecker=beartype)
    def fit(self, features: Float64[np.ndarray, "dataset_size num_features"], labels: Float64[np.ndarray, "dataset_size label_size"], epochs: int, batch_size: int):


        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            
            dataset_size: int = features.shape[0]
            indices: Int32[np.ndarray, "dataset_size"] = np.arange(dataset_size)

            np.random.shuffle(indices)
            
            for batch_cutoff in range(0, dataset_size, batch_size):
                batch_idx = indices[batch_cutoff:batch_cutoff + batch_size]
                X_batch = features[batch_idx]
                y_batch = labels[batch_idx]

                ff_data = self.feed_forward(X_batch)
                self.backward_phase(ff_data, y_batch)
        

    
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
            
            curr_layer_activation = self.activation_function(curr_net_inputs)
            FF_Data_list.append(FFLayerData(net_inputs=curr_net_inputs, activations=curr_layer_activation))


        return FF_Data_list

    
    @jaxtyped(typechecker=beartype)
    def backward_phase(self, ff_data: list[FFLayerData], labels: Float64[np.ndarray, "batch_size label_size"]):
        """
        Do backward phase after FF for a batch_size of instances.

        Args:
            ff_data: list of FFLayerData returned by feed_forward
            labels: 2D array, where each row is the label vector of an instance (vectors must match size of output layer)
        """

        output_layer_data = ff_data[-1]

        # (batch_size), (batch_size, output_layer_size) = loss((batch_size, output_layer_size), (batch_size, output_layer_size))
        loss = self.loss_function(output_layer_data.activations.T, labels)

        metric_info = ""
        if self.metric_function:
            m_val = self.metric_function(output_layer_data.activations.T, labels)
            metric_info = f" | {self.metric_function.name}: {m_val:.4f}"

        loss_derivatives = self.loss_function.derivative(output_layer_data.activations.T, labels)

        logging.info(f"Batch Loss ({self.loss_function.name}): {loss.mean():.4f}{metric_info}")

    
        # some activation functions derivatives can use the previously computed activations 
        # (output_layer_size, batch_size) = (output_layer_size, batch_size) * g'((output_layer_size, batch_size)) 
    
        assert output_layer_data.net_inputs is not None
        curr_layer_error_terms = loss_derivatives.T * self.activation_function.derivative(output_layer_data.net_inputs, output_layer_data.activations)

        gradients_list: deque[LayerGradients] = deque()

        
        for prev_layer_ff_data, curr_layer in zip(reversed(ff_data[:-1]), reversed(self.processing_layers)):
            
            gradients = self.get_gradients_for_layer(curr_layer_error_terms, prev_layer_ff_data.activations)
            
            gradients_list.appendleft(gradients)

            if prev_layer_ff_data.net_inputs is not None: # if not input layer
                curr_layer_error_terms = self.backpropagate(curr_layer_error_terms, curr_layer.weights, prev_layer_ff_data)
        

        self.gradient_descent(gradients_list)



    
    @jaxtyped(typechecker=beartype)
    def backpropagate(self, 
                      curr_layer_error_terms: Float64[np.ndarray, "curr_layer_size batch_size"], 
                      curr_layer_weights: Float64[np.ndarray, "curr_layer_size prev_layer_size"],
                      prev_ff_layer_data: FFLayerData
                      ) -> Float64[np.ndarray, "prev_layer_size batch_size"]:
        
        # NOTE: as said, some activation functions calculate the derivative with their own output 
        # (prev_layer_size, batch_size) = g'((prev_layer_size, batch_size))
        assert prev_ff_layer_data.net_inputs is not None
        dO_j__dh_j = self.activation_function.derivative(prev_ff_layer_data.net_inputs, prev_ff_layer_data.activations) 

        # (prev_layer_size, batch_size) = (prev_layer_size, curr_layer_size) @ (curr_layer_size, batch_size)
        dL__dO_j = curr_layer_weights.T @ curr_layer_error_terms 

        prev_layer_error_terms = dL__dO_j * dO_j__dh_j 

        return prev_layer_error_terms
    

    
    @jaxtyped(typechecker=beartype)
    def get_gradients_for_layer(self, 
                                error_terms: Float64[np.ndarray, "curr_layer_size batch_size"], 
                                prev_layer_activations: Float64[np.ndarray, "prev_layer_size batch_size"]
                                ) -> LayerGradients:

        # (curr_layer_size, prev_layer_size, batch_size) = np.einsum('cb,pb->cpb', (curr_layer_size, batch_size), (prev_layer_size, batch_size)) 
        weight_gradients =  np.einsum('cb,pb->cpb', error_terms, prev_layer_activations)
        
        bias_gradients = error_terms
        
        return LayerGradients(weight_gradients, bias_gradients)
    

    
    @jaxtyped(typechecker=beartype)
    def gradient_descent(self, gradients_list: Sequence[LayerGradients]) -> None:

        for processing_layer, gradients in zip(self.processing_layers, gradients_list):

            processing_layer.weights -= self.learning_rate * gradients.weight_gradients.mean(axis=2)
            processing_layer.biases -= self.learning_rate * gradients.bias_gradients.mean(axis=1)



        
if __name__ == "__main__":
    pass

