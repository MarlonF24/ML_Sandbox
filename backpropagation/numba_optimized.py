from activation_functions import *
from loss_functions import *
from metric_functions import *
from typing import Sequence, Generator
from collections import deque
from dataclasses import dataclass, field
from jaxtyping import Float64, Int64, jaxtyped
from beartype import beartype
import numpy as np, logging, time
from numba import njit, prange

# AI generated version of network.py with numba compilation

@njit(parallel=True)
def fast_dot_plus_bias(w, a, b):
    res = np.empty((w.shape[0], a.shape[1]), dtype=w.dtype)
    for i in prange(w.shape[0]):
        for j in range(a.shape[1]):
            val = b[i]
            for k in range(w.shape[1]):
                val += w[i, k] * a[k, j]
            res[i, j] = val
    return res

@njit
def fast_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit
def fast_sigmoid_der(y):
    return y * (1.0 - y)

@dataclass
class ProcessingLayer:
    weights: np.ndarray
    biases: np.ndarray

@dataclass
class FFLayerData:
    net_inputs: np.ndarray | None
    activations: np.ndarray

@dataclass
class FFNeuralNetwork:
    processing_layers: list[ProcessingLayer]
    
    def feed_forward(self, X):
        a = X.T
        data = [FFLayerData(None, a)]
        for layer in self.processing_layers:
            z = fast_dot_plus_bias(layer.weights, a, layer.biases)
            a = fast_sigmoid(z)
            data.append(FFLayerData(z, a))
        return data

class Trainer:
    def __init__(self, model):
        self.model = model
        self.mom_w = [np.zeros_like(l.weights) for l in model.processing_layers]
        self.mom_b = [np.zeros_like(l.biases) for l in model.processing_layers]

    def train(self, X, y, epochs, batch_size, lr, gamma):
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                Xi, yi = X[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
                ff = self.model.feed_forward(Xi)
                self.backward(ff, yi, lr, gamma)

    def backward(self, ff, y, lr, gamma):
        a_out = ff[-1].activations
        err = (a_out.T - y).T * fast_sigmoid_der(a_out)
        
        for i in range(len(self.model.processing_layers)-1, -1, -1):
            layer = self.model.processing_layers[i]
            a_prev = ff[i].activations
            
            dw = (err @ a_prev.T) / err.shape[1]
            db = np.sum(err, axis=1) / err.shape[1]
            
            self.mom_w[i] = lr * dw + gamma * self.mom_w[i]
            self.mom_b[i] = lr * db + gamma * self.mom_b[i]
            
            layer.weights -= self.mom_w[i]
            layer.biases -= self.mom_b[i]
            
            if i > 0:
                err = (layer.weights.T @ err) * fast_sigmoid_der(a_prev)

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    layers = [ProcessingLayer(np.random.randn(4,2)*0.1, np.zeros(4)), 
              ProcessingLayer(np.random.randn(1,4)*0.1, np.zeros(1))]
    model = FFNeuralNetwork(layers)
    trainer = Trainer(model)
    
    start = time.time()
    trainer.train(X, y, 5000, 4, 0.5, 0.9)
    print(f"Time: {time.time()-start:.4f}s")
    print(model.feed_forward(X)[-1].activations)
