import numpy as np

from jaxtyping import jaxtyped, Float64 # type: ignore
from beartype import beartype
from typing import Callable

from dataclasses import dataclass

@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class ActivationFunction:
    name: str
    _function: Callable[[Float64[np.ndarray, "layer_size batch_size"]], Float64[np.ndarray, "layer_size batch_size"]]
    _derivative: Callable[[Float64[np.ndarray, "layer_size batch_size"]], Float64[np.ndarray, "layer_size batch_size"]]
    pass_activations_to_derivative: bool = False

    def derivative(self, net_inputs: Float64[np.ndarray, "layer_size batch_size"], activations: Float64[np.ndarray, "layer_size batch_size"]) -> Float64[np.ndarray, "layer_size batch_size"]:
        if self.pass_activations_to_derivative:
            return self._derivative(activations)
        else:
            return self._derivative(net_inputs)

    def __call__(self, net_inputs: Float64[np.ndarray, "layer_size batch_size"]) -> Float64[np.ndarray, "layer_size batch_size"]:
        return self._function(net_inputs)



def get_step_function(threshold: float) -> ActivationFunction:
    return ActivationFunction(
        name=f"Step Function (threshold={threshold})",
        _function=lambda x: np.heaviside(x, threshold),
        _derivative=lambda x: np.zeros_like(x)
    )


def get_stable_sigmoid_activation() -> ActivationFunction:
    return ActivationFunction(
        name="Stable Sigmoid",
        _function=stable_sigmoid,
        _derivative=lambda x: x * (1 - x),  # can use activations directly
        pass_activations_to_derivative=True
    )

def stable_sigmoid(net_inputs: Float64[np.ndarray, "layer_size batch_size"]) -> Float64[np.ndarray, "layer_size batch_size"]:
    # avoid large exponentials for large negative numbers to avoid overflow

    net_inputs = np.clip(net_inputs, -100, 100)

    return 1 / (1 + np.exp(-net_inputs))
                    


def get_relu_activation() -> ActivationFunction:
    return ActivationFunction(
        name="ReLU",
        _function=lambda x : np.maximum(0, x),
        _derivative=lambda x: (x > 0).astype(np.float64),
        pass_activations_to_derivative=False
    )
