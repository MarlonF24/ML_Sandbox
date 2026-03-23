
from jaxtyping import jaxtyped # type: ignore
from beartype import beartype
from typing import Callable

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64


# NOTE: here instances are ROWS, so different to most of the network code
@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class LossFunction:
    name: str 
    _function: Callable[[Float64[np.ndarray, "batch_size output_layer_size"], Float64[np.ndarray, "batch_size output_layer_size"]], Float64[np.ndarray, "batch_size"]]
    _derivative: Callable[[Float64[np.ndarray, "batch_size output_layer_size"], Float64[np.ndarray, "batch_size output_layer_size"]], Float64[np.ndarray, "batch_size output_layer_size"]]

    @jaxtyped(typechecker=beartype)
    def __call__(self, predicted: Float64[np.ndarray, "batch_size output_layer_size"], target: Float64[np.ndarray, "batch_size output_layer_size"]) -> Float64[np.ndarray, "batch_size"]:
        return self._function(predicted, target)

    @jaxtyped(typechecker=beartype)
    def derivative(self, predicted: Float64[np.ndarray, "batch_size output_layer_size"], target: Float64[np.ndarray, "batch_size output_layer_size"]) -> Float64[np.ndarray, "batch_size output_layer_size"]:
        return self._derivative(predicted, target)


# NOTE: could technically make more efficient as predicted - target is computed in both function and derivative and usually loss and derivate are computed together, but for forward compatibility we keep them separate 

squareLossFunction = LossFunction(
        name="Square Loss",
        _function=lambda predicted, target: 0.5 * np.sum((predicted - target) ** 2, axis=1), 
        _derivative=lambda predicted, target: predicted - target
    )

cross_entropy_loss_function = LossFunction(
    name="Cross Entropy Loss",
    _function=lambda predicted, target: -np.sum(target * np.log(predicted + 1e-10), axis=1), 
    _derivative=lambda predicted, target: -target / (predicted + 1e-10)
)



if __name__ == "__main__":
    loss_function = squareLossFunction

    predicted = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)
    target = np.array([[1,0,1 ], [0,1,0]], dtype=np.float64)

    print("Loss:", loss_function(predicted, target))
    print("Derivative:", loss_function.derivative(predicted, target))