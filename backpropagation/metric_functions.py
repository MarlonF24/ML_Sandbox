from dataclasses import dataclass
from typing import Callable

import numpy as np
from jaxtyping import Float64, jaxtyped # type: ignore
from beartype import beartype

@dataclass(frozen=True)
class MetricFunction:
    name: str
    _function: Callable[[Float64[np.ndarray, "batch_size label_size"], Float64[np.ndarray, "batch_size label_size"]], float]

    @jaxtyped(typechecker=beartype)
    def __call__(self, predicted: Float64[np.ndarray, "batch_size label_size"], target: Float64[np.ndarray, "batch_size label_size"]) -> float:
        return self._function(predicted, target)
    