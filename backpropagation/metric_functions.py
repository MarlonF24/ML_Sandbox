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


def get_binary_accuracy_metric(threshold: float=0.5) -> MetricFunction:
    return MetricFunction(
        name="Binary Accuracy",
        _function=lambda test_preds, labels: ((test_preds > threshold) == labels).mean()
    )


categorical_accuracy_metric = MetricFunction(
    name="Categorical Accuracy",
    _function=lambda test_preds, labels: (np.argmax(test_preds, axis=1) == np.argmax(labels, axis=1)).mean()
)

def get_f1_score_metric(threshold: float=0.5) -> MetricFunction:
    return MetricFunction(
        name="F1 Score", 
        _function=lambda test_preds, labels: (2 * ((test_preds > threshold) * labels).sum()) / (((test_preds > threshold).sum() + labels.sum()) + 1e-10)
    )

