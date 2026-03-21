import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from jaxtyping import Float64, jaxtyped  # type: ignore

from network import FFNeuralNetwork
from metric_functions import MetricFunction

# Silence Matplotlib debug logs globally
logging.getLogger('matplotlib').setLevel(logging.WARNING)

@jaxtyped(typechecker=beartype)
def generate_hyperplane_dataset(
    num_instances: int, 
    num_features: int, 
    num_hyperplanes: int, 
    complexity: float = 0.5
) -> tuple[Float64[np.ndarray, "num_instances num_features"], Float64[np.ndarray, "num_instances 1"]]:
    """
    Generates data where the label is a Boolean function (disjunction of intersections) 
    of the hyperplane results.
    """
    features = np.random.uniform(-1, 1, size=(num_instances, num_features)).astype(np.float64)
    hyperplane_normals = np.random.normal(size=(num_hyperplanes, num_features)).astype(np.float64)
    hyperplane_offsets = np.random.uniform(-0.5, 0.5, size=(num_hyperplanes,)).astype(np.float64)

    # (num_instances, num_hyperplanes)
    codes = (features @ hyperplane_normals.T + hyperplane_offsets >= 0).astype(int)

    num_possible_regions = 2 ** num_hyperplanes
    rule_book = (np.random.rand(num_possible_regions) < complexity).astype(np.float64)

    powers_of_two = 2 ** np.arange(num_hyperplanes)[::-1]
    region_indices = np.dot(codes, powers_of_two)
    
    labels = rule_book[region_indices].reshape(-1, 1)

    visualize_hyperplane_dataset(features, labels, hyperplane_normals, hyperplane_offsets)

    return features, labels


def visualize_hyperplane_dataset(
    features: np.ndarray, 
    labels: np.ndarray, 
    hyperplane_normals: np.ndarray, 
    hyperplane_offsets: np.ndarray
) -> None:
    num_features = features.shape[1]
    num_hyperplanes = hyperplane_normals.shape[0]

    fig: Any = plt.figure(figsize=(10, 8))
    labels_flat = labels.flatten()
    
    match num_features:
        case 1:
            ax = fig.add_subplot(111)
            ax.scatter(features[labels_flat == 0, 0], np.zeros_like(features[labels_flat == 0, 0]), c='red', label='Class 0', alpha=0.6)
            ax.scatter(features[labels_flat == 1, 0], np.zeros_like(features[labels_flat == 1, 0]), c='blue', label='Class 1', alpha=0.6)
            
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[0] != 0:
                    ax.axvline(x=float(-b / w[0]), color='black', linestyle='--', alpha=0.4)
            ax.set_yticks([])

        case 2:
            ax = fig.add_subplot(111)
            ax.scatter(features[labels_flat == 0, 0], features[labels_flat == 0, 1], c='red', label='Class 0', alpha=0.6)
            ax.scatter(features[labels_flat == 1, 0], features[labels_flat == 1, 1], c='blue', label='Class 1', alpha=0.6)
            
            x_range = np.array([-1.0, 1.0])
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[1] != 0:
                    y_vals = -(w[0] * x_range + b) / w[1]
                    ax.plot(x_range, y_vals, '--', alpha=0.4)

        case 3:
            import mpl_toolkits.mplot3d  # type: ignore # noqa: F401
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features[labels_flat == 0, 0], features[labels_flat == 0, 1], features[labels_flat == 0, 2], c='red', label='Class 0', alpha=0.6)
            ax.scatter(features[labels_flat == 1, 0], features[labels_flat == 1, 1], features[labels_flat == 1, 2], c='blue', label='Class 1', alpha=0.6)
            
            x_ticks = np.linspace(-1, 1, 10)
            X, Y = np.meshgrid(x_ticks, x_ticks)
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[2] != 0:
                    Z = -(w[0] * X + w[1] * Y + b) / w[2]
                    ax.plot_surface(X, Y, Z, alpha=0.2)
        
        case _:
            print(f"Visualization not supported for {num_features} features.")
            plt.close(fig)
            return

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    if num_features == 3:
        ax.set_zlim(-1.0, 1.0)

    plt.legend()
    plt.title(f"{num_features}D Hyperplane Disjunction Visualization")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
        try:
            plt.show()
        except Exception:
            pass

    save_path = f"hyperplane_vis_{num_features}d.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)


def initialise_for_hyperplanes(features: int, hyperplanes_needed: int, lr: float = 0.02) -> FFNeuralNetwork:
    # Metric function: Accuracy for binary classification
   

    # Basic heuristic: [#features, hyperplanes, logic_gates, 1]
    return FFNeuralNetwork(
        layer_sizes=[features, hyperplanes_needed, 2 ** hyperplanes_needed, 1],
        learning_rate=lr,
        metric_function=MetricFunction(name="Accuracy", _function=lambda preds, labels: ((preds > 0.5) == labels).mean())
    )

if __name__ == "__main__":

    logging.basicConfig(filename='training.log', level=logging.INFO)
    num_features = 2
    num_hyperplanes = 10

    features, labels = generate_hyperplane_dataset(num_instances=1000, num_features=num_features, num_hyperplanes=num_hyperplanes, complexity=0.5)

    model = initialise_for_hyperplanes(features=num_features, hyperplanes_needed=num_hyperplanes, lr=0.2)

    model.fit(features, labels, epochs=100, batch_size=32)
