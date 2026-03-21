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
    complexity: float = 0.5,
    plot: bool = True
) -> tuple[Float64[np.ndarray, "num_instances num_features"], Float64[np.ndarray, "num_instances 1"], np.ndarray, np.ndarray]:
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
    num_positive_regions = int(num_possible_regions * complexity)
    
    # Select a fixed fraction of regions to be positive (disjunction)
    positive_indices = np.random.choice(num_possible_regions, size=num_positive_regions, replace=False)
    rule_book = np.zeros(num_possible_regions, dtype=np.float64)
    rule_book[positive_indices] = 1.0

    powers_of_two = 2 ** np.arange(num_hyperplanes)[::-1]
    region_indices = np.dot(codes, powers_of_two)
    
    labels = rule_book[region_indices].reshape(-1, 1)

    if plot:
        visualize_hyperplane_dataset(features, labels, hyperplane_normals, hyperplane_offsets)

    return features, labels, hyperplane_normals, hyperplane_offsets


def visualize_hyperplane_dataset(
    features: np.ndarray, 
    labels: np.ndarray, 
    hyperplane_normals: np.ndarray, 
    hyperplane_offsets: np.ndarray,
    model_hyperplanes: tuple[np.ndarray, np.ndarray] | None = None,
    save_suffix: str = ""
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
            
            # Ground Truth Hyperplanes
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[0] != 0:
                    ax.axvline(x=float(-b / w[0]), color='black', linestyle='--', alpha=0.4)
            
            # Model Learned Hyperplanes (First Layer)
            if model_hyperplanes:
                m_w, m_b = model_hyperplanes
                for i in range(m_w.shape[0]):
                    if m_w[i, 0] != 0:
                        ax.axvline(x=float(-m_b[i] / m_w[i, 0]), color='green', linestyle='-', alpha=0.3)
            ax.set_yticks([])

        case 2:
            ax = fig.add_subplot(111)
            ax.scatter(features[labels_flat == 0, 0], features[labels_flat == 0, 1], c='red', label='Class 0', alpha=0.6)
            ax.scatter(features[labels_flat == 1, 0], features[labels_flat == 1, 1], c='blue', label='Class 1', alpha=0.6)
            
            x_range = np.array([-1.0, 1.0])
            # Ground Truth
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[1] != 0:
                    y_vals = -(w[0] * x_range + b) / w[1]
                    ax.plot(x_range, y_vals, '--', color='black', alpha=0.4)
            
            # Model Learned
            if model_hyperplanes:
                m_w, m_b = model_hyperplanes
                for i in range(m_w.shape[0]):
                    if m_w[i, 1] != 0:
                        y_vals = -(m_w[i, 0] * x_range + m_b[i]) / m_w[i, 1]
                        ax.plot(x_range, y_vals, '-', color='green', alpha=0.3)

        case 3:
            import mpl_toolkits.mplot3d  # type: ignore # noqa: F401
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features[labels_flat == 0, 0], features[labels_flat == 0, 1], features[labels_flat == 0, 2], c='red', label='Class 0', alpha=0.6)
            ax.scatter(features[labels_flat == 1, 0], features[labels_flat == 1, 1], features[labels_flat == 1, 2], c='blue', label='Class 1', alpha=0.6)
            
            x_ticks = np.linspace(-1, 1, 10)
            X, Y = np.meshgrid(x_ticks, x_ticks)
            # Ground Truth
            for i in range(num_hyperplanes):
                w, b = hyperplane_normals[i], hyperplane_offsets[i]
                if w[2] != 0:
                    Z = -(w[0] * X + w[1] * Y + b) / w[2]
                    ax.plot_surface(X, Y, Z, color='black', alpha=0.1)
            
            # Model Learned
            if model_hyperplanes:
                m_w, m_b = model_hyperplanes
                for i in range(m_w.shape[0]):
                    if m_w[i, 2] != 0:
                        Z = -(m_w[i, 0] * X + m_w[i, 1] * Y + m_b[i]) / m_w[i, 2]
                        ax.plot_surface(X, Y, Z, color='green', alpha=0.1)
        
        case _:
            print(f"Visualization not supported for {num_features} features.")
            plt.close(fig)
            return

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    if num_features == 3:
        ax.set_zlim(-1.0, 1.0)

    plt.title(f"{num_features}D Hyperplane Comparison (Black=Truth, Green=Model)")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
        try:
            plt.show()
        except Exception:
            pass

    save_path = f"hyperplane_vis_{num_features}d{save_suffix}.png"
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
    # Configure logging to write to both file and terminal
    logging.basicConfig(
        level=logging.INFO,
        # filename="training_log.txt",
        )

    num_features = 2
    num_hyperplanes = 1

    # Generate data BUT disable plotting inside the generator
    features, labels, hp_n, hp_o = generate_hyperplane_dataset(
        num_instances=1000, 
        num_features=num_features, 
        num_hyperplanes=num_hyperplanes, 
        complexity=0.5,
        plot=False
    )

    # model = initialise_for_hyperplanes(features=num_features, hyperplanes_needed=num_hyperplanes, lr=0.1)
    model = FFNeuralNetwork(
        layer_sizes=[num_features, 1],
        learning_rate=0.1,
        metric_function=MetricFunction(name="Accuracy", _function=lambda preds, labels: ((preds > 0.5) == labels).mean())
    )

    # Train
    model.fit(features, labels, epochs=200, batch_size=32)

    # Visualization in the end with model hyperplanes
    # We grab the weights/biases from the first layer which represent hyperplanes
    first_layer = model.processing_layers[0]
    model_hp = (first_layer.weights, first_layer.biases)

    visualize_hyperplane_dataset(
        features, labels, hp_n, hp_o, 
        model_hyperplanes=model_hp,
        save_suffix="_final"
    )
    

    model.fit(features, labels, epochs=100, batch_size=32)
