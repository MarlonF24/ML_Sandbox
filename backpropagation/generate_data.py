import numpy as np
from jaxtyping import Float64 

def generate_dataset_for_hyperplanes(num_features: int, 
                                     hyperplanes_needed: int, 
                                     instances_per_region_mean: int,
                                     instances_per_region_sd: float = 0.0
                                     ) -> tuple[Float64[np.ndarray, "dataset_size num_features"], Float64[np.ndarray, "dataset_size total_hyperplanes"]]:
    """
    Generate a dataset for the FFNN initialised for the hyperplane problem, where the task is to learn to classify points based on which side of a set of hyperplanes they fall on. The output labels are the binary encodings of which region defined by the hyperplanes the point falls into.

    Args:
        num_features: number of features (dimensions) for each instance
        hyperplanes_needed: number of hyperplanes needed to create the regions (the more hyperplanes, the more complex the decision boundary and more regions there are)
        instances_per_region_mean: average number of instances to generate for each region
        instances_per_region_sd: standard deviation of the number of instances per region. Defaults to 0.0 (fixed number).
    """
    total_hyperplanes = hyperplanes_needed
    total_regions = 2 ** total_hyperplanes

    # Randomly generate hyperplane-specific instance counts based on normal distribution
    if instances_per_region_sd > 0:
        counts = np.random.normal(float(instances_per_region_mean), float(instances_per_region_sd), int(total_regions))
        instances_counts = np.round(np.maximum(counts, 0)).astype(np.int32)
    else:
        instances_counts = np.full(int(total_regions), instances_per_region_mean, dtype=np.int32)

    # Randomly generate hyperplanes by sampling normal vectors and offsets
    hyperplane_normals = np.random.normal(size=(total_hyperplanes, num_features)).astype(np.float64)
    hyperplane_offsets = np.random.normal(size=(total_hyperplanes,)).astype(np.float64)

    total_dataset_size: int = int(np.sum(instances_counts))

    features: Float64[np.ndarray, "total_dataset_size num_features"] = np.zeros((total_dataset_size, num_features), dtype=np.float64)
    labels: Float64[np.ndarray, "total_dataset_size total_hyperplanes"] = np.zeros((total_dataset_size, total_hyperplanes), dtype=np.float64)

    curr_idx: int = 0
    for region in range(total_regions):
        num_to_generate: int = int(instances_counts[region])
        if num_to_generate <= 0:
            continue

        # The region label is the binary representation of the region index
        target_label = np.array(list(np.binary_repr(region, width=total_hyperplanes)), dtype=np.float64)
        
        found_instances = 0
        region_features = np.zeros((num_to_generate, num_features), dtype=np.float64)
        
        # Rejection sampling
        attempts = 0 
        max_attempts = 100 
        
        while found_instances < num_to_generate and attempts < max_attempts:
            batch_size: int = int(max(num_to_generate * 2, 100))
            candidate_points = np.random.normal(size=(batch_size, num_features)).astype(np.float64)
            
            hp_results = (hyperplane_normals @ candidate_points.T + hyperplane_offsets[:, np.newaxis]) >= 0
            candidate_labels = hp_results.T.astype(np.float64)
            
            matches = np.all(candidate_labels == target_label, axis=1)
            valid_points = candidate_points[matches]
            
            take: int = int(min(len(valid_points), num_to_generate - found_instances))
            if take > 0:
                region_features[found_instances:found_instances+take] = valid_points[:take]
                found_instances += take
            
            attempts += 1

        idx_end: int = curr_idx + num_to_generate
        features[curr_idx:idx_end] = region_features
        labels[curr_idx:idx_end] = target_label[np.newaxis, :]
        curr_idx = idx_end

    return features, labels

