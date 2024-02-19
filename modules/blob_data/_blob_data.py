from sklearn.datasets import make_blobs


def generate_blob_data(n_samples, centers, n_features=2, cluster_std=0.6, random_state=0):
    """
    Generate blob data using make_blobs function from scikit-learn.

    Args:
    - n_samples (int): The total number of points equally divided among clusters.
    - n_features (int): The number of features for each sample.
    - centers (int or array of shape [n_centers, n_features]): The number of centers to generate or the fixed center locations.
    - cluster_std (float or sequence of floats, optional): The standard deviation of the clusters.
    - random_state (int, RandomState instance or None, optional): Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

    Returns:
    - X (array of shape [n_samples, n_features]): The generated samples.
    - cluster_assignments (array of shape [n_samples]): The integer labels for cluster membership of each sample.
    """
    X, cluster_assignments = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, cluster_assignments
