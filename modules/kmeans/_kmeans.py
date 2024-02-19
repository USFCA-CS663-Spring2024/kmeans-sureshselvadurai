# kmeans.py
import numpy as np
from modules._utils import initialize_centroids, balance_clusters
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score, v_measure_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


class KMeans:
    """K-Means clustering algorithm.

    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form.

    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization:
        - 'k-means++' : selects initial cluster centers for k-mean clustering in a smart way.
        - 'random' : chooses k observations (rows) at random from data for the initial centroids.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.

    random_state : int or None, default=None
        Determines random number generation for centroid initialization.
        Use an int for reproducible results across function calls.

    verbose : int, default=0
        Verbosity mode.

    balanced : bool, default=False
        If True, ensures balanced cluster sizes by distributing data counts equally among clusters.

    Attributes:
    -----------
    cluster_centers_ : array, shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : array, shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    """

    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None, verbose=0,
                 balanced=False):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.balanced = balanced
        self.cluster_centers_ = None

    def fit(self, X):
        """Compute k-means clustering.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.

        centroids : array, shape (n_clusters, n_features)
            Coordinates of cluster centers.

        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input data must be a 2-dimensional numpy array")
        if X.shape[0] < self.n_clusters:
            raise ValueError("Number of clusters cannot exceed number of data points")

        # Normalize the data
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

        if self.verbose:
            print("Initialization method: {}".format(self.init))

        rng = np.random.default_rng(self.random_state)
        centroids = initialize_centroids(X_normalized, self.n_clusters, self.init, rng)

        for _ in range(self.max_iter):
            labels, new_centroids = self._expectation_maximization(X_normalized, centroids)
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids

            if self.balanced:
                labels = balance_clusters(X_normalized, labels, centroids, self.balanced)

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = np.sum((X_normalized - centroids[labels]) ** 2)

        return self.labels_, self.cluster_centers_

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.

        """
        labels, _ = self.fit(X)
        return labels

    def _expectation_maximization(self, X, centroids):
        """Expectation-maximization step.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances.

        centroids : array, shape (n_clusters, n_features)
            Cluster centers.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Labels of each point.

        new_centroids : array, shape (n_clusters, n_features)
            Updated cluster centers.

        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return labels, new_centroids

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data points.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.

        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit method first.")
        # Normalize the data before predicting
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
        distances = np.linalg.norm(X_normalized[:, np.newaxis, :] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def plot(data, labels, title='Scatter Plot of Data Points'):
        """Plot 2D scatter plot of data points colored by labels.

        Parameters:
        -----------
        data : numpy array, shape (n_samples, 2)
            Data points.

        labels : numpy array, shape (n_samples,)
            Labels corresponding to each data point.

        """
        if data.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional")

        # Create a scatter plot
        plt.figure(figsize=(5, 4))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], label=label, s=9)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def evaluate_performance(self, true_labels):
        """Evaluate the performance of the clustering algorithm by comparing its labels with true labels.
        Parameters:
        -----------
        true_labels : array-like, shape (n_samples,)
            True cluster labels.
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit method first.")

        true_labels_2d = np.array(true_labels).reshape(-1, 1)  # Reshape to 2D array
        metrics = {
            'Adjusted Rand Index': adjusted_rand_score(true_labels, self.labels_),
            'Silhouette Score': silhouette_score(true_labels_2d, self.labels_),  # Pass the reshaped array
            'Completeness Score': completeness_score(true_labels, self.labels_),
            'Homogeneity Score': homogeneity_score(true_labels, self.labels_),
            'V-measure Score': v_measure_score(true_labels, self.labels_)
        }

        # Print metrics
        for metric, score in metrics.items():
            print(f"{metric}: {score}")

    def confusion_matrix(self, true_labels):
        """Compute and visualize the confusion matrix.

        Parameters:
        -----------
        true_labels : array-like, shape (n_samples,)
            True cluster labels.
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit method first.")

        cm = confusion_matrix(true_labels, self.labels_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
