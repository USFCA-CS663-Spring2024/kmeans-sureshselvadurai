import numpy as np
from modules.kmeans._kmeans import KMeans

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, tol=1e-5, random_state=42, verbose=1, balanced=True)
    labels, cluster_centers_ = kmeans.fit(X)

    labels = kmeans.predict(X)
    print(labels)

    print("Cluster centers:", kmeans.cluster_centers_)
    print("Inertia:", kmeans.inertia_)

