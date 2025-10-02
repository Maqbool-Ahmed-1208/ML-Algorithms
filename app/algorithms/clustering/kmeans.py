import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_sample_idxs = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_sample_idxs]

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def _calculate_centroids(self, X):
        centroids = []
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            centroids.append(np.mean(cluster_points, axis=0))
        return np.array(centroids)

    def predict(self, X):
        return self._assign_clusters(X)