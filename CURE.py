import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class CURE:
    def __init__(self, n_clusters=3, n_representatives=5, shrink_factor=0.2):
        """
        CURE Clustering Algorithm

        Parameters:
        - n_clusters: int, number of desired clusters
        - n_representatives: int, number of representative points per cluster
        - shrink_factor: float, shrink factor towards the centroid (0 to 1)
        """
        self.n_clusters = n_clusters
        self.n_representatives = n_representatives
        self.shrink_factor = shrink_factor

    def fit(self, X):
        """
        Fit the model to the data matrix X.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), the input data matrix
        """
        # Step 1: Initialize each point as its own cluster
        clusters = [[x] for x in X]

        while len(clusters) > self.n_clusters:
            # Step 2: Find the closest pair of clusters
            min_dist = float('inf')
            merge_idx = (-1, -1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._min_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = (i, j)

            # Step 3: Merge the closest pair
            new_cluster = clusters[merge_idx[0]] + clusters[merge_idx[1]]
            clusters.pop(max(merge_idx))
            clusters.pop(min(merge_idx))
            clusters.append(new_cluster)

        # Step 4: Calculate representative points and apply shrink factor
        self.clusters = [self._get_representative_points(cluster) for cluster in clusters]

    def _min_distance(self, cluster1, cluster2):
        """
        Calculate the minimum distance between two clusters.

        Parameters:
        - cluster1: list of points in the first cluster
        - cluster2: list of points in the second cluster

        Returns:
        - float, the minimum distance between representative points
        """
        distances = cdist(cluster1, cluster2)
        return np.min(distances)

    def _get_representative_points(self, cluster):
        """
        Select representative points and shrink them towards the centroid.

        Parameters:
        - cluster: list of points in the cluster

        Returns:
        - ndarray of representative points
        """
        cluster = np.array(cluster)
        centroid = np.mean(cluster, axis=0)
        distances = cdist([centroid], cluster)[0]
        
        # Get n_representatives farthest points from the centroid
        representative_points = cluster[np.argsort(distances)[-self.n_representatives:]]
        
        # Shrink representative points towards the centroid
        return centroid + self.shrink_factor * (representative_points - centroid)

    def predict(self, X):
        """
        Predict the cluster labels for new data points.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), the input data matrix

        Returns:
        - ndarray of cluster labels
        """
        labels = []
        for x in X:
            distances = [np.min(cdist([x], rep_points)) for rep_points in self.clusters]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def plot_clusters(self, X, labels):
        """
        Plot the clusters for 2D data.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), the input data matrix
        - labels: ndarray of cluster labels
        """
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label+1}')
        
        plt.title("CURE Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()