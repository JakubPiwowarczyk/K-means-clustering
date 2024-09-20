"""
This module implements the k-means clustering algorithm for grouping data points into k clusters
based on Euclidean distance.

The script takes four command-line arguments:
1. The path to the input file containing the data points.
2. The path to the output file where the final cluster centroids will be written.
3. The path where the resulting plot image will be saved.
4. The number of clusters (k) to generate.

The input file should be a text file where each line represents a data point in 2D space.

Example usage:
    python main.py input.txt output.txt result_image.png 3

Python 3.11
"""


import copy
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    """
    Represents a cluster in the k-means clustering algorithm.

    Attributes:
        centroid (np.ndarray): The centroid of the cluster.
        points (np.ndarray): An array of points assigned to the cluster.
    """
    def __init__(self, centroid: np.ndarray):
        self.centroid = centroid
        self.points: np.ndarray = np.ndarray((0, 2))

    def __str__(self):
        return f'Cluster {self.centroid}, {self.points}'

    def add_point(self, point: np.ndarray):
        """
        Add a data point to the cluster.

        Args:
            point (np.ndarray): The data point to add to the cluster.

        Returns:
            None
        """
        self.points = np.vstack([self.points, point])

    def clear_points(self):
        """
        Clear all data points from the cluster.

        Returns:
            None
        """
        self.points = np.ndarray((0, 2))

    def euclidean_distance(self, p: np.ndarray):
        """
        Calculate the Euclidean distance between the centroid of the cluster and a given point.

        Args:
            p (np.ndarray): The point for which to calculate the distance from the cluster centroid.

        Returns:
            float: The Euclidean distance between the cluster centroid and the given point.
        """
        result = np.sum((self.centroid - p) ** 2)
        return math.sqrt(result)

    def update_centroid(self):
        """
        Update the centroid of the cluster based on the mean of its data points.

        Returns:
            None
        """
        self.centroid[0] = self.points[:, 0].mean()
        self.centroid[1] = self.points[:, 1].mean()


def generate_clusters_with_random_centroids(k: int, data_array: np.ndarray) -> dict[int, Cluster]:
    """
    Generate clusters with random centroids.

    Args:
        k (int): The number of clusters to generate.
        data_array (np.ndarray): The array of data points from which centroids
            will be randomly selected.

    Returns:
        dict[int, Cluster]: A dictionary containing the generated clusters
            with their corresponding indices as keys.
    """
    indexes = np.random.choice(len(data_array), size=k, replace=False)
    centroids = data_array[indexes]
    clusters_dict: dict[int, Cluster] = {}
    idx = 0
    for i in centroids:
        clusters_dict[idx] = Cluster(i)
        idx += 1
    return clusters_dict


def assign_points_to_clusters(data_array: np.ndarray, clusters_dict: dict[int, Cluster]) -> None:
    """
    Assign each data point in the data array to the closest cluster.

    Args:
        data_array (np.ndarray): An array of data points to be assigned to clusters.
        clusters_dict (dict[int, Cluster]): A dictionary containing clusters,
            where the keys are cluster indices and the values are Cluster objects.

    Returns:
        None
    """
    for cluster in clusters_dict.values():
        cluster.clear_points()
    for i in data_array:
        closest_distance = clusters_dict[0].euclidean_distance(i)
        closest_cluster: Cluster = clusters_dict[0]
        for j in clusters_dict:
            distance = clusters_dict[j].euclidean_distance(i)
            if distance < closest_distance:
                closest_cluster = clusters_dict[j]
                closest_distance = distance
        closest_cluster.add_point(i)


def find_final_clusters(data_array: np.ndarray, clusters_dict: dict[int, Cluster],
                        color_map: np.ndarray, image_path: str) -> None:
    """
    Apply k-means clustering algorithm to find final clusters and save the result as an image.

    Args:
        data_array (np.ndarray): An array of data points to be clustered.
        clusters_dict (dict[int, Cluster]): A dictionary containing initial clusters,
            where the keys are cluster indices and the values are Cluster objects.
        color_map (np.ndarray): An array mapping cluster indices to colors for visualization.
        image_path (str): The path to save the resulting image.

    Returns:
        None
    """
    keep_going: bool = True
    last_iteration: dict[int, Cluster] = copy.deepcopy(clusters_dict)
    while keep_going:
        assign_points_to_clusters(data_array, clusters_dict)
        for i in clusters_dict:
            clusters_dict[i].update_centroid()

        keep_going = False
        for i in clusters_dict:
            x = last_iteration[i].centroid[0]
            y = last_iteration[i].centroid[1]
            if x != clusters_dict[i].centroid[0] or y != clusters_dict[i].centroid[1]:
                keep_going = True

        last_iteration = copy.deepcopy(clusters_dict)

    for i in clusters_dict:
        plt.scatter(clusters_dict[i].centroid[0], clusters_dict[i].centroid[1], c=color_map[i])
        plt.scatter(clusters_dict[i].points[:, 0], clusters_dict[i].points[:, 1],
                    alpha=0.3, c=color_map[i])
    plt.savefig(image_path)


if __name__ == '__main__':
    input_file_path: str = sys.argv[1]
    output_file_path: str = sys.argv[2]
    image_file_path: str = sys.argv[3]
    number_of_clusters: int = int(sys.argv[4])

    data = np.loadtxt(input_file_path)
    clusters = generate_clusters_with_random_centroids(number_of_clusters, data)
    colormap = np.random.rand(number_of_clusters, 3)

    find_final_clusters(data, clusters, colormap, image_file_path)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(f'Number of clusters: {number_of_clusters}\n')
        for index, clust in clusters.items():
            file.write(f'Cluster {index + 1} {clust.centroid[0]} {clust.centroid[1]}\n')
