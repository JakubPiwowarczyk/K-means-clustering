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

In the [data](data) folder you can find [output.txt](data/output.txt) and [clusters.png](data/clusters.png) files, that have been generated based on [data.txt](data/data.txt) file with sample data.
