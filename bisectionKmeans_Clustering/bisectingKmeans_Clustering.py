import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
dataset = pd.read_csv("../data/Mall_Customers.csv")

# Select features for clustering
X = dataset.iloc[:, [3, 4]].values

# Function to perform Bisecting K-Means
def bisecting_kmeans(X, k, max_iter=10):
    # Start with a single cluster containing all points
    clusters = [X]
    labels = np.zeros(X.shape[0])

    while len(clusters) < k:
        # Find the cluster to split (we will split the one with the largest variance)
        cluster_to_split = max(clusters, key=lambda cluster: np.var(cluster))

        # Perform KMeans to split the cluster into two
        kmeans = KMeans(n_clusters=2, max_iter=max_iter)
        kmeans.fit(cluster_to_split)

        # Get the cluster labels for the split
        new_labels = kmeans.labels_

        # Update the main labels array with the new cluster assignments
        for i, label in enumerate(new_labels):
            labels[np.where((X == cluster_to_split[i]).all(axis=1))[0]] = label

        # Split the cluster into two
        cluster_1 = cluster_to_split[new_labels == 0]
        cluster_2 = cluster_to_split[new_labels == 1]

        # Remove the old cluster and add the two new clusters
        clusters.remove(cluster_to_split)
        clusters.append(cluster_1)
        clusters.append(cluster_2)

    return labels

# Perform Bisecting K-Means with 3 clusters
labels = bisecting_kmeans(X, k=3)

# Add cluster labels to dataset
dataset['cluster_group'] = labels

# Visualize clusters
plt.figure(figsize=(8, 6))

# Plot the clusters using seaborn's scatterplot
sns.scatterplot(data=dataset, x=dataset.columns[3], y=dataset.columns[4], hue="cluster_group",
                palette="Set1", style="cluster_group", markers=["o", "s", "D", "^"], legend="full")

plt.title("Bisecting K-Means Clustering Results")
plt.xlabel(dataset.columns[3])
plt.ylabel(dataset.columns[4])

# Show the plot
plt.show()
