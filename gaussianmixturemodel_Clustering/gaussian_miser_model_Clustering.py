import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture

# Load dataset
dataset = pd.read_csv("../data/Mall_Customers.csv")

# Select features for clustering
X = dataset.iloc[:, [3, 4]].values

# Train the Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
labels = gmm.fit_predict(X)

# Add cluster labels to dataset
dataset['cluster_group'] = labels

# Visualize clusters
plt.figure(figsize=(8, 6))

# Plot the clusters using seaborn's scatterplot
sns.scatterplot(data=dataset, x=dataset.columns[3], y=dataset.columns[4], hue="cluster_group",
                palette="Set1", style="cluster_group", markers=["o", "s", "D", "^"], legend="full")

plt.title("Gaussian Mixture Model Clustering")
plt.xlabel(dataset.columns[3])
plt.ylabel(dataset.columns[4])

# Show the plot
plt.show()
