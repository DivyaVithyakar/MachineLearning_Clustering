import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

# Load dataset
dataset = pd.read_csv("../data/Mall_Customers.csv")

# Select features for clustering
X = dataset.iloc[:, [3, 4]].values

# Perform hierarchical clustering using Ward's method
ward_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = ward_model.fit_predict(X)

# Add cluster labels to dataset
dataset['cluster_group'] = labels

# Visualize clusters
plt.figure(figsize=(8, 6))

# Plot the clusters using seaborn's scatterplot
sns.scatterplot(data=dataset, x=dataset.columns[3], y=dataset.columns[4], hue="cluster_group",
                palette="Set1", style="cluster_group", markers=["o", "s", "D", "^"], legend="full")

plt.title("Ward's Method (Agglomerative Clustering) Results")
plt.xlabel(dataset.columns[3])
plt.ylabel(dataset.columns[4])

# Show the plot
plt.show()
