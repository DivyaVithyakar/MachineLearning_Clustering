import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AffinityPropagation

# Load dataset
dataset = pd.read_csv("../data/Mall_Customers.csv")

# Select features for clustering
X = dataset.iloc[:, [3, 4]].values

# Train the Affinity Propagation model
affinity_model = AffinityPropagation(random_state=42)
labels = affinity_model.fit_predict(X)

# Add cluster labels to dataset
dataset['cluster_group'] = labels

# Visualize clusters
lmplot = sns.lmplot(data=dataset, x=dataset.columns[3], y=dataset.columns[4], hue="cluster_group",
                    fit_reg=False, legend=True, palette="Set1", facet_kws={'legend_out': True})

plt.title("Affinity Propagation Clustering Results")
plt.show()
