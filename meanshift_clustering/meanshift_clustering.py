import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load dataset
dataset = pd.read_csv("../data/Mall_Customers.csv")

# Select features for clustering
X = dataset.iloc[:, [3, 4]].values

# Estimate bandwidth for Mean Shift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))

# Train the Mean Shift model
mean_shift_model = MeanShift(bandwidth=bandwidth)
labels = mean_shift_model.fit_predict(X)

# Add cluster labels to dataset
dataset['cluster_group'] = labels

# Visualize clusters
lmplot = sns.lmplot(data=dataset, x=dataset.columns[3], y=dataset.columns[4], hue="cluster_group",
                    fit_reg=False, legend=True, palette="Set1", facet_kws={'legend_out': True})

plt.title("Mean Shift Clustering Results")
plt.show()
