import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

import agglomerative_clustering

dataset  = pd.read_csv("../data/Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#plt.title("Dendogram")
#plt.xlabel("Customers")
#plt.ylabel("Euclidean distances")
#plt.show()

# train the model
agg_model = AgglomerativeClustering(n_clusters=5)
labels = agg_model.fit_predict(X)

#add label column in dataset
supervised = dataset
supervised['cluster_group'] = labels
#print(supervised.columns)

#draw cluster diagram
lmplot = sns.lmplot(data=supervised, x=supervised.columns[3], y=supervised.columns[4], hue=supervised.columns[5],
                    fit_reg=False, legend=True, facet_kws={'legend_out': True})
plt.show()


