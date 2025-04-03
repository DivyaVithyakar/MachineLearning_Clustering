import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("../data/Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

list1 = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
    k_means.fit(X)
    list1.append(k_means.inertia_)
#plt.plot(range(1,11),list1)
#plt.title("The Elbow Method")
#plt.xlabel("Number of Cluster")
#plt.ylabel("WCSS")
#plt.show()

k_means = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = k_means.fit_predict(X)
#print(y_kmeans)

supervised = dataset
#Add new coloumn in dataset (cluster group)
supervised['Cluster_group'] = y_kmeans

lmplot = sns.lmplot(data=supervised, x=supervised.columns[3], y=supervised.columns[4], hue=supervised.columns[5],
                    fit_reg=False, legend=True, facet_kws={'legend_out': True})
plt.show()



