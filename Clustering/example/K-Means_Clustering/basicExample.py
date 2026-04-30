from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
             [10, 2], [10, 4], [10, 0]])

# Initailize and fit
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Labels:\n{y_kmeans}")