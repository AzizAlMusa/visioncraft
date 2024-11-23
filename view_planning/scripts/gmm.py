import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D data
np.random.seed(42)
mean1 = [0, 0, 0]
mean2 = [5, 5, 5]
cov1 = np.eye(3)
cov2 = np.eye(3)
cluster1 = np.random.multivariate_normal(mean1, cov1, 300)
cluster2 = np.random.multivariate_normal(mean2, cov2, 300)
data = np.vstack([cluster1, cluster2])

# Fit the GMM with 2 clusters
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)

# Predict the cluster labels
labels = gmm.predict(data)

# Visualize the clustering result
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("GMM Clustering with 2 Components")
plt.show()

# Print GMM parameters
print("Means of the GMM components:")
print(gmm.means_)

print("\nCovariances of the GMM components:")
print(gmm.covariances_)

print("\nWeights of the GMM components:")
print(gmm.weights_)
