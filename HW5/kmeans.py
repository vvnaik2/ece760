import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(1)
n = 100
sigma_vals = [0.5, 1, 2, 4, 8]
datasets = []
for sigma in sigma_vals:
    P_a = np.random.multivariate_normal([-1, -1], sigma*np.array([[2, 0.5], [0.5, 1]]), n)
    P_b = np.random.multivariate_normal([1, -1], sigma*np.array([[1, -0.5], [-0.5, 2]]), n)
    P_c = np.random.multivariate_normal([0, 1], sigma*np.array([[1, 0], [0, 2]]), n)
    X = np.concatenate((P_a, P_b, P_c), axis=0)
    datasets.append(X)

# Perform K-means clustering and EM algorithm for GMMs on each dataset
k = 3
kmeans_obj_vals = []
kmeans_acc_vals = []
em_obj_vals = []
em_acc_vals = []
for i in range(len(sigma_vals)):
    X = datasets[i]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=2).fit(X)
    kmeans_obj_vals.append(kmeans.inertia_)
    kmeans_acc = np.sum(kmeans.labels_[:n] == 0) + np.sum(kmeans.labels_[n:2*n] == 1) + np.sum(kmeans.labels_[2*n:] == 2)
    kmeans_acc_vals.append(kmeans_acc / (3*n))
    
    # EM algorithm for GMMs
    gm = GaussianMixture(n_components=k, random_state=1).fit(X)
    em_obj_vals.append(gm.lower_bound_)
    em_acc = np.sum(np.argmax(gm.predict_proba(X)[:n], axis=1) == 0) + np.sum(np.argmax(gm.predict_proba(X)[n:2*n], axis=1) == 1) + np.sum(np.argmax(gm.predict_proba(X)[2*n:], axis=1) == 2)
    em_acc_vals.append(em_acc / (3*n))

print("kmeans obj vals :\n")
print(kmeans_obj_vals)
print("GMM obj vals :\n")
print(em_obj_vals)

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(sigma_vals, kmeans_obj_vals, '-o', label='K-means')
axs[0].plot(sigma_vals, em_obj_vals, '-o', label='EM algorithm for GMMs')
axs[0].set_xlabel(r'$\sigma$')
axs[0].set_ylabel('Clustering objective')
axs[0].legend()
axs[1].plot(sigma_vals, kmeans_acc_vals, '-o', label='K-means')
axs[1].plot(sigma_vals, em_acc_vals, '-o', label='EM algorithm for GMMs')
axs[1].set_xlabel(r'$\sigma$')
axs[1].set_ylabel('Clustering accuracy')
axs[1].legend()
plt.show()
