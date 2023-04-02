import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca(X, d):
    X_centered = X
    X_mean = np.mean(X_centered, axis=0)    

    # Compute SVD of X
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Compute d-dimensional representation of X
    X_pca = X_centered.dot(Vt[:d].T)
    
    # Compute estimated parameters
    explained_variances = (s ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(explained_variances)
    explained_variance_ratios = explained_variances / total_variance
    params = {
        'U': U,
        's': s,
        'Vt': Vt,
        'explained_variances': explained_variances,
        'explained_variance_ratios': explained_variance_ratios,
        'X_mean': X_mean
    }
    
    # Compute reconstructions of d-dimensional representations in D dimensions
    X_reconstructed = X_pca.dot(Vt[:d])
    X_reconstructed += X_mean
    
    return X_pca, params, X_reconstructed, s 

# Generate data
X = genfromtxt('data2D.csv', delimiter=',')
X_mean = np.mean(X, axis=0)    
X_centered = X          # Buggy PCA
X_centered = X - X_mean # Demeaned PCA
object = StandardScaler() 
X_centered=object.fit_transform(X) # Normalized PCA


# Run PCA with 2 dimensions
X_pca, params, X_reconstructed , s_hat = pca(X_centered, 1)

s_log = np.log(s_hat)
#plt.grid(alpha=0.9)
#plt.plot(s_log)

plt.scatter(X_centered[:,0], X_centered[:,1], c="blue")
plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], c="red")



