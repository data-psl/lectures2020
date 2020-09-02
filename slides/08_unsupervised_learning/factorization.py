import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA


X = np.random.laplace(size=(2000, 2))

X = np.sign(X) * np.abs(X) ** 2
X = X.dot(np.random.randn(2, 2))

s=3
f = plt.figure(figsize=(4.5, 2.5))
plt.scatter(X[:, 0], X[:, 1], s=s)
plt.savefig('images/data.png', dpi=200)

X_pca = PCA(2).fit_transform(X.copy())
f = plt.figure(figsize=(4.5, 2.5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=s)
plt.savefig('images/pca_data.png', dpi=200)


X_ica = FastICA(2).fit_transform(X.copy())
f = plt.figure(figsize=(4.5, 2.5))
plt.scatter(X_ica[:, 0], X_ica[:, 1], s=s)
plt.savefig('images/ica_data.png', dpi=200)
