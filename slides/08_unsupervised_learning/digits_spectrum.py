import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt



X, _ = load_digits(return_X_y=True)
f, ax = plt.subplots(figsize=(4.5, 3))
X -= np.mean(X, axis=0)
X /= np.std(X)
# X /= np.std(X, axis=0)
e = np.linalg.eigvalsh(np.dot(X.T, X))
plt.semilogy(e[::-1])
plt.ylim([1e-2, 1e5])
x_ = plt.xlabel('Component #')
y_ = plt.ylabel('Power')
plt.savefig('images/spectrum.png', dpi=200,
            bbox_extra_artists=[x_, y_], bbox_inches='tight')


f, ax = plt.subplots(2, 3, figsize=(6, 4))
X, _ = load_digits(return_X_y=True)
for i, axe in enumerate(ax.ravel()):
    axe.imshow(X[i + 10].reshape(8, 8))
    axe.set_xticks([])
    axe.set_yticks([])
plt.savefig('images/digits.png', dpi=200)
