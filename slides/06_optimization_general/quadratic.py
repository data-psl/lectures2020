import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


fontsize = 18
params = {
      'axes.titlesize': fontsize + 4,
      'axes.labelsize': fontsize + 2,
      'font.size': fontsize + 2,
      'legend.fontsize': fontsize + 2,
      'xtick.labelsize': fontsize,
      'ytick.labelsize': fontsize,
      'text.usetex': True}
plt.rcParams.update(params)

A =  np.array([[3, -1], [-1, 1]])

x_star = np.array([1, 1])

b = - np.dot(A, x_star)


def loss(w, train=True):
    return .5 * np.sum(w * w.dot(A), axis=1) + w.dot(b)


xm, xM = -1, 3
ym, yM = -1, 3
f, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(xm, xM)
ax.set_ylim(ym, yM)
ax.set_yticks([])
ax.set_xticks([])
plt.xlabel(r"$w'_1$")
plt.ylabel(r"$w'_2$")
xx, yy = np.meshgrid(np.linspace(xm, xM),
                     np.linspace(ym, yM))
Z = loss(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter([0, ], [0, ], color='k', s=80,
            label=r'$w$')
plt.legend()
plt.savefig('images/quadratic.png', dpi=200)
plt.scatter([1, ], [1, ], color='red', s=120, marker='x',
            label=r'$\tilde{w}$')
plt.legend()
plt.savefig('images/quadratic1.png', dpi=200)



f, ax = plt.subplots(figsize=(6, 2))

t = np.arange(1, 15)

v1 = np.exp(-t / 2)
v2 = np.exp(-t ** 2 / 10)

plt.plot(t, v1, label='gradient descent')
plt.plot(t, v2, label="Newton's method")
plt.legend()
x_ = plt.xlabel('iterations')
y_ = plt.ylabel('error')
plt.yscale('log')
plt.savefig('images/newton_cv.png', dpi=200, bbox_extra_artists=[x_, y_],
            bbox_inches='tight')
