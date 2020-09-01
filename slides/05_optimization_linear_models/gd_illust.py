import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = np.random.randint(0, 1000)
seed = 188
print(seed)

rng = np.random.RandomState(seed)
X, y = make_regression(n_samples=100, n_features=2, n_informative=2, random_state=rng)

X += 0.01 * rng.randn(100, 2)
X, X_test, y, y_test = train_test_split(X, y, random_state=rng)


n, p = X.shape


def loss(w, train=True):
    if train:
        x = X
        y_ = y
    else:
        x = X_test
        y_ = y_test
    res = np.dot(x, w.T) - y_[:, None]
    return np.mean(res ** 2, axis=0)


def gd(n_iter=10, step=.1):
    w_list = []
    w = np.zeros(p)
    for i in range(n_iter):
        w_list.append(w.copy())
        w -= step * np.dot(X.T, X.dot(w) - y) / n

    return np.array(w_list)


n_it = 25
w_gd = gd(n_it, .5)

w_star = np.linalg.pinv(X).dot(y)

xm, xM = -10, 1.5 * np.abs(w_star[0])
ym, yM = -5, 1.5 * np.abs(w_star[1])
f, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlim(xm, xM)
ax.set_ylim(ym, yM)
ax.set_yticks([])
ax.set_xticks([])
xx, yy = np.meshgrid(np.linspace(xm, xM),
                     np.linspace(ym, yM))
Z = loss(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter([0,], [0,], color='red', s=80,
            label='init')
plt.scatter([w_star[0]], [w_star[1]], color='k', s=80,
            label='solution')
plt.legend()
plt.savefig('images/gd_illust.png', dpi=200)

label = 'gd'
for i in range(1, 9):

    plt.plot(w_gd[:i + 1, 0], w_gd[:i + 1, 1], color='blue', label=label)
    if label is not None:
        label = 'gd'
    label = None
    plt.scatter(w_gd[:i + 1, 0], w_gd[:i + 1, 1], color='blue', s=50, marker='+')

    plt.legend()
    plt.savefig('images/gd_illust_%s.png' % i, dpi=200)



gd_train = loss(w_gd)
gd_test = loss(w_gd, False)
f, ax = plt.subplots(figsize=(4, 2.5))
x_ = plt.xlabel('Number of pass on the dataset')
y_ = plt.ylabel('Error')
l_min = gd_train
plt.plot(gd_train - l_min, color='royalblue', label='gd train', linewidth=2)

plt.yscale('log')
plt.legend(ncol=1, loc='lower left')
plt.savefig('images/gd_loss.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
plt.plot(gd_test - l_min, color='cyan', label='gd test', linewidth=3)

plt.legend(ncol=2)
plt.savefig('images/gd_loss1.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
