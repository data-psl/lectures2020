import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_xy(n=1000, sigma=.1):
    X = np.random.rand(n, 2)
    y = np.zeros(n)
    y[X[:, 0] + X[:, 1] < 1] = 1
    frac = 15
    X += sigma * np.random.randn(n, 2)
    # y[:n // frac] = np.random.randn(n // frac) > 0
    return X, y


f, ax = plt.subplots(figsize=(3.5, 3.5))
xm, xM = 0, 1
ax.set_xlim(xm, xM)
ax.set_ylim(xm, xM)
X, y = get_xy()
s = 3
plt.plot(np.linspace(xm, xM), 1 - np.linspace(xm, xM), c='k', label='true limit')
for i, name in enumerate(['class 1', 'class 2']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend()
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('images/trees.png', dpi=200)

depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, depth in enumerate(depths):
    tree = DecisionTreeClassifier(max_depth=depth).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(xm, xM),
                         np.linspace(xm, xM))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ['b', 'orange']
    contour = plt.contourf(xx, yy, Z, levels=1, alpha=0.3, colors=colors)
    plt.savefig('images/trees_%s.png' % (i+1), dpi=200)
    for coll in contour.collections:
        coll.remove()



depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, depth in enumerate(depths):
    tree = RandomForestClassifier(max_depth=depth).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(xm, xM),
                         np.linspace(xm, xM))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ['b', 'orange']
    contour = plt.contourf(xx, yy, Z, levels=1, alpha=0.3, colors=colors)
    plt.savefig('images/trees_%s.png' % (i+1 + 10), dpi=200)
    for coll in contour.collections:
        coll.remove()
