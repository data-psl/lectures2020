import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm

s=20
X, y = load_iris(return_X_y=True)
X = X[:, [2, 3]]

f, ax = plt.subplots(figsize=(4, 2.2))
ax.set_xlim(0, 7)
ax.set_ylim(0, 2.7)

x_ = ax.set_xlabel('Petal length')
y_ = ax.set_ylabel('Petal width')

plt.savefig('images/iris_1.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([X[0, 0]], [X[0, 1]], c='k', s=s)
plt.savefig('images/iris_2.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([X[51, 0]], [X[51, 1]], c='k', s=s)
plt.savefig('images/iris_3.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


plt.scatter(X[:, 0], X[:, 1], c='k', s=s)
plt.savefig('images/iris_4.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

for i, name in enumerate(['Setosa', 'Versicolor', 'Virginica']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend()
plt.savefig('images/iris_5.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


rf = RandomForestClassifier().fit(X, y)

xc = [1, .5]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_6.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([xc[0]], [xc[1]], c='blue', marker='x', s=4*s)
plt.savefig('images/iris_7.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


xc = [4, 1.2]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_8.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([xc[0]], [xc[1]], c='orange', marker='x', s=4*s)
plt.savefig('images/iris_9.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)



xc = [5, 2.2]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_10.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([xc[0]], [xc[1]], c='green', marker='x', s=4*s)
plt.savefig('images/iris_11.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


xc = [2.5, .8]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_12.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


xc = [4.9, 1.6]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_13.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


xc = [6, .2]
x = np.array([[xc[0], xc[1]]])


plt.scatter([xc[0]], [xc[1]], c='k', marker='x', s=4*s)
plt.savefig('images/iris_14.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


rf = RandomForestClassifier().fit(X, y)
xx, yy = np.meshgrid(np.linspace(0, 7),
                     np.linspace(0, 2.7))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
colors = ['b', 'orange', 'green']
plt.contourf(xx, yy, Z, levels=2, alpha=0.3, colors=colors)
plt.savefig('images/iris_15.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
