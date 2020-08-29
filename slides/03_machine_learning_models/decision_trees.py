import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


s=20

X, y = load_iris(return_X_y=True)
X = X[:, [2, 3]]

f, ax = plt.subplots(figsize=(4, 2.2))
ax.set_xlim(0, 7)
ax.set_ylim(0, 2.7)

x_ = ax.set_xlabel('Petal length')
y_ = ax.set_ylabel('Petal width')

for i, name in enumerate(['Setosa', 'Versicolor', 'Virginica']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend(loc='upper left')

ax.vlines(2.5, 0, 2.7, color='k')
plt.savefig('images/decision_trees.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


ax.fill([0, 0, 2.5, 2.5], [0, 2.7, 2.7, 0], c='blue', alpha=.3)
plt.savefig('images/decision_trees1.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


ax.hlines(1.75, 2.5, 7, color='k')
plt.savefig('images/decision_trees2.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


ax.fill([2.5, 2.5, 7, 7], [1.75, 2.7, 2.7, 1.75], c='green', alpha=.3)
plt.savefig('images/decision_trees3.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

ax.vlines(4.95, 0, 1.75, color='k')
plt.savefig('images/decision_trees4.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


ax.fill([2.5, 2.5, 4.95, 4.95], [0, 1.75, 1.75, 0], c='orange', alpha=.3)
plt.savefig('images/decision_trees5.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


ax.fill([4.95, 4.95, 7, 7], [0, 1.75, 1.75, 0], c='green', alpha=.3)
plt.savefig('images/decision_trees6.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
