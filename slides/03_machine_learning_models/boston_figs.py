import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from matplotlib import cm


s = 10
X, y = load_boston(return_X_y=True)
X = X[:, [0, 5]]

# print(np.argsort(X[:, 0]))
f, ax = plt.subplots(figsize=(4, 2.2))
ax.set_xlim(0.01, 90)
ax.set_ylim(4, 51)
ax.set_xscale('log')
x_ = ax.set_xlabel('Crime rate')
y_ = ax.set_ylabel('Target: median price')

plt.savefig('images/boston_1.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

plt.scatter([X[358, 0]], [y[358]], c='k', s=s, marker='x')
plt.savefig('images/boston_2.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


plt.scatter(X[:, 0], y, c='k', s=s, marker='x')
plt.savefig('images/boston_3.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


rf = RandomForestRegressor(max_depth=3).fit(X[:, 0].reshape(-1, 1), y)
xx = np.logspace(-2, 2)
plt.plot(xx, rf.predict(xx.reshape(-1, 1)), linewidth=3, c='red',
         label='prediction')
plt.legend()
plt.savefig('images/boston_31.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
plt.close('all')

f, ax = plt.subplots(figsize=(4, 2.2))
ax.set_xlim(.01, 90)
ax.set_ylim(3, 9)
ax.set_xscale('log')
x_ = ax.set_xlabel('Crime rate')
y_ = ax.set_ylabel('Average number of rooms')
sc = plt.scatter(X[:, 0], X[:, 1], c=y, s=s, marker='x')
c_ = plt.colorbar(sc)
plt.savefig('images/boston_4.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


rf = RandomForestRegressor().fit(X, y)
# rf = KernelRidge().fit(X, y)

xx, yy = np.meshgrid(np.linspace(0, 90),
                     np.linspace(3, 9))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=2, alpha=0.5)
plt.savefig('images/boston_5.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)
