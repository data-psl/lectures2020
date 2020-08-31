import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from matplotlib import cm

s=20
X, y = load_iris(return_X_y=True)
X = X[:, [2, 3]][50:]
y = y[50:] - 1

f, ax = plt.subplots(figsize=(4, 2.2))
ax.set_xlim(2.5, 7)
ax.set_ylim(0.7, 2.7)

x_ = ax.set_xlabel('Petal length')
y_ = ax.set_ylabel('Petal width')


for i, name in enumerate(['Versicolor', 'Virginica']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend(loc='upper left')
plt.savefig('images/linear_model1.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


lr = LogisticRegression().fit(X, y)

c = lr.coef_[0]
b = lr.intercept_

print(c, b)
x = np.linspace(2.5, 7)
pred = (- c[0] * x - b) / c[1]

plt.plot(x, pred, c='k', label='limit', linewidth=3)
plt.legend(loc='upper left')
plt.savefig('images/linear_model2.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)

xx, yy = np.meshgrid(np.linspace(2.5, 7),
                     np.linspace(0.7, 2.7))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
colors = ['b', 'orange']
plt.contourf(xx, yy, Z, levels=1, alpha=0.3, colors=colors)
plt.savefig('images/linear_model3.png', bbox_extra_artists=[x_, y_],
            bbox_inches='tight', dpi=200)


f, ax = plt.subplots(figsize=(3.5, 3.5))
n = 200
c = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])
y = [0, 1, 0, 1]

X = np.concatenate([0.1 * np.random.randn(n, 2) + c_ for c_ in c])
y = np.concatenate([y_ * np.ones(n) for y_ in y])
xm, xM = -.5, 1.5
ax.set_xlim(xm, xM)
ax.set_ylim(xm, xM)
s = 3
for i, name in enumerate(['class 1', 'class 2']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend()
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('images/linear_model4.png', dpi=200)

plt.close('all')
# f, ax = plt.subplots(figsize=(4, 2.2))
#
# scores = []
# train = []
# n_f = 30
# n_features_list = np.arange(1, n_f)
# n_repeat = 100
# for n_features in n_features_list:
#     print(n_features)
#     sc = []
#     tr = []
#     for i in range(n_repeat):
#         X, y = make_regression(n_samples=10, n_features=n_f, n_informative=2)
#         X_train, X_test, y_train, y_test = train_test_split(X, y)
#         lr = LinearRegression().fit(X_train[:, :n_features], y_train)
#         p = lr.predict(X_train[:, :n_features])
#         pred = lr.predict(X_test[:, :n_features])
#         score = np.sqrt(np.mean((y_test - pred) ** 2))
#         sc.append(score)
#         tr.append(np.sqrt(np.mean((p - y_train) ** 2)))
#     scores.append(np.mean(sc))
#     train.append(np.mean(tr))
#
# plt.plot(n_features_list, scores)
# plt.plot(n_features_list, train)
# plt.show()
