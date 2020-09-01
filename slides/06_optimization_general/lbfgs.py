import numpy as np
from numba import njit
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt



def loss_logreg(x, A, b, lbda):
    bAx = b * np.dot(A, x)
    return np.mean(np.log1p(np.exp(- bAx))) + lbda * np.dot(x, x) / 2.


@njit
def grad_i_logreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    b_i = b[i]
    return - a_i * b_i / (1. + np.exp(b_i * np.dot(a_i, x))) + lbda * x


@njit
def grad_logreg(x, A, b, lbda):
    """Full gradient"""
    g = np.zeros_like(x)
    for i in range(n):
        g += grad_i_logreg(i, x, A, b, lbda)
    return g / n


def gd(n_iter):
    l_cst = np.linalg.norm(A, ord=2) ** 2 / (4. * n) + lbda
    step = 2 / l_cst
    l_list = []
    x = np.zeros(p)
    for i in range(n_iter):
        x -= step * grad_logreg(x, A, b, lbda)
        l_list.append(loss_logreg(x, A, b, lbda))
    return l_list


class cb(object):
    def __init__(self):
        self.l_list = []

    def __call__(self, x):
        self.l_list.append(loss_logreg(x, A, b, lbda))


n, p = 1000, 5000
A = np.random.randn(n, p)
b = np.random.randn(n)
lbda = .1
x = np.zeros(p)

factr=10
f, ax = plt.subplots(figsize=(4, 2))
c = cb()
_ = fmin_l_bfgs_b(loss_logreg, np.zeros(p), fprime=grad_logreg, callback=c, args=(A, b, lbda),
factr=factr)

l1 = np.array(c.l_list)
l2 = np.array(gd(len(l1)))

print(l1)
print(l2)
l_m = min((np.min(l1), np.min(l2)))
plt.plot(l1 - l_m, label='L-BFGS')
plt.plot(l2 - l_m, label='Gradient descent')
x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('images/lbfgs.png', dpi=200, bbox_inches='tight',
            bbox_extra_artists=[x_, y_])
