import numpy as np

import matplotlib.pyplot as plt

fontsize = 18
params = {
      'axes.titlesize': fontsize + 4,
      'axes.labelsize': fontsize + 2,
      'font.size': fontsize + 2,
      'legend.fontsize': fontsize + 2,
      'xtick.labelsize': fontsize,
      'ytick.labelsize': fontsize,
      'text.usetex': True,
      'text.latex.preamble': r'\usepackage{{amsmath}}'}
plt.rcParams.update(params)


def plot(C, i, plot_array=False, box=True):
    f, ax = plt.subplots(figsize=(3, 3))
    n= 200
    x = np.random.randn(n, 2).dot(C.T)
    plt.scatter(x[:, 0], x[:, 1], s=10)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    s = r'$C = \begin{pmatrix} %d & %d \\ %d & %d \end{pmatrix}$' % (C[0, 0], C[1, 0], C[0, 1], C[1, 1])
    # print(s)
    ax.set_yticks([])
    ax.set_xticks([])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.savefig('images/correlation_%d0.png' % i, dpi=200)
    if box:
        plt.text(-2.5, -1, s, bbox=props)
        plt.savefig('images/correlation_%d.png' % i, dpi=200)
    if plot_array:
        u, w = np.linalg.eigh(C)
        plt.arrow(0, 0, 3 * w[0, 1], 3 *  w[1, 1], width=0.1, color='k')
        plt.savefig('images/correlation_%d_array.png' % i, dpi=200)


def plotarr(C):
    f, ax = plt.subplots(1, 2, figsize=(6, 3))
    n= 200
    x = np.random.randn(n, 2).dot(C.T)
    ax[0].scatter(x[:, 0], x[:, 1], s=10)
    ax[0].set_xlim([-5, 5])
    ax[0].set_ylim([-5, 5])
    # s = r'$C = \begin{pmatrix} %d & %d \\ %d & %d \end{pmatrix}$' % (C[0, 0], C[1, 0], C[0, 1], C[1, 1])
    # print(s)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    t = np.linspace(0, np.pi, 100)
    c_list = []
    for t_ in t:
        c, s = np.cos(t_), np.sin(t_)
        z = c * x[:, 0] + s * x[:, 1]
        c_list.append(np.mean(z ** 2))
    ax[1].plot(t, c_list)
    ax[1].set_xlabel('Angle')
    ax[1].set_ylabel('Power')
    t = np.linspace(0, np.pi, 20)
    f.tight_layout()
    for i, t_ in enumerate(t):
        c, s = np.cos(t_), np.sin(t_)
        ar = ax[0].arrow(0, 0, 3 * c, 3 * s, width=0.1, color='k')
        line = ax[1].vlines(t_, 0, 1.1 * np.max(c_list))

        plt.savefig('images/correlation_pow%d.png' % i, dpi=200)
        line.remove()
        ar.remove()


C_list = [
[[1, 0], [0, 1]],
[[2, 0], [0, 1]],
[[2, 1], [1, 1]]
]

for i, C in enumerate(C_list):
    plot(np.array(C), i, plot_array=False)

plotarr(np.array(C))
