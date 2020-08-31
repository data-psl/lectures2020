import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential
import matplotlib.pyplot as plt

n_hidden = 4
nn1 = Sequential(nn.Linear(2, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 2))
nn2 = Sequential(nn.Linear(2, n_hidden), nn.Tanh(),
                  nn.Linear(n_hidden, n_hidden), nn.Tanh(),
                  nn.Linear(n_hidden, 2))


n = 1000
n_points = 10
t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
c = np.array([(np.cos(t_), np.sin(t_)) for t_ in t])
y = np.arange(n_points) % 2

X = np.concatenate([0.1 * np.random.randn(n, 2) + c_ for c_ in c])
y = np.concatenate([y_ * np.ones(n) for y_ in y])
X = torch.tensor(X).float()
y = torch.tensor(y).long()

f, ax = plt.subplots(figsize=(3.5, 3.5))
xm, xM = -1.5, 1.5
ax.set_xlim(xm, xM)
ax.set_ylim(xm, xM)
s = 3
for i, name in enumerate(['class 1', 'class 2']):
    loc = np.where(y == i)[0]
    plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
plt.legend()
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('images/nn.png', dpi=200)

for n_hidden in [2, 3, 4, 5, 6]:
    nn1 = Sequential(nn.Linear(2, n_hidden), nn.Tanh(),
                     nn.Linear(n_hidden, 2 * n_hidden), nn.Tanh(),
                     nn.Linear(2 * n_hidden, 2))
    optimizer = optim.Adam(nn1.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for i in range(1001):
        optimizer.zero_grad()
        pred = nn1(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())


    f, ax = plt.subplots(figsize=(3.5, 3.5))
    xm, xM = -1.5, 1.5
    ax.set_xlim(xm, xM)
    ax.set_ylim(xm, xM)
    s = 3
    for i, name in enumerate(['class 1', 'class 2']):
        loc = np.where(y == i)[0]
        plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
    plt.legend()
    ax.set_xticks([])
    ax.set_yticks([])



    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5),
                         np.linspace(-1.5, 1.5))
    data = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    op = nn1(data).detach()
    z = op.numpy().argmax(axis=1)
    Z = z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=1, alpha=0.5, colors=['b', 'orange'])
    plt.savefig('images/nn_two_%s.png' % n_hidden, dpi=200)


for n_hidden in [2, 3, 4, 5, 6]:
    nn1 = Sequential(nn.Linear(2, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 2))
    optimizer = optim.Adam(nn1.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for i in range(1001):
        optimizer.zero_grad()
        pred = nn1(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())


    f, ax = plt.subplots(figsize=(3.5, 3.5))
    xm, xM = -1.5, 1.5
    ax.set_xlim(xm, xM)
    ax.set_ylim(xm, xM)
    s = 3
    for i, name in enumerate(['class 1', 'class 2']):
        loc = np.where(y == i)[0]
        plt.scatter(X[loc, 0], X[loc, 1], s=s, label=name)
    plt.legend()
    ax.set_xticks([])
    ax.set_yticks([])



    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5),
                         np.linspace(-1.5, 1.5))
    data = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    op = nn1(data).detach()
    z = op.numpy().argmax(axis=1)
    Z = z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=1, alpha=0.5, colors=['b', 'orange'])
    plt.savefig('images/nn_one_%s.png' % n_hidden, dpi=200)
