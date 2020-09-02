import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA
import mne

X = mne.io.read_raw_edf('ecgca444.edf')['data'][0].T
n_ch = X.shape[1]
X = X[40000:50000]
sf = 1000
X -= X.mean(axis=0)
X /= X.std(axis=0)


def plot(X_, savename, lim=None):
    X = X_.copy()
    if lim is not None:
        X = X[lim]
    f, ax = plt.subplots(figsize=(6, 4))
    n_s, n_f = X.shape
    time = np.linspace(0, n_s / sf, n_s)
    offset = 0
    ax.set_yticks([])
    x_ = ax.set_xlabel('time (sec.)')
    for x in X.T:
        plt.plot(time, x + offset - x.min())
        offset += 1.1 * (x.max() - x.min())
    plt.savefig(savename, bbox_extra_artists=[x_, ], bbox_inches='tight')


lim = None
plot(X, 'images/raw.png', lim)
plot(PCA(n_ch, whiten=True).fit_transform(X.copy()), 'images/raw_pca.png', lim)
plot(FastICA(n_ch).fit_transform(X.copy()), 'images/raw_ica.png', lim)
