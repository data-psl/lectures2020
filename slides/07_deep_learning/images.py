import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

from sklearn.feature_extraction import image
from sklearn.utils import check_random_state

fig, axes = plt.subplots(10, 10)
axes = axes.ravel()

data = Image.open('DSC_0360.jpg')
w, h = data.size
data = data.resize((600, 400))
data = np.array(data)

patches = image.extract_patches_2d(data, (5, 5))

rng = check_random_state(10)
for i, ax in enumerate(axes):
    i = rng.randint(len(patches))
    ax.imshow(patches[i].reshape((5, 5, 3)))
    ax.axis('off')
plt.savefig('patches.png')

fig, ax = plt.subplots(1, 1)
ax.imshow(data)
ax.axis('off')
plt.savefig('image.png')

