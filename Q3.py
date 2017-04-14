import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
from utilities import *


def depth_edges(images):
    # compute max image
    max_image = np.amax(images, axis=0)

    # compute ratio images
    ratio_images = images / (max_image + 1e-12)

    # compute edges
    down = edgedetect1d(ratio_images[0], 'down')
    left = edgedetect1d(ratio_images[1], 'left')
    right = edgedetect1d(ratio_images[2], 'right')
    up = edgedetect1d(ratio_images[3], 'up')

    # combine and normalize
    final = normalize(down + left + right + up)

    # threshold to remove non-edges
    final[final < 0.1] = 0
    final[final >= 0.1] = 1

    # suppress noise
    final = fillholes(final, 2)

    # invert
    final = 1 - final

    return final


""" ENGINE """

# retrieve images
filelist = glob.glob('T3/engine/*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# compute depth edges
final = depth_edges(rgb2gray(images))

# enhance max image with depth edges
max_image = np.amax(images, axis=0)
indices = np.where(final == 0)
max_image[indices] = 0

# compare with ratio images
ratio_images = rgb2gray(images) / (rgb2gray(max_image) + 1e-12)
compare = normalize(np.sum(ratio_images, axis=0))

plt.subplot(121)
plt.imshow(max_image)
plt.subplot(122)
plt.imshow(compare, cmap='gray')
plt.show()


""" FLOWER """

# retrieve images
filelist = glob.glob('T3/flower/*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# compute depth edges
final = depth_edges(rgb2gray(images))

# enhance max image with depth edges
max_image = np.amax(images, axis=0)
indices = np.where(final == 0)
max_image[indices] = 0

# compare with ratio images
ratio_images = rgb2gray(images) / (rgb2gray(max_image) + 1e-12)
compare = normalize(np.sum(ratio_images, axis=0))

plt.subplot(121)
plt.imshow(max_image)
plt.subplot(122)
plt.imshow(compare, cmap='gray')
plt.show()


""" PERSON """

# retrieve images
filelist = glob.glob('T3/person/*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# compute depth edges
final = depth_edges(rgb2gray(images))

# enhance max image with depth edges
max_image = np.amax(images, axis=0)
indices = np.where(final == 0)
max_image[indices] = 0

# compare with ratio images
ratio_images = rgb2gray(images) / (rgb2gray(max_image) + 1e-12)
compare = normalize(np.sum(ratio_images, axis=0))

plt.subplot(121)
plt.imshow(max_image)
plt.subplot(122)
plt.imshow(compare, cmap='gray')
plt.show()
