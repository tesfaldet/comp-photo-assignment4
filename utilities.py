import numpy as np
from scipy.ndimage import *
from skimage import feature
import cv2


def normalize(input, a=0, b=1):
    return ((input - np.min(input)) * (b - a)) / \
           (np.max(input) - np.min(input))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style


def gaussianSmooth(image, sigma):
    return gaussian_filter(image, sigma)


def laplacianImages(I, multiple=True):
    weights = [1, -2, 1]
    axis_x = 1
    axis_y = 0
    if multiple:
        axis_x = 2
        axis_y = 1
    fxx = correlate1d(I, weights=weights, axis=axis_x, mode='constant')
    fyy = correlate1d(I, weights=weights, axis=axis_y, mode='constant')
    return np.abs(fxx + fyy)


def edgedetect1d(input, direction):
    if direction == 'down':
        im = np.maximum(convolve1d(input, [1, 0, -1], axis=0), 0)
    elif direction == 'up':
        im = np.maximum(convolve1d(input, [-1, 0, 1], axis=0), 0)
    elif direction == 'left':
        im = np.maximum(convolve1d(input, [-1, 0, 1], axis=1), 0)
    elif direction == 'right':
        im = np.maximum(convolve1d(input, [1, 0, -1], axis=1), 0)

    return im


def fillholes(input, ksize=2):
    structure = np.ones((ksize, ksize))
    return binary_propagation(binary_erosion(input, structure), mask=input)


def dilate(input, ksize=2):
    structure = np.ones((ksize, ksize))
    return binary_dilation(input, structure)


def canny(input, threshold=3):
    return feature.canny(input, threshold)


def bilateralFilter(src, ksize, sigma_range, sigma_spatial):
    channels = np.array_split(src, 3, axis=2)
    return np.stack([cv2.bilateralFilter(c, d=ksize,
                     sigmaColor=sigma_range,
                     sigmaSpace=sigma_spatial) for c in channels], axis=2)


def jointBilateralFilter(joint, src, ksize, sigma_range, sigma_spatial):
    c = np.array_split(src, 3, axis=2)
    j = np.array_split(joint, 3, axis=2)
    return np.stack([cv2.ximgproc.jointBilateralFilter(j[i], c[i], d=ksize,
                     sigmaColor=sigma_range,
                     sigmaSpace=sigma_spatial) for i in range(len(c))],
                    axis=2)


def guidedFilter(guide, src, radius, eps):
    return cv2.ximgproc.guidedFilter(guide, src, radius, eps)
