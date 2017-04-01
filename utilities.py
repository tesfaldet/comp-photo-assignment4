import numpy as np
from scipy.ndimage import *


def normalize(input, a=0, b=1):
    return ((input - np.min(input)) * (b - a)) / \
           (np.max(input) - np.min(input))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # MATLAB style
