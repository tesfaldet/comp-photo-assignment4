import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
from utilities import *


# retrieve images
filelist = glob.glob('T3/engine/*.bmp')
