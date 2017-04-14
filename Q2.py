import numpy as np
import matplotlib
matplotlib.use('tkagg')  # for rendering to work
import matplotlib.pyplot as plt
from scipy import misc
import glob
from utilities import *


def merge_flash_no_flash(flash, no_flash, s_r, s_s, s_r_joint, s_s_joint,
                         shadow_thresh, specular_thresh, mask_smooth):
    # bilateral filter on no-flash image (A^Base)
    A_base = bilateralFilter(no_flash, -1, sigma_range=s_r, sigma_spatial=s_s)

    # joint bilateral filter between no-flash and flash images (A^NR)
    A_NR = jointBilateralFilter(flash, no_flash, -1, sigma_range=s_r_joint,
                                sigma_spatial=s_s_joint)

    # flash image divided by bilateral filter on flash image (F^Detail)
    F_base = bilateralFilter(flash, -1, sigma_range=s_r, sigma_spatial=s_s)
    F_detail = (flash + 0.02) / (F_base + 0.02)

    # shadow and specular mask (M)
    shadow_mask = np.zeros((flash.shape[0], flash.shape[1]))
    indices = np.where(rgb2gray(flash - no_flash) < shadow_thresh)
    shadow_mask[indices] = 1
    shadow_mask = fillholes(shadow_mask)

    specular_mask = np.zeros((flash.shape[0], flash.shape[1]))
    indices = np.where(rgb2gray(flash) > specular_thresh)
    specular_mask[indices] = 1
    specular_mask = fillholes(specular_mask)

    # merge the two masks
    mask = np.logical_xor(specular_mask, shadow_mask)
    mask = mask.astype('float32')

    # dilate mask a bit
    mask = dilate(mask, 5).astype('float32')

    # expand dims
    mask = np.expand_dims(mask, 4)

    # feather the mask
    mask = gaussianSmooth(mask, mask_smooth)

    # final merged output (A^Final)
    A_final = ((1 - mask) * A_NR * F_detail) + (mask * A_base)

    return A_base, A_NR, F_detail, mask, A_final


""" IMAGE 1 """
# retrieve images
filelist = glob.glob('T2/image1*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# normalize inputs
flash = normalize(images[0].astype('float32'))
no_flash = normalize(images[1].astype('float32'))

A_base, A_NR, F_detail, mask, A_final = \
    merge_flash_no_flash(flash, no_flash, 0.1, 3, 0.005, 2, 0.0, 0.8, 2)

# save images
misc.imsave('T2/Image1Base.bmp', A_base)
misc.imsave('T2/Image1NR.bmp', A_NR)
misc.imsave('T2/Image1Fdetail.bmp', F_detail)
misc.imsave('T2/Image1Mask.bmp', np.squeeze(mask))
misc.imsave('T2/Image1Output.bmp', A_final)

""" IMAGE 2 """
# retrieve images
filelist = glob.glob('T2/image2*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# normalize inputs
flash = normalize(images[0].astype('float32'))
no_flash = normalize(images[1].astype('float32'))

A_base, A_NR, F_detail, mask, A_final = \
    merge_flash_no_flash(flash, no_flash, 0.1, 5, 0.005, 5, -0.1, 0.8, 1)

# save images
misc.imsave('T2/Image2Base.bmp', A_base)
misc.imsave('T2/Image2NR.bmp', A_NR)
misc.imsave('T2/Image2Fdetail.bmp', F_detail)
misc.imsave('T2/Image2Mask.bmp', np.squeeze(mask))
misc.imsave('T2/Image2Output.bmp', A_final)

""" IMAGE 3 """
# retrieve images
filelist = glob.glob('T2/image3*.bmp')

# stack them into SxHxW
images = np.stack([misc.imread(fname) for fname in filelist])

# normalize inputs
flash = normalize(images[0].astype('float32'))
no_flash = normalize(images[1].astype('float32'))

A_base, A_NR, F_detail, mask, A_final = \
    merge_flash_no_flash(flash, no_flash, 0.8, 5, 0.05, 5, -0.2, 0.8, 1)

# save images
misc.imsave('T2/Image3Base.bmp', A_base)
misc.imsave('T2/Image3NR.bmp', A_NR)
misc.imsave('T2/Image3Fdetail.bmp', F_detail)
misc.imsave('T2/Image3Mask.bmp', np.squeeze(mask))
misc.imsave('T2/Image3Output.bmp', A_final)
