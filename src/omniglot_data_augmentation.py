# -*- coding: utf-8 -*-
"""
In this script we use rotations, shears, zooms and shits to augment the Omniglot
dataset that will be use to train the Siamese Neural Network model.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 04/2024
"""


import os
import random
import shutil

import cv2
import numpy as np
import scipy
from scipy import ndimage
from skimage import transform as tf


def padded_zoom(img, zoomfactor=0.8):
    """
    Returns a zoomed image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to be zoomed.
    zoomfactor : float, optional
        Zoom factor for the image (default is 0.8).

    Returns
    -------
    numpy.ndarray
        Zoomed image.
    """

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoomfactor)
    return cv2.warpAffine(img, M, img.shape[::-1], borderValue=255)


def image_aug(image, rotation_range, shear_range, zoom_range, shift_range):
    """
    Returns an image with random rotation, shear, zoom, and/or shift applied.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to be augmented.
    rotation_range : tuple
        Range of rotation angles in degrees, e.g., (min_angle, max_angle).
    shear_range : tuple
        Range of shear angles in degrees, e.g., (min_angle, max_angle).
    zoom_range : tuple
        Range of zoom factors, e.g., (min_factor, max_factor).
    shift_range : tuple
        Range of shift values, e.g., (min_shift, max_shift).

    Returns
    -------
    numpy.ndarray
        Augmented image.
    """

    image2 = image

    if random.random() > 0.5:  # rotation
        angle = random.uniform(rotation_range[0], rotation_range[1])
        # print("rotation", angle)
        image2 = ndimage.rotate(image2, angle, cval=255, reshape=False)

    if random.random() > 0.5:  # shear
        shear_val = random.uniform(shear_range[0], shear_range[1])
        # print("shear", shear_val)
        afine_tf = tf.AffineTransform(shear=shear_val)
        image2 = tf.warp(image2, inverse_map=afine_tf, cval=1)
        image2 = image2 * 255

    if random.random() > 0.5:  # zoom
        zoom_val = random.uniform(zoom_range[0], zoom_range[1])
        # print("zoom", zoom_val)
        image2 = padded_zoom(image2, zoom_val)

    if random.random() > 0.5:  # shift
        shift_val = random.uniform(shift_range[0], shift_range[1])
        # print("shift", shift_val)
        image2 = scipy.ndimage.shift(image2, shift_val, cval=255)
    return image2


def to_black_and_white(image):
    """
    Converts the input image to a binary black and white image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to be converted.

    Returns
    -------
    numpy.ndarray
        Binary black and white image.
    """

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > 127.5:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


def main():
    """
    This function sets the current working directory to the project folder, defines
    ranges for image transformations, applies transformations to the Omniglot invented
    dataset and saves the augmented images.
    """

    # We set the current working directory to the project folder.
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Ranges of the transformations.
    rotation_range = [-10, 10]
    shear_range = [-0.3, 0.3]
    zoom_range = [0.8, 1.2]
    shift_range = [-2, 2]

    # We apply 8 transformation per glyph to the omniglot invented dataset.
    source_dir = "data/raw/omniglot_invented/images_background"
    destination_dir = "data/processed/omniglot_invented_augmented/images_background"
    shutil.copytree(source_dir, destination_dir)
    for i in os.listdir(destination_dir):
        print(i)
        for j in os.listdir(destination_dir + "/" + i):
            list_char = os.listdir(destination_dir + "/" + i + "/" + j)
            for k in list_char:
                im = cv2.imread(
                    destination_dir + "/" + i + "/" + j + "/" + k, cv2.IMREAD_GRAYSCALE
                )
                im = np.array(im)

                for l in range(8):
                    im2 = image_aug(
                        im, rotation_range, shear_range, zoom_range, shift_range
                    )
                    im2 = to_black_and_white(im2)
                    im2_nom = k[:-4] + "_" + str(l) + ".png"

                    cv2.imwrite(
                        destination_dir + "/" + i + "/" + j + "/" + im2_nom, im2
                    )


if __name__ == "__main__":
    main()
