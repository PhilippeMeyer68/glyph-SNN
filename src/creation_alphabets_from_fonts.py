# -*- coding: utf-8 -*-
"""
This script permits to create alphabets from ttf font files and to export them as numpy
arrays.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 04/2024
"""


import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import dictionary_alphabets


def plot_an_alphabet(dict_alphabets, lang, xsize, ysize, yoffset, fontsize):
    """
    Plot the alphabet of a specified language.

    Parameters
    ----------
    dict_alphabets : dict
        A dictionary containing alphabets for different languages.
    lang : str
        The name of the alphabet to plot.
    xsize : int
        The width of the plotted image.
    ysize : int
        The height of the plotted image.
    yoffset : int
        The vertical offset for the position of the letter within the image.
    fontsize : int
        The font size of the plotted letter.
    """

    range_alph = dict_alphabets[lang]
    font = "data/raw/fonts/" + range_alph[1] + ".ttf"

    for i in range_alph[0]:
        letter = chr(i)
        image = Image.new("RGB", (xsize, ysize), (256, 256, 256))
        draw = ImageDraw.Draw(image)
        xoffset = (
            xsize - draw.textlength(letter, font=ImageFont.truetype(font, fontsize))
        ) / 2
        draw.text(
            (xoffset, yoffset),
            letter,
            font=ImageFont.truetype(font, fontsize),
            fill=(0, 0, 0),
        )
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.imshow(image)
        plt.show()


def create_alphabet(
    dict_alphabets, lang, xsize, ysize, yoffset, fontsize, image_unsupported
):
    """
    Create the alphabet of a specified language and exports it as a numpy array.

    Parameters
    ----------
    dict_alphabets : dict
        A dictionary containing alphabets for different languages.
    lang : str
        The language of the alphabet to create.
    xsize : int
        The width of the letters in the alphabet.
    ysize : int
        The height of the letters in the alphabet.
    yoffset : int
        The vertical offset for the position of the letter within each image.
    fontsize : int
        The font size of the letters in the alphabet.
    image_unsupported : numpy.ndarray
        A reference image used to check if a generated glyph is unsupported.

    Returns
    -------
    numpy.ndarray
        An array containing the alphabet letters represented as numpy arrays.
    """

    range_alph = dict_alphabets[lang]
    font = "data/raw/fonts/" + range_alph[1] + ".ttf"

    X_glyphs = []
    for i in range_alph[0]:
        letter = chr(i)
        image = Image.new("RGB", (xsize, ysize), (256, 256, 256))
        draw = ImageDraw.Draw(image)
        xoffset = (
            xsize - draw.textlength(letter, font=ImageFont.truetype(font, fontsize))
        ) / 2
        draw.text(
            (xoffset, yoffset),
            letter,
            font=ImageFont.truetype(font, fontsize),
            fill=(0, 0, 0),
        )
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        difference = cv2.subtract(image, image_unsupported)
        if not np.any(difference) == False:
            X_glyphs.append(image)
        else:
            pass
            print("Problem with a glyph.")
    return np.array(X_glyphs)


def main():
    """
    This function sets the current working directory, imports the dictionary of
    alphabets and unicode ranges, performs some statistics, tests loading an alphabet,
    creates a tofu image to test glyph loading and exports all the alphabets as numpy
    arrays.
    """

    # We set the current working directory to the project folder.
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # We import the dictionary of alphabets and unicodes ranges.
    dict_alphabets = dictionary_alphabets.dict_alphabets

    # Parameters of alphabets that will be exported.
    xsize, ysize = 105, 105  # Size of the images.
    yoffset = 10  # Offset to centralize characters in the images.
    fontsize = 51  # Size of the glyphs in the images.

    # We give some statistics and we test an alphabet.
    print(
        "There are",
        len(dict_alphabets),
        "alphabets in our database and",
        sum([len(dict_alphabets[lang][0]) for lang in dict_alphabets.keys()]),
        "glyphs.",
    )

    for lang in dict_alphabets:
        print(lang, ":", len(dict_alphabets[lang][0]))
    lang = "Meroitic Hieroglyphs"
    print("We test the load of the alphabet", lang)
    plot_an_alphabet(dict_alphabets, lang, xsize, ysize, yoffset, fontsize)

    # Creation of the tofu image "unsupported" to test if the glyphs are well loaded
    letter = chr(1)
    image_unsupported = Image.new("RGB", (xsize, ysize), (256, 256, 256))
    draw = ImageDraw.Draw(image_unsupported)
    font_arial = "data/raw/fonts/ArialUnicode.ttf"
    xoffset = (
        105 - draw.textlength(letter, font=ImageFont.truetype(font_arial, fontsize))
    ) / 2
    draw.text(
        (xoffset, yoffset),
        letter,
        font=ImageFont.truetype(font_arial, fontsize),
        fill=(0, 0, 0),
    )
    image_unsupported = np.array(image_unsupported)
    image_unsupported = cv2.cvtColor(image_unsupported, cv2.COLOR_BGR2GRAY)

    # We export all the alphabets as numpy arrays.
    for lang in dict_alphabets:
        X_glyphs = create_alphabet(
            dict_alphabets, lang, xsize, ysize, yoffset, fontsize, image_unsupported
        )
        np.save("data/processed/alphabets/X_" + lang + ".npy", X_glyphs)


if __name__ == "__main__":
    main()
