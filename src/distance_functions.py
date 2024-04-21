# -*- coding: utf-8 -*-
"""
In this script we define the siamese-based distance between writing systems.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 04/2024
"""


import numpy as np


def closest_glyph(i, X_glyph, dict_dist_loc, invert):
    """
    Finds the glyph in X_glyph closest to the glyph at index i.

    Parameters
    ----------
    i : int
        Index of the glyph in X_glyph to find the closest glyph for.
    X_glyph : numpy.ndarray
        Array containing glyphs.
    dict_dist_loc : dict
        Dictionary containing distances between glyphs.
    invert : bool
        Flag indicating whether to invert the distance lookup.

    Returns
    -------
    float
        The distance between the glyph i and its closest glyph in X_glyph.
    int
        The index of the closest glyph in X_glyph.
    """

    if invert:
        l = [dict_dist_loc[(j, i)] for j in range(len(X_glyph))]
    else:
        l = [dict_dist_loc[(i, j)] for j in range(len(X_glyph))]
    return (min(l), np.argmin(l))


def similarity_between_scripts_tilde(X_glyph_1, X_glyph_2, dict_dist_loc, invert):
    """
    Computes the similarity of X_glyph_1 to X_glyph_2, corresponding to
    d^tilde(X_glyph_1, X_glyph_2) in the paper.

    Parameters
    ----------
    X_glyph_1 : numpy.ndarray
        Array containing glyphs for the first set.
    X_glyph_2 : numpy.ndarray
        Array containing glyphs for the second set.
    dict_dist_loc : dict
        Dictionary containing distances between glyphs.
    invert : bool
        Flag indicating whether to invert the distance lookup.

    Returns
    -------
    float
        The mean similarity between of the glyphs X_glyph_1 to the glyphs of X_glyph_2.
    """

    l = []
    for i in range(len(X_glyph_1)):
        l.append(closest_glyph(i, X_glyph_2, dict_dist_loc, invert)[0])
    return np.mean(l)


def similarity_between_scripts(X_glyph_1, X_glyph_2, dict_dist_loc):
    """
    Computes the siamese-based distance between two sets of glyphs, corresponding to
    d(X_glyph_1, X_glyph_2) in the paper.

    Parameters
    ----------
    X_glyph_1 : numpy.ndarray
        Array containing glyphs for the first set.
    X_glyph_2 : numpy.ndarray
        Array containing glyphs for the second set.
    dict_dist_loc : dict
        Dictionary containing distances between glyphs.

    Returns
    -------
    float
        The siamese-based distance between the two sets of glyphs.
    """

    sim_1 = similarity_between_scripts_tilde(
        X_glyph_1, X_glyph_2, dict_dist_loc, invert=False
    )
    sim_2 = similarity_between_scripts_tilde(
        X_glyph_2, X_glyph_1, dict_dist_loc, invert=True
    )
    sim = (sim_1 + sim_2) / 2
    return sim
