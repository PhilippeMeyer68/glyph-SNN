# -*- coding: utf-8 -*-
"""
In this script we define the siamese-based distance between writing systems.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 03/2024
"""

# packages
import numpy as np


def closest_glyph_1(i, X_glyph, dict_dist_loc):
    """Exports the glyph of X_glyph_2 closest to the glyph i of X_glyph_1."""
    l = [dict_dist_loc[(i, j)] for j in range(len(X_glyph))]
    return (min(l), np.argmin(l))


def closest_glyph_2(i, X_glyph, dict_dist_loc):
    """Exports the glyph of X_glyph_1 closest to the glyph i of X_glyph_2."""
    l = [dict_dist_loc[(j, i)] for j in range(len(X_glyph))]
    return (min(l), np.argmin(l))


def similarity_between_languages_1(X_glyph_1, X_glyph_2, dict_dist_loc):
    """
    Computes the similarity of X_glyph_1 to X_glyph_2.
    Corresponds to d^tilde(X_glyph_1, X_glyph_2) in the paper.
    """
    l = []
    for i in range(len(X_glyph_1)):
        l.append(closest_glyph_1(i, X_glyph_2, dict_dist_loc)[0])
    return np.mean(l)


def similarity_between_languages_2(X_glyph_1, X_glyph_2, dict_dist_loc):
    """
    Computes the similarity of X_glyph_2 to X_glyph_1.
    Corresponds to d^tilde(X_glyph_2, X_glyph_1) in the paper.
    """
    l = []
    for i in range(len(X_glyph_2)):
        l.append(closest_glyph_2(i, X_glyph_1, dict_dist_loc)[0])
    return np.mean(l)


def similarity_between_languages(X_glyph_1, X_glyph_2, dict_dist_loc):
    """
    Computes the siamese-based distance between X_glyph_1 and X_glyph_2.
    Corresponds to d(X_glyph_1, X_glyph_2) in the paper.
    """
    sim_1 = similarity_between_languages_1(X_glyph_1, X_glyph_2, dict_dist_loc)
    sim_2 = similarity_between_languages_2(X_glyph_1, X_glyph_2, dict_dist_loc)
    return (sim_1 + sim_2) / 2
