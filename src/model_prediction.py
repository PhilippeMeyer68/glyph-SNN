# -*- coding: utf-8 -*-
"""
In this script we predict the distances between alphabets using the Siamese
Neural Network model.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 03/2024
"""

# packages
import itertools
import numpy as np
import os
import pickle
import tensorflow as tf
from keras import backend as K


def main():
    # We set the current working directory to the project folder.
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    model_name = "siamese"
    siamese_net = tf.keras.models.load_model("models/" + model_name)

    alphabets = os.listdir("data/processed/alphabets")
    alphabets = sorted(alphabets)
    print("Number of alphabets", len(alphabets))

    # We create the pairwise distances between the alphabets.
    for i in range(len(alphabets) - 1):
        for j in range(i + 1, len(alphabets)):
            alphabet_1 = alphabets[i]
            alphabet_2 = alphabets[j]
            X_glyph_1 = np.load("data/processed/alphabets/" + alphabet_1)
            X_glyph_2 = np.load("data/processed/alphabets/" + alphabet_2)

            X_prod = np.array(
                list(
                    itertools.product(
                        X_glyph_1.reshape(len(X_glyph_1), 105, 105, 1),
                        X_glyph_2.reshape(len(X_glyph_2), 105, 105, 1),
                    )
                )
            )
            X_prod = [X_prod[:, 0, :, :], X_prod[:, 1, :, :]]

            pred = siamese_net.predict(X_prod)

            dict_dist = {}
            for i2 in range(len(X_glyph_1)):
                for j2 in range(len(X_glyph_2)):
                    k = i2 * len(X_glyph_2) + j2
                    dict_dist[(i2, j2)] = 1 - pred[k][0]

            # Export.
            pickle.dump(
                dict_dist,
                open(
                    "data/processed/distances/between_alphabets/dict_dist_X_"
                    + alphabet_1[2:-4]
                    + "_"
                    + alphabet_2[2:-4]
                    + "_"
                    + model_name
                    + ".dat",
                    "wb",
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    # We create the pairwise distances between the alphabets and inside the alphabets.
    for alphabet_1, alphabet_2 in [
        ["X_Latin.npy", "X_Old Italic.npy"],
        ["X_Coptic.npy", "X_Old Persian.npy"],
    ]:
        X_glyph_1 = np.load("data/processed/alphabets/" + alphabet_1)
        X_glyph_2 = np.load("data/processed/alphabets/" + alphabet_2)
        X_glyphs = np.concatenate((X_glyph_1, X_glyph_2))
        dict_name = (
            "dict_dist_X_pair_inside_"
            + alphabet_1[2:-4]
            + "_"
            + alphabet_2[2:-4]
            + "_"
            + model_name
            + ".dat"
        )

        X_prod = np.array(
            list(
                itertools.product(
                    X_glyphs.reshape(len(X_glyphs), 105, 105, 1),
                    X_glyphs.reshape(len(X_glyphs), 105, 105, 1),
                )
            )
        )
        X_prod = [X_prod[:, 0, :, :], X_prod[:, 1, :, :]]

        pred = siamese_net.predict(X_prod)
        dict_dist = {}
        for i2 in range(len(X_glyphs)):
            for j2 in range(i2 + 1, len(X_glyphs)):
                k = i2 * len(X_glyphs) + j2
                dict_dist[(i2, j2)] = 1 - pred[k][0]

        # Export.
        pickle.dump(
            dict_dist,
            open(
                "data/processed/distances/between_inside_alphabets/" + dict_name, "wb"
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    main()
