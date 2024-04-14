# -*- coding: utf-8 -*-
"""
This script defines the Siamese Neural Network model.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 03/2024
"""

# packages
import cv2
import keras
import numpy as np
import numpy.random as rng
import os
import tensorflow
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model, Sequential
from keras.regularizers import l2


class Siamese_Loader:
    """Siamese Neural Network class."""

    def __init__(self):
        self.data = {}
        self.categories = {}
        self.info = {}

        def loadimgs(path, n=0):
            """To load the images."""

            X = []
            y = []
            cat_dict = {}
            lang_dict = {}
            curr_y = n

            # We load every alphabet separately.
            for alphabet in os.listdir(path):
                print("loading alphabet: " + alphabet)
                lang_dict[alphabet] = [curr_y, None]
                alphabet_path = os.path.join(path, alphabet)
                for letter in os.listdir(alphabet_path):
                    cat_dict[curr_y] = (alphabet, letter)
                    category_images = []
                    letter_path = os.path.join(alphabet_path, letter)
                    for filename in os.listdir(letter_path):
                        image_path = os.path.join(letter_path, filename)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        category_images.append(image)
                        y.append(curr_y)
                        X.append(np.stack(category_images))

                    curr_y += 1
                    lang_dict[alphabet][1] = curr_y - 1
            y = np.vstack(y)
            X = np.stack(X)
            return X, y, lang_dict

        train_folder = "data/processed/omniglot_invented_augmented/images_background"
        X, y, c = loadimgs(train_folder)
        self.data["train"] = X
        self.categories["train"] = c

    def get_batch(self, batch_size, s="train"):
        """Creates batch of n pairs, half same class, half different class."""

        X = self.data[s]
        n_classes, n_examples, w, h = X.shape

        # Randomly sample several classes to use in the batch.
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        # Initialize 2 empty arrays for the input image batch.
        pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
        # Initialize vector for the targets, and make one half of it '1's, so
        # 2nd half of batch has same class.
        targets = np.zeros((batch_size,))
        targets[batch_size // 2 :] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, n_examples)
            # pick images of same class for 1st half, different for 2nd.
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to
                # ensure 2nd image has different category.
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
            pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)
        return pairs, targets


def W_init(shape, name=None, dtype=None):
    """Initializes weights."""

    values = rng.normal(loc=0, scale=0.01, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None, dtype=None):
    """Initializes bias."""

    values = rng.normal(loc=0.5, scale=0.01, size=shape)
    return K.variable(values, name=name)


# We set the current working directory to the project folder.
os.chdir(os.path.dirname(os.path.dirname(__file__)))

# We define the architecture of the Siamese Neural Network model.
lambj = 2e-4
input_shape = (105, 105, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
convnet = Sequential()
convnet.add(
    Conv2D(
        64,
        (10, 10),
        activation="relu",
        input_shape=input_shape,
        kernel_initializer=W_init,
        kernel_regularizer=l2(lambj),
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        128,
        (7, 7),
        activation="relu",
        kernel_regularizer=l2(lambj),
        kernel_initializer=W_init,
        bias_initializer=b_init,
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        128,
        (4, 4),
        activation="relu",
        kernel_initializer=W_init,
        kernel_regularizer=l2(lambj),
        bias_initializer=b_init,
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        256,
        (4, 4),
        activation="relu",
        kernel_initializer=W_init,
        kernel_regularizer=l2(lambj),
        bias_initializer=b_init,
    )
)
convnet.add(Flatten())
convnet.add(
    Dense(
        4096, activation="sigmoid", kernel_regularizer=l2(1e-3), bias_initializer=b_init
    )
)
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1, activation="sigmoid", bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
