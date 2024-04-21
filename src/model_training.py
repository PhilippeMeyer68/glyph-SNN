# -*- coding: utf-8 -*-
"""
In this script we train the Siamese Neural Network model and save it for reuse.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 04/2024
"""


import os

from keras.optimizers import SGD

import model


def main():
    """
    This function sets the current working directory to the project folder, initializes
    the SGD optimizer, loads the Siamese Neural Network model, compiles it, loads the
    Siamese Neural Network class, trains the Siamese Neural Network model and saves the
    trained model.
    """

    # We set the current working directory to the project folder.
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # We choose the SGD optimizer.
    optimizer = SGD(lr=0.001, momentum=0.5)

    # We load the Siamese Neural Network model.
    siamese_net = model.siamese_net
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    # We load the Siamese Neural Network class.
    loader = model.Siamese_Loader()

    # We train the Siamese Neural Network model.
    batch_size = 16
    n_iter = 1350000
    for i in range(1, n_iter):
        (inputs, targets) = loader.get_batch(batch_size)
        loss = siamese_net.train_on_batch(inputs, targets)
        print(i, loss)

    # We save the model.
    siamese_net.save("models/siamese")


if __name__ == "__main__":
    main()
