# -*- coding: utf-8 -*-
"""
In this script we train the Siamese Neural Network model and save it for reuse.

Author: Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 02/2024
"""

# packages
import model
import os
from keras.optimizers import SGD


def main():
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
    for i in range(1, 5):
        (inputs, targets) = loader.get_batch(batch_size)
        loss = siamese_net.train_on_batch(inputs, targets)
        print(i, loss)

    # We save the model.
    siamese_net.save("models/siamese")


if __name__ == "__main__":
    main()
