# 2048-ML-CEM

This repository has all of the files for a neural network trained using the Cross-Entropy Method to play 2048.

Each file's purpose is as follows:

game.py - acts as a the controller to play a game of 2048

Enviornment.py - acts as the intermediary between the game file and MyFlow providing the neural network with the state and score

MyFlow.py - this is the custom machine learning library, it contains classes for a fully connected neural network, and a custom neural network architecture which allows layers to feed into themselves. It also has support for genetic training.

Noter.ipynb - This is the notebook which creates networks and trains them.
