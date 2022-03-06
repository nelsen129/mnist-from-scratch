import numpy as np


def relu(x):
    return np.fmax(x, 0.)


def d_relu(x):
    return np.where(x > 0., 1., 0.)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid(x):
    return np.exp(x) / (1. + np.exp(x)) ** 2


def linear(x):
    return x


def d_linear(x):
    return 1.


activation_dict = {
    'relu': [relu, d_relu],
    'sigmoid': [sigmoid, d_sigmoid],
    'linear': [linear, d_linear]
}
