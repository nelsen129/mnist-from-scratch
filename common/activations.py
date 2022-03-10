import numpy as np


# Standard ReLU. Returns x if x > 0, else 0.
def relu(x):
    return np.fmax(x, 0.)


# Derivative of ReLU. Returns 1 if x > 0, else 0. Discontinuous at 0
def d_relu(x):
    return np.where(x > 0., 1., 0.)


# Standard sigmoid (logistic) function. Limits x to between 0 and 1.
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# Derivative of sigmoid.
def d_sigmoid(x):
    return np.exp(x) / (1. + np.exp(x)) ** 2


# Linear activation. Returns x. Equivalent to not having an activation layer.
def linear(x):
    return x


# Derivative of linear activation.
def d_linear(x):
    return 1.


# Maps an activation key to both its function and derivative.
activation_dict = {
    'relu': [relu, d_relu],
    'sigmoid': [sigmoid, d_sigmoid],
    'linear': [linear, d_linear]
}
