import numpy as np


# Mean Square Error. Will need to be averaged across the batch dimension to get full mse.
def mse(y_true, y_pred):
    return (y_pred - y_true) ** 2


# Derivative of mse, for backpropagation
def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true)


# Categorical Accuracy. Number of samples where y_true == y_pred. Needs to be divided by the number of samples to get
#   total percent.
def categorical_acc(y_true, y_pred):
    y_true_label = np.argmax(y_true, axis=-1)
    y_pred_label = np.argmax(y_pred, axis=-1)
    return np.count_nonzero(y_pred_label == y_true_label)
