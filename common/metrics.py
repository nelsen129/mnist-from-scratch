import numpy as np


def mse(y_true, y_pred):
    return (y_pred - y_true) ** 2


def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true)


def categorical_acc(y_true, y_pred):
    y_true_label = np.argmax(y_true, axis=-1)
    y_pred_label = np.argmax(y_pred, axis=-1)
    return np.count_nonzero(y_pred_label == y_true_label) / len(y_true_label)
