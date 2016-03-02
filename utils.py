__author__ = 'sumant'

import numpy as np


def convert_zero_mean(X, in_place=False):
    m, n = X.shape
    if not in_place:
        X = np.copy(X)
    mean_mat = np.mean(X, axis=0, keepdims=True)
    mean_mat = np.repeat(mean_mat, m, axis=0)
    X = X - mean_mat
    return X


def convert_zero_mean_unit_dev(X):
    std = np.std(X, axis=0, keepdims=True)
    X = convert_zero_mean(X)
    X = X / np.repeat(std, X.shape[0], axis=0)
    return X