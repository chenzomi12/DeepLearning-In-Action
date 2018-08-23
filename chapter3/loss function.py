__author__ = 'm.bashari'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

###############################################

###############################################
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)


###############################################

###############################################
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)


###############################################

###############################################
def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = np.log(np.clip(y_pred, 10e-6, None) + 1.)
    second_log = np.log(np.clip(y_true, 10e-6, None) + 1.)
    return np.mean(np.square(first_log - second_log), axis=-1)


###############################################

###############################################
def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs((y_pred - y_true) / np.clip(np.abs(y_true), 10e-6, None))
    return 100 * np.mean(diff, axis=-1)


###############################################

###############################################
def hinge(y_true, y_pred):
    return np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1)


###############################################

###############################################
def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 10e-6))
