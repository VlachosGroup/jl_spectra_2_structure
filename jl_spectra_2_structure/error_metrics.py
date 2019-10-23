# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np

def get_r2(y_true, y_pred):
    SStot = np.sum((y_true-y_true.mean())**2)
    SSres = np.sum((y_true-y_pred)**2)
    return 1 - SSres/SStot

def get_rmse(y_true, y_pred):
    SSres = np.mean((y_true-y_pred)**2)
    return SSres**0.5

def get_max_error(y_true, y_pred):
    return np.array(y_pred-y_true)[np.argmax(np.abs(y_pred-y_true))]
