# Module to import functions from in examples for multiprocessing backend
import numpy as np


def obj_fun(x, noise_level=0.1):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +\
           np.random.randn() * noise_level
