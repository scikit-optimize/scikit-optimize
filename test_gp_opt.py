from math import pi, cos

import numpy as np
from sklearn.utils.testing import assert_less

from gp_opt import gp_minimize

a = 1
b = 5.1 / (4 * pi**2)
c = 5.0 / pi
r = 6
s = 10
t = 1 / (8*pi)

P = 10**-4 * np.asarray([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381]])

A = np.asarray(
        [[10, 3, 17, 3.50, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]])

alpha = np.asarray([1.0, 1.2, 3.0, 3.2])

def branin(x):
    x1 = x[0]
    x2 = x[1]
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * cos(x1) + s

def hartmann_6(x):
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P)**2, axis=1)))

def test_branin_bayes():
    for random_state in [0, 1, 2, 3, 4]:
        x, f, d = gp_minimize(
            branin, [[-5, 10], [0, 15]], random_state=random_state,
            search='lbfgs', maxiter=200, acq='UCB')
        assert_less(f, 0.47)

        x, f, d = gp_minimize(
            branin, [[-5, 10], [0, 15]], random_state=random_state,
            search='sampling', maxiter=200, acq='UCB')
        assert_less(f, 0.41)

def test_hartmann_6():
    bounds = np.tile((0, 1), (6, 1))
    for random_state in [0, 1, 2, 3, 4]:
        x, f, d = gp_minimize(
            hartmann_6, bounds, random_state=random_state,
            search='lbfgs', maxiter=200, acq='UCB')
        assert_less(f, -3.0)

        x, f, d = gp_minimize(
            hartmann_6, bounds, random_state=random_state,
            search='sampling', maxiter=200, acq='UCB')
        assert_less(f, -2.5)
