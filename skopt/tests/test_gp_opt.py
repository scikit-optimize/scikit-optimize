import numpy as np
from math import pi, cos
from sklearn.utils.testing import assert_less

from skopt.gp_opt import gp_minimize


def branin(x, a=1, b=5.1 / (4 * pi ** 2), c=5. / pi,
           r=6, s=10, t=1. / (8 * pi)):
    return (a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 +
            s * (1 - t) * cos(x[0]) + s)


def hartmann_6(x,
               alpha=np.asarray([1.0, 1.2, 3.0, 3.2]),
               P=10**-4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                      [2329, 4135, 8307, 3736, 1004, 9991],
                                      [2348, 1451, 3522, 2883, 3047, 6650],
                                      [4047, 8828, 8732, 5743, 1091, 381]]),
               A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                             [0.05, 10, 17, 0.1, 8, 14],
                             [3, 3.5, 1.7, 10, 17, 8],
                             [17, 8, 0.05, 10, 0.1, 14]])):
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P)**2, axis=1)))


def check_branin(search):
    res = gp_minimize(branin, [[-5, 10], [0, 15]], random_state=0,
                      search=search, maxiter=150, acq='UCB')
    assert_less(res.fun, 0.5)


def test_branin():
    for search in ["lbfgs", "sampling"]:
        yield check_branin, search


def test_branin_hartmann_sampling():
    bounds = np.tile((0, 1), (6, 1))
    res = gp_minimize(hartmann_6, bounds, random_state=0,
                      search='sampling', maxiter=150, acq='UCB')
    assert_less(res.fun, -2.5)
