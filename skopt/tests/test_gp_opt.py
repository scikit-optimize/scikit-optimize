import numpy as np
from math import pi, cos
from sklearn.utils.testing import assert_less

from skopt.gp_opt import gp_minimize
from skopt.benchmarks import branin
from skoprt.benchmarks import hartmann_6


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
