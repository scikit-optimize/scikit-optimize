import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises

from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.acquisition import gaussian_pi


class ConstSurrogate:
    def predict(self, X, return_std=True):
        return np.zeros(X.shape[0]), np.ones(X.shape[0])

def test_acquisition_ei_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    ei = gaussian_ei(X, ConstSurrogate(), -0.5, xi=0.)
    assert_array_almost_equal(ei, [0.1977966] * 4)

def test_acquisition_pi_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    pi = gaussian_pi(X, ConstSurrogate(), -0.5, xi=0.)
    assert_array_almost_equal(pi, [0.308538] * 4)

def test_acquisition_lcb_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    lcb = gaussian_lcb(X, ConstSurrogate(), kappa=0.3)
    assert_array_almost_equal(lcb, [-0.3] * 4)

def test_acquisition_api():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10)
    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    for method in [gaussian_ei, gaussian_lcb, gaussian_pi]:
        assert_array_equal(method(X, gpr).shape, 10)
        assert_raises(ValueError, method, rng.rand(10), gpr)
