import numpy as np

from scipy import optimize

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises

from skopt.acquisition import acquisition_1D
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.acquisition import gaussian_pi
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.kernels import WhiteKernel


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


def check_gradient_correctness(X_new, model, acq_func, y_opt):
    analytic_grad = acquisition_1D(X_new, model, y_opt, acq_func)[1]
    num_grad_func = lambda x: acquisition_1D(
        x, model, y_opt, acq_func=acq_func)[0]
    num_grad = optimize.approx_fprime(X_new, num_grad_func, 1e-5)
    assert_array_almost_equal(analytic_grad, num_grad, 4)


def test_acquisition_gradient():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = rng.randn(20)
    X_new = rng.randn(5)
    mat = Matern()
    wk = WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=mat + wk)
    gpr.fit(X, y)

    for acq_func in ["LCB", "PI", "EI"]:
        check_gradient_correctness(X_new, gpr, acq_func, np.max(y))
