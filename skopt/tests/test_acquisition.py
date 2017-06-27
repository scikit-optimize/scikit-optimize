from math import log
import numpy as np
import pytest

from scipy import optimize

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises

from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.acquisition import gaussian_pi
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.kernels import WhiteKernel


class ConstSurrogate:
    def predict(self, X, return_std=True):
        X = np.array(X)
        return np.zeros(X.shape[0]), np.ones(X.shape[0])


class MultiOutputSurrogate:
    def fit(self, X, y):
        """
        The first estimator returns a constant value.
        The second estimator is a gaussian process regressor that
        models the logarithm of the time.
        """
        X = np.array(X)
        y = np.array(y)
        gpr = GaussianProcessRegressor()
        gpr.fit(X, y[:, 1])
        self.estimators_ = []
        self.estimators_.append(ConstSurrogate())
        self.estimators_.append(gpr)
        return self


@pytest.mark.fast_test
def test_acquisition_ei_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    ei = gaussian_ei(X, ConstSurrogate(), -0.5, xi=0.)
    assert_array_almost_equal(ei, [0.1977966] * 4)


@pytest.mark.fast_test
def test_acquisition_pi_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    pi = gaussian_pi(X, ConstSurrogate(), -0.5, xi=0.)
    assert_array_almost_equal(pi, [0.308538] * 4)


@pytest.mark.fast_test
def test_acquisition_lcb_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    lcb = gaussian_lcb(X, ConstSurrogate(), kappa=0.3)
    assert_array_almost_equal(lcb, [-0.3] * 4)


@pytest.mark.fast_test
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
    analytic_grad = gaussian_acquisition_1D(
        X_new, model, y_opt, acq_func)[1]
    num_grad_func = lambda x:  gaussian_acquisition_1D(
        x, model, y_opt, acq_func=acq_func)[0]
    num_grad = optimize.approx_fprime(X_new, num_grad_func, 1e-5)
    assert_array_almost_equal(analytic_grad, num_grad, 4)


@pytest.mark.fast_test
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


def test_acquisition_per_second():
    X = [[3], [5], [7]]
    y = [[1, log(3)], [1, log(5)], [1, log(7)]]
    mos = MultiOutputSurrogate()
    mos.fit(X, y)

    X_pred =  [[1], [2], [4], [6], [8], [9]]
    for acq_func in ["EIps", "PIps"]:
        vals = _gaussian_acquisition(X_pred, mos, y_opt=1.0, acq_func=acq_func)
        for fast, slow in zip([0, 1, 2], [5, 4, 3]):
            assert_greater(vals[slow], vals[fast])
