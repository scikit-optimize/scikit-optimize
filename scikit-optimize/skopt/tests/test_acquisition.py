import numpy as np
import pytest

from scipy import optimize

from sklearn.multioutput import MultiOutputRegressor
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb
from skopt.acquisition import gaussian_pi
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.kernels import WhiteKernel
from skopt.space import Space
from skopt.utils import cook_estimator


class ConstSurrogate:
    def predict(self, X, return_std=True):
        X = np.array(X)
        return np.zeros(X.shape[0]), np.ones(X.shape[0])

# This is used to test that given constant acquisition values at
# different points, acquisition functions "EIps" and "PIps"
# prefer candidate points that take lesser time.
# The second estimator mimics the GP regressor that is fit on
# the log of the input.

class ConstantGPRSurrogate(object):
    def __init__(self, space):
        self.space = space

    def fit(self, X, y):
        """
        The first estimator returns a constant value.
        The second estimator is a gaussian process regressor that
        models the logarithm of the time.
        """
        X = np.array(X)
        gpr = cook_estimator("GP", self.space, normalize_y=False)
        gpr.fit(X, np.log(np.ravel(X)))
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
def test_acquisition_variance_correctness():
    # check that it works with a vector as well
    X = 10 * np.ones((4, 2))
    var = gaussian_lcb(X, ConstSurrogate(), kappa='inf')
    assert_array_almost_equal(var, [-1.0] * 4)


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
    assert_array_almost_equal(analytic_grad, num_grad, 3)


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


@pytest.mark.fast_test
def test_acquisition_gradient_cookbook():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = rng.randn(20)
    X_new = rng.randn(5)
    gpr = cook_estimator("GP", Space(((-5.0, 5.0),)), random_state=0)
    gpr.fit(X, y)

    for acq_func in ["LCB", "PI", "EI"]:
        check_gradient_correctness(X_new, gpr, acq_func, np.max(y))


@pytest.mark.fast_test
@pytest.mark.parametrize("acq_func", ["EIps", "PIps"])
def test_acquisition_per_second(acq_func):
    X = np.reshape(np.linspace(4.0, 8.0, 10), (-1, 1))
    y = np.vstack((np.ones(10), np.ravel(np.log(X)))).T
    cgpr = ConstantGPRSurrogate(Space(((1.0, 9.0),)))
    cgpr.fit(X, y)

    X_pred = np.reshape(np.linspace(1.0, 11.0, 20), (-1, 1))
    indices = np.arange(6)
    vals = _gaussian_acquisition(X_pred, cgpr, y_opt=1.0, acq_func=acq_func)
    for fast, slow in zip(indices[:-1], indices[1:]):
        assert vals[slow] > vals[fast]

    acq_wo_time = _gaussian_acquisition(
        X, cgpr.estimators_[0], y_opt=1.2, acq_func=acq_func[:2])
    acq_with_time = _gaussian_acquisition(
        X, cgpr, y_opt=1.2, acq_func=acq_func)
    assert_array_almost_equal(acq_wo_time / acq_with_time, np.ravel(X), 2)


def test_gaussian_acquisition_check_inputs():
    model = ConstantGPRSurrogate(Space(((1.0, 9.0),)))
    with pytest.raises(ValueError) as err:
        _gaussian_acquisition(np.arange(1, 5), model)
    assert("it must be 2-dimensional" in err.value.args[0])


@pytest.mark.fast_test
@pytest.mark.parametrize("acq_func", ["EIps", "PIps"])
def test_acquisition_per_second_gradient(acq_func):
    rng = np.random.RandomState(0)
    X = rng.randn(20, 10)
    # Make the second component large, so that mean_grad and std_grad
    # do not become zero.
    y = np.vstack((X[:, 0], np.abs(X[:, 0])**3)).T

    for X_new in [rng.randn(10), rng.randn(10)]:
        gpr = cook_estimator("GP", Space(((-5.0, 5.0),)), random_state=0)
        mor = MultiOutputRegressor(gpr)
        mor.fit(X, y)
        check_gradient_correctness(X_new, mor, acq_func, 1.5)
