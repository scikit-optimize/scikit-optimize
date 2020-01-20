import numpy as np
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
try:
    from sklearn.preprocessing import OrdinalEncoder
    UseOrdinalEncoder = True
except ImportError:
    UseOrdinalEncoder = False
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import pytest

from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import DotProduct
from skopt.learning.gaussian_process.kernels import ExpSineSquared
from skopt.learning.gaussian_process.kernels import HammingKernel
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.kernels import RationalQuadratic
from skopt.learning.gaussian_process.kernels import RBF
from skopt.learning.gaussian_process.kernels import WhiteKernel


KERNELS = []

for length_scale in [np.arange(1, 6), [0.2, 0.3, 0.5, 0.6, 0.1]]:
    KERNELS.extend([
        RBF(length_scale=length_scale),
        Matern(length_scale=length_scale, nu=0.5),
        Matern(length_scale=length_scale, nu=1.5),
        Matern(length_scale=length_scale, nu=2.5),
        RationalQuadratic(alpha=2.0, length_scale=2.0),
        ExpSineSquared(length_scale=2.0, periodicity=3.0),
        ConstantKernel(constant_value=1.0),
        WhiteKernel(noise_level=2.0),
        Matern(length_scale=length_scale, nu=2.5) ** 3.0,
        RBF(length_scale=length_scale) + Matern(length_scale=length_scale,
                                                nu=1.5),
        RBF(length_scale=length_scale) * Matern(length_scale=length_scale,
                                                nu=1.5),
        DotProduct(sigma_0=2.0)
    ])


# Copied (shamelessly) from sklearn.gaussian_process.kernels
def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk), ), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


def kernel_X_Y(x, y, kernel):
    X = np.expand_dims(x, axis=0)
    Y = np.expand_dims(y, axis=0)
    return kernel(X, Y)[0][0]


def numerical_gradient(X, Y, kernel, step_size=1e-4):
    grad = []
    for y in Y:
        num_grad = optimize.approx_fprime(X, kernel_X_Y, step_size, y, kernel)
        grad.append(num_grad)
    return np.asarray(grad)


def check_gradient_correctness(kernel, X, Y, step_size=1e-4):
    X_grad = kernel.gradient_x(X, Y)
    num_grad = numerical_gradient(X, Y, kernel, step_size)
    assert_array_almost_equal(X_grad, num_grad, decimal=3)


@pytest.mark.fast_test
@pytest.mark.parametrize("kernel", KERNELS)
def test_gradient_correctness(kernel):
    rng = np.random.RandomState(0)
    X = rng.randn(5)
    Y = rng.randn(10, 5)
    check_gradient_correctness(kernel, X, Y)


@pytest.mark.fast_test
@pytest.mark.parametrize("random_state", [0, 1])
@pytest.mark.parametrize("kernel", KERNELS)
def test_gradient_finiteness(random_state, kernel):
    """
    When x is the same as X_train, gradients might become undefined because
    they are divided by d(x, X_train).
    Check they are equal to numerical gradients at such points.
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(5).tolist()
    Y = [X]
    check_gradient_correctness(kernel, X, Y, 1e-6)


@pytest.mark.fast_test
def test_distance_string():
    # Inspired by test_hamming_string_array in scipy.tests.test_distance
    a = np.array(['eggs', 'spam', 'spam', 'eggs', 'spam', 'spam', 'spam',
                  'spam', 'spam', 'spam', 'spam', 'eggs', 'eggs', 'spam',
                  'eggs', 'eggs', 'eggs', 'eggs', 'eggs', 'spam'],
                 dtype='|S4')
    b = np.array(['eggs', 'spam', 'spam', 'eggs', 'eggs', 'spam', 'spam',
                  'spam', 'spam', 'eggs', 'spam', 'eggs', 'spam', 'eggs',
                  'spam', 'spam', 'eggs', 'spam', 'spam', 'eggs'],
                 dtype='|S4')
    true_values = np.array([[0, 0.45], [0.45, 0]])
    X = np.vstack((a, b))
    hm = HammingKernel()
    assert_array_almost_equal(-np.log(hm(X)) / 20.0, true_values)


@pytest.mark.fast_test
def test_isotropic_kernel():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 4, (5, 3))
    hm = HammingKernel()

    # Scipy calulates the mean. We need exp(-sum)
    hamming_distance = squareform(pdist(X, metric='hamming'))
    scipy_dist = np.exp(-hamming_distance * X.shape[1])
    assert_array_almost_equal(scipy_dist, hm(X))


@pytest.mark.fast_test
def test_anisotropic_kernel():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 4, (5, 3))
    hm = HammingKernel()
    X_kernel = hm(X)
    hm_aniso = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    X_kernel_aniso = hm_aniso(X)
    assert_array_almost_equal(X_kernel, X_kernel_aniso)

    hm = HammingKernel(length_scale=2.0)
    X_kernel = hm(X)
    hm_aniso = HammingKernel(length_scale=[2.0, 2.0, 2.0])
    X_kernel_aniso = hm_aniso(X)
    assert_array_almost_equal(X_kernel, X_kernel_aniso)


@pytest.mark.fast_test
def test_kernel_gradient():
    rng = np.random.RandomState(0)
    hm = HammingKernel(length_scale=2.0)
    X = rng.randint(0, 4, (5, 3))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (5, 5, 1))

    def eval_kernel_for_theta(theta, kernel):
        kernel_clone = kernel.clone_with_theta(theta)
        K = kernel_clone(X, eval_gradient=False)
        return K

    K_gradient_approx = _approx_fprime(
        hm.theta, eval_kernel_for_theta, 1e-10, (hm,))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)

    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    K_gradient_approx = _approx_fprime(
        hm.theta, eval_kernel_for_theta, 1e-10, (hm,))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (5, 5, 3))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)

    X = rng.randint(0, 4, (3, 2))
    hm = HammingKernel(length_scale=[0.1, 2.0])
    K_gradient_approx = _approx_fprime(
        hm.theta, eval_kernel_for_theta, 1e-10, (hm,))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (3, 3, 2))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)


@pytest.mark.fast_test
def test_Y_is_not_None():
    rng = np.random.RandomState(0)
    hm = HammingKernel()
    X = rng.randint(0, 4, (5, 3))

    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    assert_array_equal(hm(X), hm(X, X))


@pytest.mark.fast_test
def test_gp_regressor():
    rng = np.random.RandomState(0)
    X = np.asarray([
        ["ham", "spam", "ted"],
        ["ham", "ted", "ted"],
        ["ham", "spam", "spam"]])
    y = rng.randn(3)
    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    if UseOrdinalEncoder:
        enc = OrdinalEncoder()
        enc.fit(X)

    gpr = GaussianProcessRegressor(hm)
    if UseOrdinalEncoder:
        gpr.fit(enc.transform(X), y)
        assert_array_almost_equal(gpr.predict(enc.transform(X)), y)
        assert_array_almost_equal(gpr.predict(enc.transform(X[:2])), y[:2])
    else:
        gpr.fit(X, y)
        assert_array_almost_equal(gpr.predict(X), y)
        assert_array_almost_equal(gpr.predict(X[:2]), y[:2])
