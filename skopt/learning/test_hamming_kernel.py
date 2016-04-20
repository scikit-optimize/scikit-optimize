import numpy as np
from scipy.spatial.distance import pdist, squareform

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal

from skopt.learning.kernels import HammingKernel


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


def test_isotropic_kernel():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 4, (5, 3))
    hm = HammingKernel()

    # Scipy calulates the mean. We need exp(-sum)
    hamming_distance = squareform(pdist(X, metric='hamming'))
    scipy_dist = np.exp(-hamming_distance * X.shape[1])
    assert_array_almost_equal(scipy_dist, hm(X))


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


def test_kernel_gradient():
    rng = np.random.RandomState(0)
    hm = HammingKernel()
    X = rng.randint(0, 4, (5, 3))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (5, 5, 1))

    def eval_kernel_for_exp_theta(exp_theta, kernel):
        kernel_clone = kernel.clone_with_theta(np.log(exp_theta))
        K = kernel_clone(X, eval_gradient=False)
        return K

    K_gradient_approx = _approx_fprime(
        np.exp(hm.theta), eval_kernel_for_exp_theta, 1e-10, (hm,))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)

    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    K_gradient_approx = _approx_fprime(
        np.exp(hm.theta), eval_kernel_for_exp_theta, 1e-10, (hm,))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (5, 5, 3))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)

    X = rng.randint(0, 4, (3, 2))
    hm = HammingKernel(length_scale=[0.1, 2.0])
    K_gradient_approx = _approx_fprime(
        np.exp(hm.theta), eval_kernel_for_exp_theta, 1e-10, (hm,))
    K, K_gradient = hm(X, eval_gradient=True)
    assert_array_equal(K_gradient.shape, (3, 3, 2))
    assert_array_almost_equal(K_gradient_approx, K_gradient, 4)


def test_Y_is_not_None():
    rng = np.random.RandomState(0)
    hm = HammingKernel()
    X = rng.randint(0, 4, (5, 3))

    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])
    assert_array_equal(hm(X), hm(X, X))


def test_gp_regressor():
    rng = np.random.RandomState(0)
    X = np.asarray([
        ["ham", "spam", "ted"],
        ["ham", "ted", "ted"],
        ["ham", "spam", "spam"]])
    y = rng.randn(3)
    hm = HammingKernel(length_scale=[1.0, 1.0, 1.0])

    gpr = GaussianProcessRegressor(hm)
    gpr.fit(X, y)
    assert_array_almost_equal(gpr.predict(X), y)
    assert_array_almost_equal(gpr.predict(X[:2]), y[:2])
