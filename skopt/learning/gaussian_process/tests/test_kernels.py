import numpy as np
from scipy import optimize
from sklearn.utils.testing import assert_array_almost_equal

from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import DotProduct
from skopt.learning.gaussian_process.kernels import Exponentiation
from skopt.learning.gaussian_process.kernels import ExpSineSquared
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
        RBF(length_scale=length_scale) + Matern(length_scale=length_scale, nu=1.5),
        RBF(length_scale=length_scale) * Matern(length_scale=length_scale, nu=1.5),
        DotProduct(sigma_0=2.0)
    ])


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

def test_gradient_correctness():
    rng = np.random.RandomState(0)
    X = rng.randn(5)
    Y = rng.randn(10, 5)
    for kernel in KERNELS:
        yield check_gradient_correctness, kernel, X, Y

def test_gradient_finiteness():
    """
    When x is the same as X_train, gradients might become undefined because
    they are divided by d(x, X_train).
    Check they are equal to numerical gradients at such points.
    """
    for random_state in [0, 1]:
        rng = np.random.RandomState(random_state)
        X = rng.randn(5).tolist()
        Y = [X]
        for kernel in KERNELS:
            yield check_gradient_correctness, kernel, X, Y, 1e-6
