import numpy as np
from scipy import optimize
from sklearn.utils.testing import assert_array_almost_equal

from skopt.learning.gp_kernels import ConstantKernel
from skopt.learning.gp_kernels import DotProduct
from skopt.learning.gp_kernels import Exponentiation
from skopt.learning.gp_kernels import ExpSineSquared
from skopt.learning.gp_kernels import Matern
from skopt.learning.gp_kernels import Product
from skopt.learning.gp_kernels import RationalQuadratic
from skopt.learning.gp_kernels import RBF
from skopt.learning.gp_kernels import Sum
from skopt.learning.gp_kernels import WhiteKernel

KERNELS = [
    RBF(length_scale=[1.0, 2.0, 3.0]),
    Matern(length_scale=[1.0, 2.0, 3.0], nu=0.5),
    Matern(length_scale=[1.0, 2.0, 3.0], nu=1.5),
    Matern(length_scale=[1.0, 2.0, 3.0], nu=2.5),
    RationalQuadratic(alpha=2.0, length_scale=2.0),
    ExpSineSquared(length_scale=2.0, periodicity=3.0),
    ConstantKernel(constant_value=1.0),
    WhiteKernel(noise_level=2.0),
    Matern(length_scale=[1.0, 2.0, 3.0], nu=2.5) ** 3.0,
    RBF(length_scale=[1.0, 2.0, 3.0]) + Matern(length_scale=[1.0, 2.0, 3.0], nu=1.5),
    RBF(length_scale=[1.0, 2.0, 3.0]) * Matern(length_scale=[1.0, 2.0, 3.0], nu=1.5),
    DotProduct(sigma_0=2.0)
]

rng = np.random.RandomState(0)
X = rng.randn(3)
Y = rng.randn(10, 3)

def kernel_X_Y(x, y, kernel):
    X = np.expand_dims(x, axis=0)
    Y = np.expand_dims(y, axis=0)
    return kernel(X, Y)[0][0]

def numerical_gradient(X, Y, kernel):
    grad = []
    for y in Y:
        num_grad = optimize.approx_fprime(X, kernel_X_Y, 1e-4, y, kernel)
        grad.append(num_grad)
    return np.asarray(grad)

def test_gradient_correctness():
    for kernel in KERNELS:
        X_grad = kernel.gradient_X(X, Y)
        num_grad = numerical_gradient(X, Y, kernel)
        assert_array_almost_equal(X_grad, num_grad, decimal=3)
