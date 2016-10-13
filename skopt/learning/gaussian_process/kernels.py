from math import sqrt

import numpy as np
from sklearn.gaussian_process.kernels import Kernel as sk_Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as sk_ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct as sk_DotProduct
from sklearn.gaussian_process.kernels import Exponentiation as sk_Exponentiation
from sklearn.gaussian_process.kernels import ExpSineSquared as sk_ExpSineSquared
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import Product as sk_Product
from sklearn.gaussian_process.kernels import RationalQuadratic as sk_RationalQuadratic
from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import WhiteKernel as sk_WhiteKernel


class Kernel(sk_Kernel):
    """
    Base class for skopt.gaussian_process kernels.
    Supports computation of the gradient of the kernel with respect to X
    """
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def gradient_X(self, x, X_train):
        """
        Computes gradient of K(x, X_train) with respect to x

        Parameters
        ----------
        x: array-like, shape=(n_features,)
            A single test point.

        Y: array-like, shape=(n_samples, n_features)
            Training data used to fit the gaussian process.

        Returns
        -------
        gradient_X: array-like, shape=(n_samples, n_features)
            Gradient of K(x, X_train) with respect to x.
        """
        raise NotImplementedError


class RBF(Kernel, sk_RBF):
    def gradient_X(self, x, X_train):
        # TODO: Do in-place operations
        length_scale = np.array(self.length_scale)
        diff = x - X_train
        diff /= length_scale
        squared = np.exp(-0.5 * np.sum(diff**2, axis=1))
        squared = np.expand_dims(squared, axis=1)
        return -squared * (diff / length_scale)


class Matern(Kernel, sk_Matern):
    def gradient_X(self, x, X_train):
        length_scale = np.array(self.length_scale)

        # euclidean distance
        diff = x - X_train
        diff /= length_scale
        sq_dist = np.sum(diff**2, axis=1)
        dist = np.sqrt(sq_dist)

        if self.nu == 0.5:
            e_dist = -np.exp(-dist) / dist
            e_dist = np.expand_dims(e_dist, axis=1)
            return e_dist * (diff / length_scale)

        elif self.nu == 1.5:
            # grad(fg) = f'g + fg'
            # where f = 1 + sqrt(3) * euclidean((X - Y) / length_scale)
            # where g = exp(-sqrt(3) * euclidean((X - Y) / length_scale))
            f = np.expand_dims(1 + sqrt(3) * dist, axis=1)

            dist_expand = np.expand_dims(sqrt(3) / dist, axis=1)
            f_grad = dist_expand * (diff / length_scale)

            g = np.expand_dims(np.exp(-sqrt(3) * dist), axis=1)
            g_grad = -g * f_grad
            return f * g_grad + g * f_grad

        elif self.nu == 2.5:
            # grad(fg) = f'g + fg'
            # where f = (1 + sqrt(5) * euclidean((X - Y) / length_scale) +
            #            5 / 3 * sqeuclidean((X - Y) / length_scale))
            # where g = exp(-sqrt(5) * euclidean((X - Y) / length_scale))
            f1 = sqrt(5) * dist
            f2 = (5.0 / 3.0) * sq_dist
            f = np.expand_dims(1 + f1 + f2, axis=1)

            dist_expand = np.expand_dims(sqrt(5) / dist, axis=1)
            diff_by_ls = diff / length_scale
            f1_grad = dist_expand * diff_by_ls
            f2_grad = (10.0 / 3.0) * diff_by_ls
            f_grad = f1_grad + f2_grad

            g = np.expand_dims(np.exp(-sqrt(5) * dist), axis=1)
            g_grad = -g * f1_grad
            return f * g_grad + g * f_grad


class RationalQuadratic(Kernel, sk_RationalQuadratic):

    def gradient_X(self, x, X_train):
        alpha = self.alpha
        length_scale = self.length_scale

        diff = x - X_train
        diff /= length_scale
        sq_dist = np.sum(diff**2, axis=1)
        sq_dist /= (2 * self.alpha)
        sq_dist += 1
        sq_dist **= (-alpha - 1)
        sq_dist *= -1

        sq_dist = np.expand_dims(sq_dist, axis=1)
        diff_by_ls = diff / length_scale
        return sq_dist * diff_by_ls


class ExpSineSquared(Kernel, sk_ExpSineSquared):

    def gradient_X(self, x, X_train):
        length_scale = self.length_scale
        periodicity = self.periodicity

        diff = x - X_train
        sq_dist = np.sum(diff**2, axis=1)
        dist = np.sqrt(sq_dist)

        pi_by_period = dist * (np.pi / periodicity)
        sine = np.sin(pi_by_period) / length_scale
        sine_squared = -2 * sine**2
        exp_sine_squared = np.exp(sine_squared)

        grad_wrt_exp = -2 * np.sin(2 * pi_by_period) / length_scale**2
        grad_wrt_theta = np.pi / (periodicity * dist)
        return np.expand_dims(
            grad_wrt_theta * exp_sine_squared * grad_wrt_exp, axis=1) * diff


class ConstantKernel(Kernel, sk_ConstantKernel):

    def gradient_X(self, x, X_train):
        return np.zeros_like(X_train)


class WhiteKernel(Kernel, sk_WhiteKernel):

    def gradient_X(self, x, X_train):
        return np.zeros_like(X_train)


class Exponentiation(Kernel, sk_Exponentiation):

    def gradient_X(self, x, X_train):
        expo = self.exponent
        kernel = self.kernel

        K = np.expand_dims(
            kernel(np.expand_dims(x, axis=0), X_train)[0], axis=1)
        return expo * K ** (expo - 1) * kernel.gradient_X(x, X_train)


class Sum(Kernel, sk_Sum):

    def gradient_X(self, x, X_train):
        return (
            self.k1.gradient_X(x, X_train) +
            self.k2.gradient_X(x, X_train)
        )

class Product(Kernel, sk_Product):

    def gradient_X(self, x, X_train):
        X = np.expand_dims(x, axis=0)
        f_ggrad = (
            np.expand_dims(self.k1(x, X_train)[0], axis=1) *
            self.k2.gradient_X(x, X_train)
        )
        fgrad_g = (
            np.expand_dims(self.k2(x, X_train)[0], axis=1) *
            self.k1.gradient_X(x, X_train)
        )
        return f_ggrad + fgrad_g


class DotProduct(Kernel, sk_DotProduct):

    def gradient_X(self, x, X_train):
        return X_train
