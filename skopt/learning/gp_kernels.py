from math import sqrt

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel as sk_ConstantKernel
from sklearn.gaussian_process.kernels import Exponentiation as sk_Exponentiation
from sklearn.gaussian_process.kernels import ExpSineSquared as sk_ExpSineSquared
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import RationalQuadratic as sk_RationalQuadratic
from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import WhiteKernel as sk_WhiteKernel

class RBF(sk_RBF):
    def gradient_X(self, X, Y):
        """
        Computes gradient of K(X, Y) with respect to X
        """
        # TODO: Do in-place operations
        length_scale = np.array(self.length_scale)
        diff = X - Y
        diff /= length_scale
        squared = np.exp(-0.5 * np.sum(diff**2, axis=1))
        squared = np.expand_dims(squared, axis=1)
        return -squared * (diff / length_scale)


class Matern(sk_Matern):
    def gradient_X(self, X, Y):
        length_scale = np.array(self.length_scale)

        # euclidean distance
        diff = X - Y
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


class RationalQuadratic(sk_RationalQuadratic):

    def gradient_X(self, X, Y):
        alpha = self.alpha
        length_scale = self.length_scale

        diff = X - Y
        diff /= length_scale
        sq_dist = np.sum(diff**2, axis=1)
        sq_dist /= (2 * self.alpha)
        sq_dist += 1
        sq_dist **= (-alpha - 1)
        sq_dist *= -1

        sq_dist = np.expand_dims(sq_dist, axis=1)
        diff_by_ls = diff / length_scale
        return sq_dist * diff_by_ls


class ExpSineSquared(sk_ExpSineSquared):

    def gradient_X(self, X, Y):
        length_scale = self.length_scale
        periodicity = self.periodicity

        diff = X - Y
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


class ConstantKernel(sk_ConstantKernel):

    def gradient_X(self, X, Y):
        return np.zeros_like(Y)


class WhiteKernel(sk_WhiteKernel):

    def gradient_X(self, X, Y):
        return np.zeros_like(Y)


class Exponentiation(sk_Exponentiation):

    def gradient_X(self, X, Y):
        expo = self.exponent
        kernel = self.kernel

        kernel_value = np.expand_dims(
            kernel(np.expand_dims(X, axis=0), Y)[0], axis=1)
        return expo * kernel_value ** (expo - 1) * kernel.gradient_X(X, Y)


class Sum(sk_Sum):

    def gradient_X(self, X, Y):
        return self.k1.gradient_X(X, Y) + self.k2.gradient_X(X, Y)
