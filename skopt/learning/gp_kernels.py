from math import sqrt

import numpy as np
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import RationalQuadratic as sk_RationalQuadratic
from sklearn.gaussian_process.kernels import RBF as sk_RBF


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
