"""
Kernels not in sklearn.gaussian_process.kernels
"""

import numpy as np

from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import StationaryKernelMixin

class HammingKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    The HammingKernel is used to handle categorical inputs.

    ``K(x_1, x_2)`` is given by \sum_{j=1}^{d}exp(-ls_j * (I(x_1j != x_2j)))
    If all the categorical features of two samples are the same then
    ``K(x_1, x_2)`` reduces to 1 and if they are completely different,
    it reduces to ``exp(-d)``

    Parameters
    -----------
    * `length_scale` [float, array-like, shape=[n_features,], 1.0 (default)]
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    * `length_scale_bounds` [array-like, [1e-5, 1e5] (default)]
        The lower and upper bound on length_scale
    """

    def __init__(self, length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5)):
        if np.iterable(length_scale):
            if len(length_scale) > 1:
                self.anisotropic = True
                self.length_scale = np.asarray(length_scale, dtype=np.float)
            else:
                self.anisotropic = False
                self.length_scale = float(length_scale[0])
        else:
            self.anisotropic = False
            self.length_scale = float(length_scale)
        self.length_scale_bounds = length_scale_bounds

        if self.anisotropic:  # anisotropic length_scale
            self.hyperparameter_length_scale = \
                Hyperparameter("length_scale", "numeric", length_scale_bounds,
                               len(length_scale))
        else:
            self.hyperparameter_length_scale = \
                Hyperparameter("length_scale", "numeric", length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)

        * `Y` [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        * `eval_gradient` [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        * `K` [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)

        * `K_gradient` [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        length_scale = self.length_scale
        X = np.atleast_2d(X)
        if self.anisotropic and X.shape[1] != len(length_scale):
            raise ValueError(
                "Expected X to have %d features, got %d" %
                (X.shape, len(length_scale)))

        n_samples, n_dim = X.shape

        if eval_gradient:
            if self.anisotropic:
                grad = np.zeros((n_samples, n_samples, n_dim))
            else:
                grad = np.zeros((n_samples, n_samples, 1))

        Y_is_None = Y is None
        if Y_is_None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        if Y_is_None:
            whd = np.ones((n_samples, n_samples))
        else:
            whd = np.ones((n_samples, Y.shape[0]))

        for i in range(n_samples):

            if Y_is_None:
                start_ind = i + 1
                end_ind = n_samples
            else:
                start_ind = 0
                end_ind = Y.shape[0]

            for j in range(start_ind, end_ind):

                mask = X[i] != Y[j]
                if self.anisotropic:
                    hamming_dist = np.exp(-np.sum(length_scale[mask]))
                else:
                    hamming_dist = np.exp(-length_scale * np.sum(mask))

                if Y_is_None:
                    whd[i, j] = whd[j, i] = hamming_dist
                else:
                    whd[i, j] = hamming_dist

                if eval_gradient:
                    if self.anisotropic:
                        grad[i, j][mask] = grad[j, i][mask] = -hamming_dist
                    else:
                        ind_sum = np.sum(mask)
                        grad[i, j][0] = grad[j, i][0] = -ind_sum * hamming_dist

        if eval_gradient:
            return whd, grad
        return whd
