"""Gaussian processes regression with a Beta Warping. """

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by Stefano Cereda <stefano1.cereda@mail.polimi.it>
#   Based on "Input Warping for Bayesian Optimization of Non-Stationary Functions"
#   By Snoek et al.
#
# License: BSD 3 clause

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
import scipy.stats as sps

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated
from sklearn.gaussian_process import GaussianProcessRegressor


def truncate_01(func):
    """Decorator function.

    BetaWarping needs Xs between 0 and 1. They should be already like this, but
    we must be sure about that.
    """
    def inner(cls_instance, inputs, *args):
        inputs = inputs.copy()
        if np.any(inputs < 0):
            warnings.warn('BetaWarp encountered negative values: %s' % inputs[inputs<0])
            inputs[inputs<0] = 0.0
        if np.any(inputs > 1):
            warnings.warn('BetaWarp encountered values above 1: %s' % inputs[inputs>1])
            inputs[inputs>1] = 1.0

        return func(cls_instance, inputs, *args)
    return inner


def _approx_fprime(xk, shape, f, epsilon, args=()):
    """Approximate the gradient.
    When optimizing the BetaWarping HPs, we need to compute their gradient, I haven't tried (yet) to do that
    analitically"""
    f0 = f(*((xk,) + args))
    grad = np.zeros((shape[0], shape[1], len(xk)), float)
    ei = np.zeros((len(xk), ), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:,:,k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


class BetaWarp(object):
    """BetaWarp transformer"""
    def __init__(self, num_dims, alphas=None, betas=None, scale=1.5, mean=0.0):
        self.num_dims=num_dims
        self.scale = scale
        self.mean = 0.0

        self.alphas_default = np.ones(num_dims)
        self.alphas_bounds = np.array([[0.1, 10.0],]*num_dims)
        self.alphas = self.alphas_default

        self.betas_default = np.ones(num_dims)
        self.betas_bounds = np.array([[0.1, 10.0],]*num_dims)
        self.betas = self.betas_default

        if alphas is not None:
            self.alphas = alphas
        if betas is not None:
            self.betas = betas

    @truncate_01
    def forward(self, X):
        X = np.copy(X)
        #check if X is a list of points
        if len(X.shape) == 1:
            X = np.array([X])
        self._X = list()
        ret = list()
        for point in X:
            self._X.append(point)
            ret.append(sps.beta.cdf(point, self.alphas, self.betas))
        return np.array(ret)

    def backward(self, V):
        raise NotImplemented()
        dx = sps.beta.pdf(self._inputs, self.alphas, self.betas)
        dx[np.logical_not(np.isfinite(dx))] = 1.0
        return dx*V

    def logprob(self):
        if np.any(self.alphas < self.alphas_bounds[0,0]) \
           or np.any(self.alphas > self.alphas_bounds[0,1]) \
           or np.any(self.betas < self.betas_bounds[0,0]) \
           or np.any(self.betas > self.betas_bounds[0,1]):
            return -np.inf

        return np.sum(sps.lognorm.logpdf([self.alphas, self.betas], self.scale, loc=self.mean))



class GaussianProcessRegressor_BW(GaussianProcessRegressor):
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GaussianProcessRegressor_BW, self).__init__(kernel=kernel,
                                                          alpha=alpha,
                                                          optimizer=optimizer,
                                                          n_restarts_optimizer=n_restarts_optimizer,
                                                          normalize_y=normalize_y,
                                                          copy_X_train=copy_X_train,
                                                          random_state=random_state)
        self.space_warper = None  # Wait to see the dimension


    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        # DIFFERENCE, CREATE THE WARPER
        if self.space_warper is None:
            self.space_warper = BetaWarp(X.shape[1])

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            #DIFFERENCE: WE NOW OPTIMIZE A GENERIC hyperparameters VARIABLE, WHERE WE STORE ALPHAS BETAS AND THETAS
            num_thetas = len(self.kernel_.theta)
            num_alphas = len(self.space_warper.alphas)
            num_betas = len(self.space_warper.betas)
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(hyperparameters, eval_gradient=True):
                theta = hyperparameters[0:num_thetas]
                alphas = hyperparameters[num_thetas:num_thetas+num_alphas]
                betas =  hyperparameters[num_thetas+num_alphas:num_thetas+num_alphas+num_betas]
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(theta, alphas, betas, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, alphas, betas)

            #DIFFERENCE: THE INITIAL POINT AND BOUNDS ALSO CONTAIN ALHPAS AND BETAS
            # First optimize starting from theta specified in kernel
            init_point = []
            init_point.extend(self.kernel_.theta)
            init_point.extend(self.space_warper.alphas_default)
            init_point.extend(self.space_warper.betas_default)
            init_point = np.array(init_point)
            bounds = []
            bounds.extend(self.kernel_.bounds)
            bounds.extend(self.space_warper.alphas_bounds)
            bounds.extend(self.space_warper.betas_bounds)
            bounds = np.array(bounds)

            optima = [(self._constrained_optimization(obj_func, init_point,
                                                      bounds))]


            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                for iteration in range(self.n_restarts_optimizer):
                    hp_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(obj_func,
                                                                 hp_initial,
                                                                 bounds))

            # DIFFERENCE: WE NEED TO UNPACK THE HYPERPARAMETERS
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            hp_optimal = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            self.kernel_.theta = hp_optimal[0:num_thetas]
            self.space_warper.alphas = hp_optimal[num_thetas:num_thetas+num_alphas]
            self.space_warper.betas =  hp_optimal[num_thetas+num_alphas:]

            # DIFFERENCE
            # We have found the optimal warping, we can now warp the observed values
            self.X_train_ = self.space_warper.forward(self.X_train_)

        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             self.space_warper.alphas,
                                             self.space_warper.betas)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def predict(self, X, return_std=False, return_cov=False):
        # DIFFERENCE: in order to predict a point, we need to warp it with the warper we have used to warp the training
        # samples
        X = self.space_warper.forward(X)
        return super(GaussianProcessRegressor_BW, self).predict(X=X, return_std=return_std, return_cov=return_cov)


    # DIFFERENCE: also receive alphas and betas
    def log_marginal_likelihood(self, theta=None, alphas=None, betas=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta)
        # DIFFERENCE: create a warper that we can modify so to optimize its hps
        warper = BetaWarp(self.X_train_.shape[1], alphas, betas)
        X = warper.forward(self.X_train_)

        if eval_gradient:
            K, K_gradient = kernel(X, eval_gradient=True)
        else:
            K = kernel(X)

        # DIFFERENCE: we need to extend K_gradient with the derivatives for alphas and
        # betas.
        if eval_gradient:
            numalphas = alphas.shape[0]
            def f(alphasbetas):
                m_alphas = alphasbetas[0:numalphas]
                m_betas = alphasbetas[numalphas:]
                return self.log_marginal_likelihood(theta, m_alphas, m_betas,
                                                    eval_gradient=None)

            W_gradient = _approx_fprime(np.concatenate([alphas, betas]),
                                        [X.shape[0], X.shape[0]], f, 1e-10)

            K_gradient = np.concatenate([K_gradient, W_gradient], axis=2)


        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        # DIFFERENCE: add the likelihood of alphas and betas
        log_likelihood += warper.logprob()

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)


        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

