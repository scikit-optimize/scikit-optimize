from sklearn.gaussian_process import GaussianProcessRegressor as sk_GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Sum


def _param_for_white_kernel_in_Sum(kernel, kernel_str=""):
    """
    Check if a WhiteKernel exists in a Sum Kernel
    and if it does return the corresponding key in
    `kernel.get_params()`
    """
    if kernel_str != "":
        kernel_str = kernel_str + "__"
    if isinstance(kernel, Sum):
        for param, child in kernel.get_params(deep=False).items():
            if isinstance(child, WhiteKernel):
                return True, kernel_str + param
            else:
                present, child_str = _param_for_white_kernel_in_Sum(
                    child, kernel_str + param)
                if present:
                    return True, child_str
    return False, "_"


class GaussianProcessRegressor(sk_GaussianProcessRegressor):
    """
    GaussianProcessRegressor that allows noise tunability.

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations
        and reduce potential numerical issue during fitting. If an array is
        passed, it must have the same number of entries as the data used for
        fitting and is used as datapoint-dependent noise level. Note that this
        is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
        the noise level directly as a parameter is mainly for convenience and
        for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer: int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y: boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    noise: string, "gaussian", optional
        If set to "gaussian", then it is assumed that `y` is a noisy
        estimate of `f(x)` where the noise is gaussian.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_: array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)

    kernel_: kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_: array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    alpha_: array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_: float
        The log-marginal-likelihood of ``self.kernel_.theta``
    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None,
                 noise=None):
        self.noise = noise
        super(GaussianProcessRegressor, self).__init__(
            kernel=kernel, alpha=alpha, optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train,
            random_state=random_state)

    def fit(self, X, y):
        """Fit Gaussian process regression model

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
        if self.noise and self.noise != "gaussian":
            raise ValueError("expected noise to be 'gaussian', got %s"
                             % self.noise)

        if self.kernel is not None and self.noise:
            self.kernel = self.kernel + WhiteKernel()
        super(GaussianProcessRegressor, self).fit(X, y)

        if self.noise:
            # The noise component of this kernel should be set to zero
            # while estimating K(X, X_test) and K(X_test, X_test)
            # Note that the term K(X, X) should include the noise but
            # this (K(X, X))^-1y is precomputed as the attribute `alpha_`.
            # (Notice the underscore).
            # This has been described in Eq 2.24 of
            # http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
            # Hence this hack
            if isinstance(self.kernel_, WhiteKernel):
                self.kernel_.set_params(noise_level=0.0)
            else:
                white_present, white_param = _param_for_white_kernel_in_Sum(
                    self.kernel_)
                # This should always be true. Just in case.
                if white_present:
                    self.noise_ = self.kernel.get_params()[white_param]
                    self.kernel_.set_params(
                        **{white_param: WhiteKernel(noise_level=0.0)})
        return self
