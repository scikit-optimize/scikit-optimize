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
        if noise is None:
            # Nothing special
            return super(GaussianProcessRegressor, self).fit(X, y)
        # else:

            # The noise component of this kernel should be set to zero
            # while estimating K(X, X_test) and K(X_test, X_test)
            # Note that the term K(X, X) should include the noise but
            # this (K(X, X))^-1y is precomputed as the attribute `alpha_`.
            # (Notice the underscore).
            # This has been described in Eq 2.24 of
            # http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
            # Hence this hack
            # self._gp = sk_GaussianProcessRegressor()
            # params = self.get_params().copy()
            # params.pop['noise']
            # self._gp.set_params(**params)
            # self._gp.set_params(kernel=self.kernel + WhiteKernel())
            # self._gp.fit(X, y)
            # white_present, white_param = param_for_white_kernel_in_Sum(
            #     self._gp.kernel_)
            # if white_present:
            #     self._gp.kernel_.set_params(
            #         **{white_param: WhiteKernel(noise_level=0.0)})
