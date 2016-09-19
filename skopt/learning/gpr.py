from sklearn.gaussian_process import GaussianProcessRegressor as sk_GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

class GaussianProcessRegressor(sk_GaussianProcessRegressor):
    """
    GaussianProcessRegressor that allows noise tunability.
    """

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
        super(GaussianProcessRegressor, self).fit(X, y)
        for param, value in self.kernel_.get_params().items():
            if param.endswith('noise_level'):
                self.noise_ = value
                break

        # The noise component of this kernel should be set to zero
        # while estimating K(X, X_test) and K(X_test, X_test)
        # Note that the term K(X, X) should include the noise but
        # this (K(X, X))^-1y is precomputed as the attribute `alpha_`.
        # (Notice the underscore).
        # This has been described in Eq 2.24 of
        # http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
        if isinstance(self.kernel_, WhiteKernel):
            self.kernel_.set_params(noise_level=0.0)
        else:
            for param, value in self.kernel_.get_params().items():
                if isinstance(value, WhiteKernel):
                    self.kernel_.set_params(
                        **{param:WhiteKernel(noise_level=0.0)})
        return self
