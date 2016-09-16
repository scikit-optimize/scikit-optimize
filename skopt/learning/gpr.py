from sklearn.gaussian_process import GaussianProcessRegressor as sk_GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel

class GaussianProcessRegressor(sk_GaussianProcessRegressor):
    """
    GaussianProcessRegressor that allows noise tunability.
    """

    def fit(self, X, y):
        super(GaussianProcessRegressor, self).fit(X, y)
        for param, value in self.kernel_.get_params().items():
            # XXX: Should return this only in the case where a
            # WhiteKernel is added.
            if param.endswith('noise_level'):
                self.noise_ = value
                break
