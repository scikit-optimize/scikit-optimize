from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingRegressorWithStd(BaseEstimator, RegressorMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        """Fit regressor"""
        self.regressor_ = GradientBoostingRegressor(loss='quantile')
        self.rgr_up_ = GradientBoostingRegressor(loss='quantile',
                                                 alpha=0.5 + self.alpha/2.)
        self.rgr_down_ = GradientBoostingRegressor(loss='quantile',
                                                   alpha=0.5 - self.alpha/2.)

        self.regressor_.fit(X, y)
        self.rgr_up_.fit(X, y)
        self.rgr_down_.fit(X, y)

    def predict(self, X, return_std=False):
        """Prediction with uncertainties"""
        up = self.rgr_up_.predict(X)
        down = self.rgr_down_.predict(X)
        std = up - down
        central = self.regressor_.predict(X)

        return (central, std)
