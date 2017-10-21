"""Machine learning extensions for model-based optimization."""

from skgarden import RandomForestRegressor
from skgarden import ExtraTreesRegressor

from .gbrt import GradientBoostingQuantileRegressor
from .gaussian_process import GaussianProcessRegressor, GaussianProcessRegressor_BetaWarped


__all__ = ("RandomForestRegressor",
           "ExtraTreesRegressor",
           "GradientBoostingQuantileRegressor",
           "GaussianProcessRegressor",
           "GaussianProcessRegressor_BetaWarped")
