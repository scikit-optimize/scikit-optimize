"""Machine learning extensions for model-based optimization."""

from .forest import RandomForestRegressor
from .forest import ExtraTreesRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .gaussian_process import GaussianProcessRegressor


__all__ = ("RandomForestRegressor",
           "ExtraTreesRegressor",
           "GradientBoostingQuantileRegressor",
           "GaussianProcessRegressor")
