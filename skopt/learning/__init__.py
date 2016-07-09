"""Machine learning extensions for black box optimisation."""

from .gbrt import GradientBoostingQuantileRegressor
from .tree import DecisionTreeRegressor

__all__ = ("GradientBoostingQuantileRegressor", "DecisionTreeRegressor")
