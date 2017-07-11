"""
Scikit-Optimize, or ``skopt``, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization.
"""

from . import acquisition
from . import benchmarks
from . import callbacks
from . import learning
from . import optimizer
from . import plots
from . import space
from .optimizer import dummy_minimize
from .optimizer import forest_minimize
from .optimizer import gbrt_minimize
from .optimizer import gp_minimize
from .optimizer import Optimizer
from .searchcv import BayesSearchCV
from .space import Space
from .utils import dump
from .utils import expected_minimum
from .utils import load

__version__ = "0.4-dev"


__all__ = (
    "acquisition",
    "benchmarks",
    "callbacks",
    "learning",
    "optimizer",
    "plots",
    "space",
    "gp_minimize",
    "dummy_minimize",
    "forest_minimize",
    "gbrt_minimize",
    "Optimizer",
    "dump",
    "load",
    "expected_minimum",
    "BayesSearchCV",
    "Space"
)
