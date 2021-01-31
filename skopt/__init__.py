"""
Scikit-Optimize, or `skopt`, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. `skopt` aims
to be accessible and easy to use in many contexts.
"""
import platform
import struct
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("scikit-optimize")
except PackageNotFoundError:
    __version__ = '?.?.?'  # Not installed


from . import acquisition
from . import benchmarks
from . import callbacks
from . import learning
from . import optimizer
from . import sampler
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
from .utils import expected_minimum_random_sampling
from .utils import load

__all__ = (
    "acquisition",
    "benchmarks",
    "callbacks",
    "learning",
    "optimizer",
    "plots",
    "sampler",
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
IS_PYPY = platform.python_implementation() == 'PyPy'
_IS_32BIT = 8 * struct.calcsize("P") == 32
