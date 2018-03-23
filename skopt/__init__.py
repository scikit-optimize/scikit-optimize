"""
Scikit-Optimize, or `skopt`, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. `skopt` is reusable
in many contexts and accessible.

[![Build Status](https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master)](https://travis-ci.org/scikit-optimize/scikit-optimize)

## Install

```
pip install scikit-optimize
```

## Getting started

Find the minimum of the noisy function `f(x)` over the range `-2 < x < 2`
with `skopt`:

```python
import numpy as np
from skopt import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

res = gp_minimize(f, [(-2.0, 2.0)])
```

For more read our [introduction to bayesian optimization](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)
and the other [examples](https://github.com/scikit-optimize/scikit-optimize/tree/master/examples).


## Development

The library is still experimental and under heavy development.

The development version can be installed through:

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -r requirements.txt
    python setup.py develop

Run the tests by executing `pytest` in the top level directory.
"""

from . import acquisition
from . import benchmarks
from . import callbacks
from . import learning
from . import optimizer

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

__version__ = "0.5.2"


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
