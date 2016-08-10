"""
Scikit-Optimize, or `skopt`, is a simple and efficient library
for sequential model-based optimization, accessible to everybody
and reusable in various contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

[![Build Status](https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master)](https://travis-ci.org/scikit-optimize/scikit-optimize)

## Install

These instructions will setup the latest released version of `scikit-optimize`.
Currently `scikit-optimize` relies on a yet unreleased version of `scikit-learn`.
This means you will have to install that version by hand and probably want to
create a separate virtualenv or conda environment for it.

```
pip install -e git+https://github.com/scikit-learn/scikit-learn.git#egg=scikit-learn-0.18dev
```

After this you can install `scikit-optimize` with:
```
pip install scikit-optimize
```

## Development

The library is still experimental and under heavy development.

The development version can be installed through:

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -r requirements.txt
    python setup.py develop

Run the tests by executing `nosetests` in the top level directory.
"""

from . import acquisition
from . import benchmarks
from . import learning
from . import plots
from . import space

from .gp_opt import gp_minimize
from .dummy_opt import dummy_minimize
from .forest_opt import forest_minimize
from .forest_opt import gbrt_minimize


__version__ = "0.1"


__all__ = ("acquisition",
           "benchmarks",
           "learning",
           "plots",
           "space",
           "gp_minimize",
           "dummy_minimize",
           "forest_minimize",
           "gbrt_minimize")
