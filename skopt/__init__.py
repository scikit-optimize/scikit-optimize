"""
Scikit-Optimize, or `skopt`, is meant to be a simple and efficient library
for black box optimization, accessible to everybody and reusable in various
contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

[![Build Status](https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master)](https://travis-ci.org/scikit-optimize/scikit-optimize)

## Development

The library is still experimental and under heavy development.

The development version can be installed through:

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -r requirements.txt
    python setup.py develop

Run the tests by executing `nosetests` in the top level directory.
"""

from . import benchmarks
from . import learning
from . import space

from .gp_opt import gp_minimize
from .dummy_opt import dummy_minimize
from .gbrt_opt import gbrt_minimize

__version__ = "0.0"


__all__ = ("benchmarks",
           "learning",
           "space",
           "gp_minimize",
           "dummy_minimize",
           "gbrt_minimize")
