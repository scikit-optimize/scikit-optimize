# Scikit-Optimize

Scikit-Optimize, or `skopt`, is a simple and efficient library
for sequential model-based optimization, accessible to everybody and reusable in various
contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

[![Build Status](https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master)](https://travis-ci.org/scikit-optimize/scikit-optimize)

![Approximated objective](https://github.com/scikit-optimize/scikit-optimize/blob/master/media/bo-objective.png)

_Approximated objective function after 50 iterations of `gp_minimize`. Plot made using `skopt.plots.plot_objective`._

## Documentation

- [Static documentation](https://scikit-optimize.github.io/)
- Examples can be found under the [`examples/`](https://github.com/scikit-optimize/scikit-optimize/tree/master/examples) directory.

## Development

The library is still experimental and under heavy development.

The development version can be installed through:
```
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -r requirements.txt
python setup.py develop
```

Run the tests by executing `nosetests` in the top level directory.

Contributors are welcome!
