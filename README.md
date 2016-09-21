[![Build Status](https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master)](https://travis-ci.org/scikit-optimize/scikit-optimize)
[![Build Status](https://circleci.com/gh/scikit-optimize/scikit-optimize/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-optimize/scikit-optimize)

# Scikit-Optimize

Scikit-Optimize, or `skopt`, is a simple and efficient library
for sequential model-based optimization, accessible to everybody and reusable in various
contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

We do not do gradient-based optimization. For gradient-based optimization you should be looking at [`scipy.optimize`](http://docs.scipy.org/doc/scipy/reference/optimize.html)

![Approximated objective](https://github.com/scikit-optimize/scikit-optimize/blob/master/media/bo-objective.png)

_Approximated objective function after 50 iterations of `gp_minimize`. Plot made using `skopt.plots.plot_objective`._

## Important links

- Static documentation - [Static documentation](https://scikit-optimize.github.io/)
- Example notebooks - can be found under the [`examples/`](https://github.com/scikit-optimize/scikit-optimize/tree/master/examples) directory.
- Issue tracker - https://github.com/scikit-optimize/scikit-optimize/issues
- Releases - https://pypi.python.org/pypi/scikit-optimize


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

The library is still experimental and under heavy development. Checkout the
[ROADMAP](https://github.com/scikit-optimize/scikit-optimize/issues/202) for
the next release or look at some [easy issues](https://github.com/scikit-optimize/scikit-optimize/issues?q=is%3Aissue+is%3Aopen+label%3AEasy)
to get started contributing.

The development version can be installed through:
```
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -r requirements.txt
python setup.py develop
```

Run the tests by executing `nosetests` in the top level directory.

Contributors are welcome!
