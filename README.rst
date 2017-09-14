|Travis Status| |CircleCI Status|

Scikit-Optimize
===============

Scikit-Optimize, or ``skopt``, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. ``skopt`` aims
to be accessible and easy to use in many contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

We do not perform gradient-based optimization. For gradient-based
optimization algorithms look at
``scipy.optimize``
`here <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

.. figure:: https://github.com/scikit-optimize/scikit-optimize/blob/master/media/bo-objective.png
   :alt: Approximated objective

Approximated objective function after 50 iterations of ``gp_minimize``.
Plot made using ``skopt.plots.plot_objective``.

Important links
---------------

-  Static documentation - `Static
   documentation <https://scikit-optimize.github.io/>`__
-  Example notebooks - can be found in the
   `examples directory <https://github.com/scikit-optimize/scikit-optimize/tree/master/examples>`_.
-  Issue tracker -
   https://github.com/scikit-optimize/scikit-optimize/issues
-  Releases - https://pypi.python.org/pypi/scikit-optimize

Install
-------

The latest released version of scikit-optimize is v0.4, which you can install
with:
::

    pip install numpy # install numpy explicitly first
    pip install scikit-optimize

Getting started
---------------

Find the minimum of the noisy function ``f(x)`` over the range
``-2 < x < 2`` with ``skopt``:

.. code:: python

    import numpy as np
    from skopt import gp_minimize

    def f(x):
        return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +
                np.random.randn() * 0.1)

    res = gp_minimize(f, [(-2.0, 2.0)])


For more control over the optimization loop you can use the ``skopt.Optimizer``
class:

.. code:: python

    from skopt import Optimizer

    opt = Optimizer([(-2.0, 2.0)])

    for i in range(20):
        suggested = opt.ask()
        y = f(suggested)
        opt.tell(suggested, y)
        print('iteration:', i, suggested, y)


Read our `introduction to bayesian
optimization <https://scikit-optimize.github.io/notebooks/bayesian-optimization.html>`__
and the other
`examples <https://github.com/scikit-optimize/scikit-optimize/tree/master/examples>`__.


Development
-----------

The library is still experimental and under heavy development. Checkout
the `next
milestone <https://github.com/scikit-optimize/scikit-optimize/milestone/4>`__
for the plans for the next release or look at some `easy
issues <https://github.com/scikit-optimize/scikit-optimize/issues?q=is%3Aissue+is%3Aopen+label%3AEasy>`__
to get started contributing.

The development version can be installed through:

::

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -e.

Run all tests by executing ``pytest`` in the top level directory.

To only run the subset of tests with low run time, you can use ``pytest -m 'fast_test'``. On the other hand ``pytest -m 'slow_test'`` is also possible. To exclude all slow running tests try ``pytest -m 'not slow_test'``.

This is implemented using pytest `attributes <https://docs.pytest.org/en/latest/mark.html>`__. If a tests runs longer than 1 second, it is marked as slow, else as fast.

All contributors are welcome!

.. |Travis Status| image:: https://travis-ci.org/scikit-optimize/scikit-optimize.svg?branch=master
   :target: https://travis-ci.org/scikit-optimize/scikit-optimize
.. |CircleCI Status| image:: https://circleci.com/gh/scikit-optimize/scikit-optimize/tree/master.svg?style=shield&circle-token=:circle-token
   :target: https://circleci.com/gh/scikit-optimize/scikit-optimize
