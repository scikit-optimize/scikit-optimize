==========
Quickstart
==========

Find the minimum of the noisy function ``f(x)`` over the range ``-2 < x < 2``
with ``skopt``::

  import numpy as np
  from skopt import gp_minimize

  def f(x):
      return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
              np.random.randn() * 0.1)

  res = gp_minimize(f, [(-2.0, 2.0)])

For more control over the optimization loop you can use the ``skopt.Optimizer``
class::

  from skopt import Optimizer

  opt = Optimizer([(-2.0, 2.0)])

  for i in range(20):
      suggested = opt.ask()
      y = f(suggested)
      opt.tell(suggested, y)
      print('iteration:', i, suggested, y)

For more read our `introduction to bayesian optimization`_ and the other
`examples`_.



.. _introduction to bayesian optimization: https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
.. _examples: https://github.com/scikit-optimize/scikit-optimize/tree/master/examples
