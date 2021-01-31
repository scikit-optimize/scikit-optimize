
|Logo|

|pypi| |conda| |CI Status| |binder| |gitter| |Zenodo DOI|

Scikit-Optimize
===============

Scikit-Optimize, or ``skopt``, is a simple and efficient library for
optimizing (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. ``skopt`` aims
to be accessible and easy to use in many contexts.

The library is built on top of NumPy, SciPy, and Scikit-Learn.

We do not perform gradient-based optimization. For gradient-based
optimization algorithms, look at |scipy.optimize|_.

.. figure:: https://github.com/scikit-optimize/scikit-optimize/blob/master/doc/image/bo-objective.png
   :alt: Approximated objective

Approximated objective function after 50 iterations of ``gp_minimize``.
Plot made using ``skopt.plots.plot_objective``.


Important links
---------------

-  `Project website <https://scikit-optimize.github.io/>`__
-  Example notebooks - can be found in examples_.
-  `Discussion forum
   <https://github.com/scikit-optimize/scikit-optimize/discussions>`__
-  `Issue tracker
   <https://github.com/scikit-optimize/scikit-optimize/issues>`__
-  Releases - https://pypi.org/project/scikit-optimize


Install
-------

scikit-optimize requires Python >= 3.6.
You can install the latest release with:
::

    pip install scikit-optimize

This installs the essentials. To install plotting functionality,
you can instead do:
::

    pip install 'scikit-optimize[plots]'

This will additionally install Matplotlib.

If you're using Anaconda platform, there is a `conda-forge <https://conda-forge.org/>`_
package of scikit-optimize:
::

    conda install -c conda-forge scikit-optimize

Using conda-forge is probably the easiest way to install scikit-optimize on
Windows.


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
optimization <https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html>`__
and the other examples_.


Development
-----------

See `CONTRIBUTING.md <CONTRIBUTING.md>`_.


Commercial support
------------------

Feel free to `get in touch <mailto:tim@wildtreetech.com>`_ if you need commercial
support or would like to sponsor development. Resources go towards paying
for additional work by seasoned engineers and researchers.


Made possible by
----------------

The scikit-optimize project was made possible with the support of

.. image:: https://avatars1.githubusercontent.com/u/18165687?v=4&s=128
   :alt: Wild Tree Tech
   :target: https://wildtreetech.com

.. image:: https://i.imgur.com/lgxboT5.jpg
    :alt: NYU Center for Data Science
    :target: https://cds.nyu.edu/

.. image:: https://i.imgur.com/V1VSIvj.jpg
    :alt: NSF
    :target: https://www.nsf.gov

.. image:: https://i.imgur.com/3enQ6S8.jpg
    :alt: Northrop Grumman
    :target: https://www.northropgrumman.com

If your employer allows you to work on scikit-optimize during the day and would like
recognition, feel free to add them to the "Made possible by" list.


.. |pypi| image:: https://img.shields.io/pypi/v/scikit-optimize.svg
   :target: https://pypi.python.org/pypi/scikit-optimize
.. |conda| image:: https://anaconda.org/conda-forge/scikit-optimize/badges/version.svg
   :target: https://anaconda.org/conda-forge/scikit-optimize
.. |CI Status| image:: https://github.com/scikit-optimize/scikit-optimize/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/scikit-optimize/scikit-optimize/actions/workflows/ci.yml?query=branch%3Amaster
.. |Logo| image:: https://avatars2.githubusercontent.com/u/18578550?v=4&s=80
.. |binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/scikit-optimize/scikit-optimize/master?filepath=examples
.. |gitter| image:: https://badges.gitter.im/scikit-optimize/scikit-optimize.svg
   :target: https://gitter.im/scikit-optimize/Lobby
.. |Zenodo DOI| image:: https://zenodo.org/badge/54340642.svg
   :target: https://zenodo.org/badge/latestdoi/54340642
.. |scipy.optimize| replace:: ``scipy.optimize``
.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _examples: https://scikit-optimize.github.io/stable/auto_examples/index.html
