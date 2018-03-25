===========================================
Welcome to scikit-optimize's documentation!
===========================================

Scikit-Optimize, or ``skopt``, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. ``skopt`` aims
to be accessible and easy to use in many contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

We do not perform gradient-based optimization. For gradient-based
optimization algorithms look at
``scipy.optimize``
`here <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

.. figure:: /_static/bo-objective.png
   :alt: Approximated objective

Approximated objective function after 50 iterations of ``gp_minimize``.
Plot made using ``skopt.plots.plot_objective``.

.. toctree::
    :maxdepth: 1
    :caption: Functions:

    api/minimize_functions

.. toctree::
    :maxdepth: 1
    :caption: Classes:

    api/classes

.. toctree::
    :maxdepth: 1
    :caption: Sub-modules:

    api/sub_modules

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    intro
    api/index

.. toctree::
    :maxdepth: 1
    :caption: Example notebooks:

    Ask and tell <examples/ask-and-tell.ipynb>
    Bayesian optimization <examples/bayesian-optimization.ipynb>
    Hyperparameter optimization <examples/hyperparameter-optimization.ipynb>
    Parallel optimization <examples/parallel-optimization.ipynb>
    Sklearn gridsearchcv replacement <examples/sklearn-gridsearchcv-replacement.ipynb>
    Store and load results <examples/store-and-load-results.ipynb>
    Strategy comparison <examples/strategy-comparison.ipynb>
    Visualizing results <examples/visualizing-results.ipynb>
    

Indices and tables
==================

* :ref:`genindex`

* :ref:`search`