.. currentmodule:: skopt
.. _minimize-functions:

``skopt``'s top level minimization functions
============================================

These are easy to get started with. They mirror the ``scipy.optimize``
API and provide a high level interface to various pre-configured
optimizers.

:class:`dummy_minimize`
-----------------------
Random search by uniform sampling within the given bounds.

:class:`forest_minimize`
------------------------
Sequential optimisation using decision trees.

A tree based regression model is used to model the expensive to evaluate
function `func`. The model is improved by sequentially evaluating
the expensive function at the next best point. Thereby finding the
minimum of `func` with as few evaluations as possible.

:class:`gbrt_minimize`
----------------------
Sequential optimization using gradient boosted trees.

Gradient boosted regression trees are used to model the (very)
expensive to evaluate function `func`. The model is improved
by sequentially evaluating the expensive function at the next
best point. Thereby finding the minimum of `func` with as
few evaluations as possible.

:class:`gp_minimize`
--------------------
Bayesian optimization using Gaussian Processes.

If every function evaluation is expensive, for instance
when the parameters are the hyperparameters of a neural network
and the function evaluation is the mean cross-validation score across
ten folds, optimizing the hyperparameters by standard optimization
routines would take for ever!

The idea is to approximate the function using a Gaussian process.
In other words the function values are assumed to follow a multivariate
gaussian. The covariance of the function values are given by a
GP kernel between the parameters. Then a smart choice to choose the
next parameter to evaluate can be made by the acquisition function
over the Gaussian prior which is much quicker to evaluate.
