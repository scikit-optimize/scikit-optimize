.. _api_ref:

=============
API Reference
=============

Scikit-Optimize, or skopt, is a simple and efficient library to minimize (very) expensive and noisy black-box functions. It implements several methods for sequential model-based optimization. skopt is reusable in many contexts and accessible.


:mod:`skopt`: module
====================

Base classes
------------
.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BayesSearchCV
    Optimizer
    Space

Functions
---------
.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: function.rst

    dummy_minimize
    dump
    expected_minimum
    expected_minimum_random_sampling
    forest_minimize
    gbrt_minimize
    gp_minimize
    load

.. _acquisition_ref:

:mod:`skopt.acquisition`: Acquisition
=====================================

.. automodule:: skopt.acquisition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`acquisition` section for further details.

.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: function.rst

    acquisition.gaussian_acquisition_1D
    acquisition.gaussian_ei
    acquisition.gaussian_lcb
    acquisition.gaussian_pi

.. _benchmarks_ref:

:mod:`skopt.benchmarks`: A collection of benchmark problems.
============================================================

.. automodule:: skopt.benchmarks
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`benchmarks` section for
further details.

.. currentmodule:: skopt

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

    benchmarks.bench1
    benchmarks.bench1_with_time
    benchmarks.bench2
    benchmarks.bench3
    benchmarks.bench4
    benchmarks.bench5
    benchmarks.branin
    benchmarks.hart6

.. _callbacks_ref:

:mod:`skopt.callbacks`: Callbacks
=================================

.. automodule:: skopt.callbacks
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`callbacks` section for further
details.

.. currentmodule:: skopt

.. autosummary::
    :toctree: generated
    :template: class.rst

    callbacks.CheckpointSaver
    callbacks.DeadlineStopper
    callbacks.DeltaXStopper
    callbacks.DeltaYStopper
    callbacks.EarlyStopper
    callbacks.TimerCallback
    callbacks.VerboseCallback


.. _learning_ref:

:mod:`skopt.learning`: Machine learning extensions for model-based optimization.
================================================================================

.. automodule:: skopt.learning
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`learning` section for further details.

.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    learning.ExtraTreesRegressor
    learning.GaussianProcessRegressor
    learning.GradientBoostingQuantileRegressor
    learning.RandomForestRegressor


.. _optimizer_ref:

:mod:`skopt.optimizer`: Optimizer
=================================

.. automodule:: skopt.optimizer
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`optimizer` section for further details.

.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    optimizer.Optimizer

.. autosummary::
   :toctree: generated/
   :template: function.rst

    optimizer.base_minimize
    optimizer.dummy_minimize
    optimizer.forest_minimize
    optimizer.gbrt_minimize
    optimizer.gp_minimize

.. _plots_ref:

:mod:`skopt.plots`: Plotting functions.
=======================================

.. automodule:: skopt.plots
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`plots` section for further details.


.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plots.partial_dependence
    plots.plot_convergence
    plots.plot_evaluations
    plots.plot_objective
    plots.plot_regret

.. _utils_ref:

:mod:`skopt.utils`: Utils functions.
====================================

.. automodule:: skopt.utils
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`utils` section for further details.


.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.cook_estimator
    utils.dimensions_aslist
    utils.expected_minimum
    utils.expected_minimum_random_sampling
    utils.dump
    utils.load
    utils.point_asdict
    utils.point_aslist
    utils.use_named_args


.. _space_ref:

:mod:`skopt.space.space`: Space
===============================

.. automodule:: skopt.space.space
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`space` section for further details.

.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    space.space.Categorical
    space.space.Dimension
    space.space.Integer
    space.space.Real
    space.space.Space

.. autosummary::
   :toctree: generated/
   :template: function.rst

    space.space.check_dimension

.. _transformers_ref:

:mod:`skopt.space.transformers`: transformers
=============================================

.. automodule:: skopt.space.transformers
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`transformers` section for further details.

.. currentmodule:: skopt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    space.transformers.CategoricalEncoder
    space.transformers.Identity
    space.transformers.LogN
    space.transformers.Normalize
    space.transformers.Pipeline
    space.transformers.Transformer


