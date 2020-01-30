"""
=====================
Parallel optimization
=====================

Iaroslav Shcherbatyi, May 2017.
Reviewed by Manoj Kumar and Tim Head.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

Introduction
============

For many practical black box optimization problems expensive objective can be
evaluated in parallel at multiple points. This allows to get more objective
evaluations per unit of time, which reduces the time necessary to reach good
objective values when appropriate optimization algorithms are used, see for
example results in [1]_ and the references therein.


One such example task is a selection of number and activation function of a
neural network which results in highest accuracy for some machine learning
problem. For such task, multiple neural networks with different combinations
of number of neurons and activation function type can be evaluated at the same
time in parallel on different cpu cores / computational nodes.

The “ask and tell” API of scikit-optimize exposes functionality that allows to
obtain multiple points for evaluation in parallel. Intended usage of this
interface is as follows:

1. Initialize instance of the `Optimizer` class from skopt
2. Obtain n points for evaluation in parallel by calling the `ask` method of an optimizer instance with the `n_points` argument set to n > 0
3. Evaluate points
4. Provide points and corresponding objectives using the `tell` method of an optimizer instance
5. Continue from step 2 until eg maximum number of evaluations reached
"""

print(__doc__)
import numpy as np

#############################################################################
# Example
# =======
#
# A minimalistic example that uses joblib to parallelize evaluation of the
# objective function is given below.

from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
# example objective taken from skopt
from skopt.benchmarks import branin

optimizer = Optimizer(
    dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
    random_state=1,
    base_estimator='gp'
)

for i in range(10):
    x = optimizer.ask(n_points=4)  # x is a list of n_points points
    y = Parallel(n_jobs=4)(delayed(branin)(v) for v in x)  # evaluate points in parallel
    optimizer.tell(x, y)

# takes ~ 20 sec to get here
print(min(optimizer.yi))  # print the best objective found

#############################################################################
# Note that if `n_points` is set to some integer > 0 for the `ask` method, the
# result will be a list of points, even for `n_points` = 1. If the argument is
# set to `None` (default value) then a single point (but not a list of points)
# will be returned.
#
# The default "minimum constant liar" [1]_ parallelization strategy is used in
# the example, which allows to obtain multiple points for evaluation with a
# single call to the `ask` method with any surrogate or acquisition function.
# Parallelization strategy can be set using the "strategy" argument of `ask`.
# For supported parallelization strategies see the documentation of
# scikit-optimize.
#
# .. [1] `<https://hal.archives-ouvertes.fr/hal-00732512/document>`_