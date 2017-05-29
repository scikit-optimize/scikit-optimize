"""This script contains set of functions that test parallel optimization with
skopt, where constant liar parallelization strategy is used.
"""

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from skopt.space import Real
from skopt.learning import ExtraTreesRegressor
from skopt import Optimizer
from skopt.benchmarks import branin
from scipy.spatial.distance import pdist


def test_constant_liar_runs():
    optimizer = Optimizer(
        base_estimator=ExtraTreesRegressor(),
        dimensions=[Real(-5.0, 10.0),Real(0.0, 15.0)],
        acq_optimizer='sampling'
    )

    n_points, n_steps, Y = 2, 13, []

    for i in range(n_steps):
        strategy = optimizer.par_strats[i % len(optimizer.par_strats)]
        x = optimizer.ask(n_points, strategy)
        optimizer.tell(x, [branin(v) for v in x])

def test_all_points_different():
    # tests whether the parallel optimizer always generates different points to evaluate

    optimizer = Optimizer(
        base_estimator=ExtraTreesRegressor(),
        dimensions=[Real(-5.0, 10.0),Real(0.0, 15.0)],
        acq_optimizer='sampling'
    )

    n_points, n_steps, Y = 8, 5, []
    tolerance = 1e-3 # minimum distance at which points are considered to be the same


    for i in range(n_steps):
        strategy = optimizer.par_strats[i % len(optimizer.par_strats)]
        x = optimizer.ask(n_points, strategy)
        optimizer.tell(x, [branin(v) for v in x])
        distances = pdist(x)
        if any(distances < tolerance):
            raise ValueError("Same points were generated!")


def test_same_set_of_points_ask():

    optimizer = Optimizer(
        base_estimator=ExtraTreesRegressor(),
        dimensions=[Real(-5.0, 10.0),Real(0.0, 15.0)],
        acq_optimizer='sampling'
    )

    n_points, n_steps, Y = 8, 5, []

    for i in range(n_steps):
        strategy = optimizer.par_strats[i % len(optimizer.par_strats)]
        xa = optimizer.ask(n_points, strategy)
        xb = optimizer.ask(n_points, strategy)

        optimizer.tell(xa, [branin(v) for v in xa])

        if not (str(xa) == str(xb)):
            raise ValueError("Different sets of points were generated for two consecutive calls to ask.")
