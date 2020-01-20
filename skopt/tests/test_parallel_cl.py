"""This script contains set of functions that test parallel optimization with
skopt, where constant liar parallelization strategy is used.
"""


from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skopt.space import Real
from skopt import Optimizer
from skopt.benchmarks import branin
import skopt.learning as sol

from scipy.spatial.distance import pdist
import pytest

# list of all strategies for parallelization
supported_strategies = ["cl_min", "cl_mean", "cl_max"]

# test one acq function that incorporates the runtime, and one that does not
supported_acq_functions = ["EI", "EIps"]

# Extract available surrogates, so that new ones are used automatically
available_surrogates = [
    getattr(sol, name) for name in sol.__all__
    if "GradientBoostingQuantileRegressor" not in name
]  # excluding the GradientBoostingQuantileRegressor, will open issue later

n_steps = 5  # number of steps to test the algorithms with
n_points = 4  # number of points to evaluate at a single step
# n_steps x n_points > n_random_restarts should hold


@pytest.mark.parametrize("strategy", supported_strategies)
@pytest.mark.parametrize("surrogate", available_surrogates)
@pytest.mark.parametrize("acq_func", supported_acq_functions)
def test_constant_liar_runs(strategy, surrogate, acq_func):
    """
    Tests whether the optimizer runs properly during the random
    initialization phase and beyond

    Parameters
    ----------
    * `strategy` [string]:
        Name of the strategy to use during optimization.

    * `surrogate` [scikit-optimize surrogate class]:
        A class of the scikit-optimize surrogate used in Optimizer.
    """
    optimizer = Optimizer(
        base_estimator=surrogate(),
        dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
        acq_func=acq_func,
        acq_optimizer='sampling',
        random_state=0
    )

    # test arguments check
    assert_raises(ValueError, optimizer.ask, {"strategy": "cl_maen"})
    assert_raises(ValueError, optimizer.ask, {"n_points": "0"})
    assert_raises(ValueError, optimizer.ask, {"n_points": 0})

    for i in range(n_steps):
        x = optimizer.ask(n_points=n_points, strategy=strategy)
        # check if actually n_points was generated
        assert_equal(len(x), n_points)

        if "ps" in acq_func:
            optimizer.tell(x, [[branin(v), 1.1] for v in x])
        else:
            optimizer.tell(x, [branin(v) for v in x])


@pytest.mark.parametrize("strategy", supported_strategies)
@pytest.mark.parametrize("surrogate", available_surrogates)
def test_all_points_different(strategy, surrogate):
    """
    Tests whether the parallel optimizer always generates
    different points to evaluate.

    Parameters
    ----------
    * `strategy` [string]:
        Name of the strategy to use during optimization.

    * `surrogate` [scikit-optimize surrogate class]:
        A class of the scikit-optimize surrogate used in Optimizer.
    """
    optimizer = Optimizer(
        base_estimator=surrogate(),
        dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
        acq_optimizer='sampling',
        random_state=1
    )

    tolerance = 1e-3  # distance above which points are assumed same
    for i in range(n_steps):
        x = optimizer.ask(n_points, strategy)
        optimizer.tell(x, [branin(v) for v in x])
        distances = pdist(x)
        assert all(distances > tolerance)


@pytest.mark.parametrize("strategy", supported_strategies)
@pytest.mark.parametrize("surrogate", available_surrogates)
def test_same_set_of_points_ask(strategy, surrogate):
    """
    For n_points not None, tests whether two consecutive calls to ask
    return the same sets of points.

    Parameters
    ----------
    * `strategy` [string]:
        Name of the strategy to use during optimization.

    * `surrogate` [scikit-optimize surrogate class]:
        A class of the scikit-optimize surrogate used in Optimizer.
    """

    optimizer = Optimizer(
        base_estimator=surrogate(),
        dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
        acq_optimizer='sampling',
        random_state=2
    )

    for i in range(n_steps):
        xa = optimizer.ask(n_points, strategy)
        xb = optimizer.ask(n_points, strategy)
        optimizer.tell(xa, [branin(v) for v in xa])
        assert_equal(xa, xb)  # check if the sets of points generated are equal


@pytest.mark.parametrize("strategy", supported_strategies)
@pytest.mark.parametrize("surrogate", available_surrogates)
def test_reproducible_runs(strategy, surrogate):
    # two runs of the optimizer should yield exactly the same results

    optimizer = Optimizer(
        base_estimator=surrogate(random_state=1),
        dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
        acq_optimizer='sampling',
        random_state=1
    )

    points = []
    for i in range(n_steps):
        x = optimizer.ask(n_points, strategy)
        points.append(x)
        optimizer.tell(x, [branin(v) for v in x])

    # the x's should be exaclty as they are in `points`
    optimizer = Optimizer(
        base_estimator=surrogate(random_state=1),
        dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
        acq_optimizer='sampling',
        random_state=1
    )
    for i in range(n_steps):
        x = optimizer.ask(n_points, strategy)

        assert points[i] == x

        optimizer.tell(x, [branin(v) for v in x])
