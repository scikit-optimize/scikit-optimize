from functools import partial
from sklearn.tree import DecisionTreeClassifier
import pytest

from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4


MINIMIZERS = [("ET", partial(forest_minimize, base_estimator='ET')),
              ("RF", partial(forest_minimize, base_estimator='RF')),
              ("gbrt", gbrt_minimize)]


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", [42, DecisionTreeClassifier()])
def test_forest_minimize_api(base_estimator):
    # invalid string value
    with pytest.raises(ValueError):
        forest_minimize(lambda x: 0., [], base_estimator='abc')

    # not a string nor a regressor
    with pytest.raises(ValueError):
        forest_minimize(lambda x: 0., [], base_estimator=base_estimator)


def check_minimize(minimizer, func, y_opt, dimensions, margin,
                   n_calls, n_initial_points=10, x0=None, n_jobs=1):
    for n in range(3):
        r = minimizer(
            func, dimensions, n_calls=n_calls, random_state=n,
            n_initial_points=n_initial_points, x0=x0, n_jobs=n_jobs)
        assert r.fun < y_opt + margin


@pytest.mark.slow_test
@pytest.mark.parametrize("name, minimizer", MINIMIZERS)
def test_tree_based_minimize(name, minimizer):
    check_minimize(minimizer, bench1, 0.05,
                   [(-2.0, 2.0)], 0.05, 25, 5)

    # XXX: We supply points at the edge of the search
    # space as an initial point to the minimizer.
    # This makes sure that the RF model can find the minimum even
    # if all the randomly sampled points are one side of the
    # the minimum, since for a decision tree any point greater than
    # max(sampled_points) would give a constant value.
    X0 = [[-5.6], [-5.8], [5.8], [5.6]]
    check_minimize(minimizer, bench2, -4.7,
                   [(-6.0, 6.0)], 0.1, 20, 10, X0)
    check_minimize(minimizer, bench3, -0.4,
                   [(-2.0, 2.0)], 0.05, 10, 5)
    check_minimize(minimizer, bench4, 1.,
                   [("-2", "-1", "0", "1", "2")], 0.05, 5, 1)


@pytest.mark.slow_test
def test_tree_based_minimize_n_jobs():
    check_minimize(forest_minimize, bench1, 0.05,
                   [(-2.0, 2.0)], 0.05, 25, 5, n_jobs=2)


@pytest.mark.fast_test
def test_categorical_integer():
    def f(params):
        return 0

    dims = [[1]]
    res = forest_minimize(f, dims, n_calls=1, random_state=1,
                          n_initial_points=1)
    assert res.x_iters[0][0] == dims[0][0]
