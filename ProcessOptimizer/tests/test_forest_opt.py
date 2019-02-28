from functools import partial

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raise_message
import pytest

from ProcessOptimizer import gbrt_minimize
from ProcessOptimizer import forest_minimize
from ProcessOptimizer.benchmarks import bench1
from ProcessOptimizer.benchmarks import bench2
from ProcessOptimizer.benchmarks import bench3
from ProcessOptimizer.benchmarks import bench4


MINIMIZERS = [("ET", partial(forest_minimize, base_estimator='ET')),
              ("RF", partial(forest_minimize, base_estimator='RF')),
              ("gbrt", gbrt_minimize)]


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", [42, DecisionTreeClassifier()])
def test_forest_minimize_api(base_estimator):
    # invalid string value
    assert_raise_message(ValueError,
                         "Valid strings for the base_estimator parameter",
                         forest_minimize, lambda x: 0., [],
                         base_estimator='abc')

    # not a string nor a regressor
    assert_raise_message(ValueError,
                         "has to be a regressor",
                         forest_minimize, lambda x: 0., [],
                         base_estimator=base_estimator)


def check_minimize(minimizer, func, y_opt, dimensions, margin,
                   n_calls, n_random_starts=10, x0=None):
    for n in range(3):
        r = minimizer(
            func, dimensions, n_calls=n_calls, random_state=n,
            n_random_starts=n_random_starts, x0=x0)
        assert_less(r.fun, y_opt + margin)


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
