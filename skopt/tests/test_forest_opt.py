from functools import partial

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raise_message

from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4


MINIMIZERS = [("ET", partial(forest_minimize, base_estimator='ET')),
              ("RF", partial(forest_minimize, base_estimator='RF')),
              ("gbrt", gbrt_minimize)]


def test_forest_minimize_api():
    # invalid string value
    assert_raise_message(ValueError,
                         "Valid strings for the base_estimator parameter",
                         forest_minimize, lambda x: 0., [],
                         base_estimator='abc')

    for base_estimator in [42, DecisionTreeClassifier()]:
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


def test_tree_based_minimize():
    for name, minimizer in MINIMIZERS:
        yield (check_minimize, minimizer, bench1, 0.,
               [(-2.0, 2.0)], 0.05, 25, 5)

        # XXX: We supply points at the edge of the search
        # space as an initial point to the minimizer.
        # This makes sure that the RF model can find the minimum even
        # if all the randomly sampled points are one side of the
        # the minimum, since for a decision tree any point greater than
        # max(sampled_points) would give a constant value.
        X0 = [[-5.6], [-5.8], [5.8], [5.6]]
        yield (check_minimize, minimizer, bench2, -5,
               [(-6.0, 6.0)], 0.1, 100, 10, X0)
        yield (check_minimize, minimizer, bench3, -0.9,
               [(-2.0, 2.0)], 0.05, 25)
        yield (check_minimize, minimizer, bench4, 0.0,
               [("-2", "-1", "0", "1", "2")], 0.05, 10, 1)
