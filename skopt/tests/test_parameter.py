from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_true

from skopt.parameter import check_grid
from skopt.parameter import sample_points
from skopt.parameter import Real
from skopt.parameter import Integer
from skopt.parameter import Categorical


def check_distribution(Dist, vals, random_val):
    x = Dist(*vals)
    assert_equal(x.rvs(random_state=1), random_val)


def test_distributions():
    a = Real(1, 25)
    yield (check_distribution, Real, (1., 4.), 2.668088018810296)
    yield (check_distribution, Real, (1, 4), 2.668088018810296)
    yield (check_distribution, Integer, (1, 4), 2)
    yield (check_distribution, Integer, (1., 4.), 2)
    yield (check_distribution, Categorical, ('a', 'b', 'c', 'd'), 'b')
    yield (check_distribution, Categorical, (1., 2., 3., 4.), 2.)


def check_distribution_limits(value, lower_bound, upper_bound):
    assert_less_equal(lower_bound, value)
    assert_greater_equal(upper_bound, value)


def test_real():
    a = Real(1, 25)
    for i in range(9):
        yield (check_distribution_limits, a.rvs(random_state=i), 1, 25)
    assert_array_equal(a.rvs(random_state=0, n_samples=10).shape, (10))


# def test_categorical_transform():
#     categories = ['apple', 'orange', 'banana']
#     cat = Categorical(*categories)

#     # LabelEncoder sorts classes alphabetically -> banana == 1
#     assert_array_equal(cat.transform(['apple', 'banana']), [0, 1])
#     assert_array_equal(cat.inverse_transform([0, 1]), ['apple', 'banana'])


# def test_real_transform():
#     dist = Real(0, 10)
#     values = [9.1, 2.3]

#     assert_array_equal(dist.transform(values), values)
#     assert_array_equal(dist.inverse_transform(values), values)


# def test_simple_grid():
#     expected = [(2, 1), (1, 2), (2, 2), (2, 1), (1, 2)]

#     for i, p in enumerate(sample_points([(1, 3), (1, 4)],
#                                        len(expected), random_state=1)):
#         assert_equal(p, expected[i])


# def check_simple_grid(values, expected_rvs, dist_type):
#     grid = check_grid([values])

#     dist = grid[0][0]
#     rvs = dist.rvs(n_samples=2, random_state=1)
#     assert_true(isinstance(dist, dist_type))
#     assert_almost_equal(rvs, expected_rvs, decimal=3)


# def test_check_grid():
#     yield (check_simple_grid, (1, 4), [2, 1], Integer)
#     yield (check_simple_grid, (1., 4.), [2.668, 3.881], Real)
#     yield (check_simple_grid, (1, 2, 3), [2, 3], Categorical)


# def test_sub_grids():
#     expected = [('a', 1), ('a', 2), ('b', 5), ('b', 5), ('a', 1)]

#     for i,p in enumerate(sample_points([(['a'], (1, 4)), (['b'], (4, 6))],
#                                        len(expected), random_state=3)):
#         assert_equal(p, expected[i])



