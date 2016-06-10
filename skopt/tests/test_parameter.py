import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true

from skopt.parameter import _check_grid
from skopt.parameter import sample_points
from skopt.parameter import Real
from skopt.parameter import Integer
from skopt.parameter import Categorical


def check_distribution(Dist, vals, random_val):
    x = Dist(*vals)
    assert_equal(x.rvs(random_state=1), random_val)


def test_distributions():
    yield (check_distribution, Real, (1., 4.), 2.251066014107722)
    yield (check_distribution, Real, (1, 4), 2.251066014107722)
    yield (check_distribution, Integer, (1, 4), 2)
    yield (check_distribution, Integer, (1., 4.), 2)
    yield (check_distribution, Categorical, ('a', 'b', 'c', 'd'), 'b')
    yield (check_distribution, Categorical, (1., 2., 3., 4.), 2.)


def check_limits(value, lower_bound, upper_bound):
    assert_less_equal(lower_bound, value)
    assert_greater(upper_bound, value)


def test_real():
    a = Real(1, 25)
    for i in range(10):
        yield (check_limits, a.rvs(random_state=i), 1, 25)
    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)

    log_uniform = Real(10**-5, 10**5, prior="log-uniform")
    for i in range(10):
        random_val = log_uniform.rvs(random_state=i)
        yield (check_limits, random_val, 10**-5, 10**5)
    random_values = log_uniform.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    transformed_vals = log_uniform.transform(random_values)
    assert_array_equal(transformed_vals, np.log10(random_values))
    assert_array_equal(
        log_uniform.inverse_transform(transformed_vals), random_values)


def test_integer():
    a = Integer(1, 10)
    for i in range(10):
        yield (check_limits, a.rvs(random_state=i), 1, 11)
    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)


def test_categorical_transform():
    categories = ['apple', 'orange', 'banana']
    cat = Categorical(*categories)

    apple = [1.0, 0.0, 0.0]
    orange = [0.,  0.,  1]
    banana = [0.,  1.,  0.]
    assert_array_equal(
        cat.transform(categories), apple + orange + banana)
    assert_array_equal(
        cat.transform(["apple", "orange"]), apple + orange)
    assert_array_equal(
        cat.transform(['apple', 'banana']), apple + banana)
    assert_array_equal(cat.inverse_transform(apple + orange),
                       ['apple', 'orange'])
    assert_array_equal(cat.inverse_transform(apple + banana),
                       ['apple', 'banana'])
    ent_inverse = cat.inverse_transform(apple + orange + banana)
    assert_array_equal(ent_inverse, categories)


def test_simple_grid():
    expected = [(2, 4), (1, 1), (2, 4), (2, 4), (1, 1)]

    for i, p in enumerate(sample_points([(1, 3), (1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def check_simple_grid(values, expected_rvs, dist_type):
    grid = _check_grid([values])

    dist = grid[0][0]
    rvs = dist.rvs(n_samples=2, random_state=1)
    assert_true(isinstance(dist, dist_type))
    assert_almost_equal(rvs, expected_rvs, decimal=3)


def test_check_grid():
    yield (check_simple_grid, (1, 4), [2, 4], Integer)
    yield (check_simple_grid, (1., 4.), [2.251,  3.161], Real)
    yield (check_simple_grid, (1, 2, 3), [2, 3], Categorical)


def test_sub_grids():
    expected = [('a', 4), ('a', 2), ('b', 5), ('b', 6), ('a', 2)]

    for i,p in enumerate(sample_points([(['a'], (1, 4)), (['b'], (4, 6))],
                                       len(expected), random_state=3)):
        assert_equal(p, expected[i])


def test_sample_grid_consistency():
    real_points_one = list(sample_points(
        [Real(0.0, 1.0)], random_state=0, n_points=10))
    real_points_two = list(sample_points(
        [Real(0.0, 1.0)], random_state=0, n_points=10))
    real_points_three = list(sample_points(
        [Real(0, 1)], random_state=0, n_points=10))
    real_points_four = list(sample_points(
        [(0.0, 1.0)], random_state=0, n_points=10))
    assert_array_equal(real_points_one, real_points_two)
    assert_array_equal(real_points_one, real_points_three)
    assert_array_equal(real_points_one, real_points_four)

    int_points_one = list(sample_points(
        [Integer(1.0, 5.0)], random_state=0, n_points=10))
    int_points_two = list(sample_points(
        [(1, 5)], random_state=0, n_points=10))
    int_points_three = list(sample_points(
        [Integer(1.0, 5.0)], random_state=0, n_points=10))
    assert_array_equal(int_points_one, int_points_two)
    assert_array_equal(int_points_three, int_points_two)

    cat_points_one = list(sample_points(
        [Categorical("a", "b", "c")], random_state=0, n_points=10))
    cat_points_two = list(sample_points(
        [("a", "b", "c")], random_state=0, n_points=10))
    assert_array_equal(cat_points_one, cat_points_two)
