from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal

from skopt.parameter import sample_points
from skopt.parameter import Uniform
from skopt.parameter import Integer
from skopt.parameter import Categorical


def check_distribution(Dist, vals, random_val):
    x = Dist(*vals)
    assert_equal(x.rvs(random_state=1), random_val)


def test_distributions():
    yield (check_distribution, Uniform, (1., 4.), 2.668088018810296)
    yield (check_distribution, Uniform, (1, 4), 2.668088018810296)
    yield (check_distribution, Integer, (1, 4), 2)
    yield (check_distribution, Integer, (1., 4.), 2)
    yield (check_distribution, Categorical, ('a', 'b', 'c', 'd'), 'b')
    yield (check_distribution, Categorical, (1., 2., 3., 4.), 2.)


def test_simple_grid():
    expected = [(2, 1), (1, 2), (2, 2), (2, 1), (1, 2)]

    for i,p in enumerate(sample_points([(1, 3), (1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_categorical():
    expected = [(2, 1), (1, 2), (2, 2), (2, 1), (1, 2)]

    for i,p in enumerate(sample_points([Categorical(1, 2), (1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_integer():
    expected = [(2, 1), (1, 2), (2, 2), (2, 1), (1, 2)]

    for i,p in enumerate(sample_points([Categorical(1, 2), (1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])

    for i,p in enumerate(sample_points([Categorical(1, 2), Integer(1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_uniform():
    expected =[(2, 4.988739243755474), (1, 1.0004574992693795)]

    for i,p in enumerate(sample_points([Categorical(1, 2), (1., 4.)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])

    for i,p in enumerate(sample_points([Categorical(1, 2), Uniform(1, 4)],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_sub_grids():
    expected = [('b', 4), ('b', 4), ('a', 1), ('b', 4), ('b', 4)]

    for i,p in enumerate(sample_points([(['a'], (1, 4)), (['b'], (4, 5))],
                                       len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_categorical_transform():
    categories = ['apple', 'orange', 'banana']
    cat = Categorical(*categories)

    # LabelEncoder sorts classes alphabetically -> banana == 1
    assert_array_equal(cat.transform(['apple', 'banana']), [0, 1])
    assert_array_equal(cat.inverse_transform([0, 1]), ['apple', 'banana'])


def test_uniform_transform():
    dist = Uniform(0, 10)
    values = [9.1, 2.3]

    assert_array_equal(dist.transform(values), values)
    assert_array_equal(dist.inverse_transform(values), values)
