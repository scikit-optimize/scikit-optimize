from sklearn.utils.testing import assert_equal

from skopt.sample import points
from skopt.sample import Categorical
from skopt.sample import Continuous
from skopt.sample import Discrete


def check_distribution(Dist, vals, random_val):
    x = Dist(*vals)
    assert_equal(x.rvs(random_state=1), random_val)


def test_distributions():
    yield (check_distribution, Continuous, (1., 4.), 2.668088018810296)
    yield (check_distribution, Discrete, (1, 4), 2)
    yield (check_distribution, Categorical, (1, 2, 3, 4), 2)
    yield (check_distribution, Categorical, (1., 2., 3., 4.), 2.)


def test_simple_grid():
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i,p in enumerate(points({'a': [1, 3], 'b': [1, 4]},
                                len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_categorical():
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i,p in enumerate(points({'a': Categorical(1, 2), 'b': [1, 4]},
                                len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_discrete():
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i,p in enumerate(points({'a': Categorical(1, 2), 'b': Discrete(1, 4)},
                                len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_simple_continuous():
    expected =[{'a': 2, 'b': 4.988739243755474},
               {'a': 1, 'b': 1.0004574992693795}]

    for i,p in enumerate(points({'a': Categorical(1, 2), 'b': Continuous(1, 4)},
                                len(expected), random_state=1)):
        assert_equal(p, expected[i])

    for i,p in enumerate(points({'a': Categorical(1, 2), 'b': [1., 4.]},
                              len(expected), random_state=1)):
        assert_equal(p, expected[i])


def test_sub_grids():
    expected =[{'a': 'b', 'c': 4}, {'a': 'b', 'c': 4}, {'a': 'a', 'b': 1},
               {'a': 'b', 'c': 4}, {'a': 'b', 'c': 4}]

    for i,p in enumerate(points([{'a': ['a'], 'b': [1, 4]},
                                 {'a': ['b'], 'c': [4, 5]}],
                                len(expected), random_state=1)):
        assert_equal(p, expected[i])
