from sklearn.utils.testing import assert_equal

from skopt.sample import points
from skopt.sample import Categorical
from skopt.sample import Discrete


def test_simple_grid():
    g = points({'a': [1, 3], 'b': [1, 4]}, random_state=1)
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i in range(5):
        assert_equal(next(g), expected[i])


def test_simple_categorical():
    g = points({'a': Categorical(1, 2), 'b': [1, 4]}, random_state=1)
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i in range(5):
        assert_equal(next(g), expected[i])


def test_simple_discrete():
    g = points({'a': Categorical(1, 2), 'b': Discrete(1, 4)},
                     random_state=1)
    expected =[{'b': 1, 'a': 2}, {'b': 2, 'a': 1}, {'b': 2, 'a': 2},
               {'b': 1, 'a': 2}, {'b': 2, 'a': 1}]

    for i in range(5):
        assert_equal(next(g), expected[i])


def test_sub_grids():
    g = points(
        [{'a': ['a'], 'b': [1, 4]}, {'a': ['b'], 'c': [4, 5]}], random_state=1)
    expected =[{'a': 'b', 'c': 4}, {'a': 'b', 'c': 4}, {'a': 'a', 'b': 1},
               {'a': 'b', 'c': 4}, {'a': 'b', 'c': 4}]

    for i in range(5):
        assert_equal(next(g), expected[i])
