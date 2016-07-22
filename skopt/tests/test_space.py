import numbers
import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true

from skopt.space import Space
from skopt.space import Real
from skopt.space import Integer
from skopt.space import Categorical


def check_dimension(Dimension, vals, random_val):
    x = Dimension(*vals)
    assert_equal(x.rvs(random_state=1), random_val)


def check_categorical(vals, random_val):
    x = Categorical(vals)
    assert_equal(x.rvs(random_state=1), random_val)


def test_dimensions():
    yield (check_dimension, Real, (1., 4.), 2.251066014107722)
    yield (check_dimension, Real, (1, 4), 2.251066014107722)
    yield (check_dimension, Integer, (1, 4), 2)
    yield (check_dimension, Integer, (1., 4.), 2)
    yield (check_categorical, ('a', 'b', 'c', 'd'), 'b')
    yield (check_categorical, (1., 2., 3., 4.), 2.)


def check_limits(value, lower_bound, upper_bound):
    assert_less_equal(lower_bound, value)
    assert_greater(upper_bound, value)


def test_real():
    a = Real(1, 25)
    for i in range(50):
        yield (check_limits, a.rvs(random_state=i), 1, 25)
    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)

    log_uniform = Real(10**-5, 10**5, prior="log-uniform")
    for i in range(50):
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
    for i in range(50):
        yield (check_limits, a.rvs(random_state=i), 1, 11)
    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)


def test_categorical_transform():
    categories = ["apple", "orange", "banana", None]
    cat = Categorical(categories)

    apple = [0., 1.0, 0.0, 0.0]
    banana = [0., 0., 1., 0.]
    orange = [0., 0., 0., 1.]
    none = [1., 0., 0., 0.]

    assert_equal(cat.transformed_size, 4)
    assert_equal(cat.transformed_size, cat.transform(["apple"]).size)
    assert_array_equal(cat.transform(categories), [apple, orange, banana, none])
    assert_array_equal(cat.transform(["apple", "orange"]), [apple, orange])
    assert_array_equal(cat.transform(["apple", "banana"]), [apple, banana])
    assert_array_equal(cat.inverse_transform([apple, orange]),
                       ["apple", "orange"])
    assert_array_equal(cat.inverse_transform([apple, banana]),
                       ["apple", "banana"])
    ent_inverse = cat.inverse_transform([apple, orange, banana, none])
    assert_array_equal(ent_inverse, categories)


def test_categorical_transform_binary():
    categories = ["apple", "orange"]
    cat = Categorical(categories)

    apple = [0.]
    orange = [1.]

    assert_equal(cat.transformed_size, 1)
    assert_equal(cat.transformed_size, cat.transform(["apple"]).size)
    assert_array_equal(cat.transform(categories), [apple, orange])
    assert_array_equal(cat.transform(["apple", "orange"]), [apple, orange])
    assert_array_equal(cat.inverse_transform([apple, orange]),
                       ["apple", "orange"])
    ent_inverse = cat.inverse_transform([apple, orange])
    assert_array_equal(ent_inverse, categories)


def test_space_consistency():
    # Reals (uniform)
    s1 = Space([Real(0.0, 1.0)]).rvs(n_samples=10, random_state=0)
    s2 = Space([Real(0.0, 1.0)]).rvs(n_samples=10, random_state=0)
    s3 = Space([Real(0, 1)]).rvs(n_samples=10, random_state=0)
    s4 = Space([(0.0, 1.0)]).rvs(n_samples=10, random_state=0)
    s5 = Space([(0.0, 1.0, "uniform")]).rvs(n_samples=10, random_state=0)
    assert_array_equal(s1, s2)
    assert_array_equal(s1, s3)
    assert_array_equal(s1, s4)
    assert_array_equal(s1, s5)

    # Reals (log-uniform)
    s1 = Space([Real(10**-3.0,
                     10**3.0,
                     prior="log-uniform")]).rvs(n_samples=10, random_state=0)
    s2 = Space([Real(10**-3.0,
                     10**3.0,
                     prior="log-uniform")]).rvs(n_samples=10, random_state=0)
    s3 = Space([Real(10**-3,
                     10**3,
                     prior="log-uniform")]).rvs(n_samples=10, random_state=0)
    s4 = Space([(10**-3.0, 10**3.0, "log-uniform")]).rvs(n_samples=10,
                                                         random_state=0)
    assert_array_equal(s1, s2)
    assert_array_equal(s1, s3)
    assert_array_equal(s1, s4)

    # Integers
    s1 = Space([Integer(1, 5)]).rvs(n_samples=10, random_state=0)
    s2 = Space([Integer(1.0, 5.0)]).rvs(n_samples=10, random_state=0)
    s3 = Space([(1, 5)]).rvs(n_samples=10, random_state=0)
    assert_array_equal(s1, s2)
    assert_array_equal(s1, s3)

    # Categoricals
    s1 = Space([Categorical(["a", "b", "c"])]).rvs(n_samples=10, random_state=0)
    s2 = Space([Categorical(["a", "b", "c"])]).rvs(n_samples=10, random_state=0)
    assert_array_equal(s1, s2)


def test_space_api():
    space = Space([(0.0, 1.0), (-5, 5),
                   ("a", "b", "c"), (1.0, 5.0, "log-uniform")])

    assert_equal(len(space.dimensions), 4)
    assert_true(isinstance(space.dimensions[0], Real))
    assert_true(isinstance(space.dimensions[1], Integer))
    assert_true(isinstance(space.dimensions[2], Categorical))
    assert_true(isinstance(space.dimensions[3], Real))

    samples = space.rvs(n_samples=10, random_state=0)
    assert_equal(len(samples), 10)
    assert_equal(len(samples[0]), 4)

    assert_true(isinstance(samples, list))
    for n in range(4):
        assert_true(isinstance(samples[n], list))

    assert_true(isinstance(samples[0][0], numbers.Real))
    assert_true(isinstance(samples[0][1], numbers.Integral))
    assert_true(isinstance(samples[0][2], str))
    assert_true(isinstance(samples[0][3], numbers.Real))

    samples_transformed = space.transform(samples)
    assert_equal(samples_transformed.shape[0], len(samples))
    assert_equal(samples_transformed.shape[1], 1 + 1 + 3 + 1)
    assert_array_equal(samples, space.inverse_transform(samples_transformed))

    samples = space.inverse_transform(samples_transformed)
    assert_true(isinstance(samples[0][0], numbers.Real))
    assert_true(isinstance(samples[0][1], numbers.Integral))
    assert_true(isinstance(samples[0][2], str))
    assert_true(isinstance(samples[0][3], numbers.Real))

    for b1, b2 in zip(space.bounds,
                      [(0.0, 1.0), (-5, 5),
                       np.asarray(["a", "b", "c"]), (1.0, 5.0)]):
        assert_array_equal(b1, b2)

    for b1, b2 in zip(space.transformed_bounds,
                      [(0.0, 1.0), (-5, 5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                       (np.log10(1.0), np.log10(5.0))]):
        assert_array_equal(b1, b2)
