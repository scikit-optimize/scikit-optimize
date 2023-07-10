import pytest
import tempfile

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
import numpy as np

from skopt import gp_minimize, forest_minimize
from skopt import load
from skopt import dump
from skopt import expected_minimum, expected_minimum_random_sampling
from skopt.benchmarks import bench1
from skopt.benchmarks import bench3
from skopt.learning import ExtraTreesRegressor
from skopt import Optimizer
from skopt import Space
from skopt.space import Dimension
from skopt.utils import point_asdict
from skopt.utils import point_aslist
from skopt.utils import dimensions_aslist
from skopt.utils import has_gradients
from skopt.utils import cook_estimator
from skopt.utils import normalize_dimensions
from skopt.utils import use_named_args
from skopt.utils import check_list_types
from skopt.utils import check_dimension_names
from skopt.space import Real, Integer, Categorical


def check_optimization_results_equality(res_1, res_2):
    # Check if the results objects have the same keys
    assert_equal(sorted(res_1.keys()), sorted(res_2.keys()))
    # Shallow check of the main optimization results
    assert_array_equal(res_1.x, res_2.x)
    assert_array_equal(res_1.x_iters, res_2.x_iters)
    assert_array_equal(res_1.fun, res_2.fun)
    assert_array_equal(res_1.func_vals, res_2.func_vals)


@pytest.mark.fast_test
def test_dump_and_load():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      acq_func="LCB",
                      n_calls=2,
                      n_random_starts=1,
                      random_state=1)

    # Test normal dumping and loading
    with tempfile.TemporaryFile() as f:
        dump(res, f)
        f.seek(0)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert "func" in res_loaded.specs["args"]

    # Test dumping without objective function
    with tempfile.TemporaryFile() as f:
        dump(res, f, store_objective=False)
        f.seek(0)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert not ("func" in res_loaded.specs["args"])

    # Delete the objective function and dump the modified object
    del res.specs["args"]["func"]
    with tempfile.TemporaryFile() as f:
        dump(res, f, store_objective=False)
        f.seek(0)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert not ("func" in res_loaded.specs["args"])


@pytest.mark.fast_test
def test_dump_and_load_optimizer():
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer([(-2.0, 2.0)], base_estimator, n_initial_points=1,
                    acq_optimizer="sampling")

    opt.run(bench1, n_iter=3)

    with tempfile.TemporaryFile() as f:
        dump(opt, f)
        f.seek(0)
        load(f)


@pytest.mark.fast_test
def test_expected_minimum():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      noise=1e-8,
                      n_calls=8,
                      n_random_starts=3,
                      random_state=1)

    x_min, f_min = expected_minimum(res, random_state=1)
    x_min2, f_min2 = expected_minimum(res, random_state=1)

    assert f_min <= res.fun  # true since noise ~= 0.0
    assert x_min == x_min2
    assert f_min == f_min2


@pytest.mark.fast_test
def test_expected_minimum_random_sampling():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      noise=1e-8,
                      n_calls=8,
                      n_random_starts=3,
                      random_state=1)

    x_min, f_min = expected_minimum_random_sampling(res, random_state=1)
    x_min2, f_min2 = expected_minimum_random_sampling(res, random_state=1)

    assert f_min <= res.fun  # true since noise ~= 0.0
    assert x_min == x_min2
    assert f_min == f_min2


@pytest.mark.fast_test
def test_dict_list_space_representation():
    """
    Tests whether the conversion of the dictionary and list representation
    of a point from a search space works properly.
    """

    chef_space = {
        'Cooking time': (0, 1200),  # in minutes
        'Main ingredient': [
            'cheese', 'cherimoya', 'chicken', 'chard', 'chocolate', 'chicory'
        ],
        'Secondary ingredient': [
            'love', 'passion', 'dedication'
        ],
        'Cooking temperature': (-273.16, 10000.0)  # in Celsius
    }

    opt = Optimizer(dimensions=dimensions_aslist(chef_space))
    point = opt.ask()

    # check if the back transformed point and original one are equivalent
    assert_equal(
        point,
        point_aslist(chef_space, point_asdict(chef_space, point))
    )


@pytest.mark.fast_test
@pytest.mark.parametrize("estimator, gradients",
                         zip(["GP", "RF", "ET", "GBRT", "DUMMY"],
                             [True, False, False, False, False]))
def test_has_gradients(estimator, gradients):
    space = Space([(-2.0, 2.0)])

    assert has_gradients(cook_estimator(estimator, space=space)) == gradients


@pytest.mark.fast_test
def test_categorical_gp_has_gradients():
    space = Space([('a', 'b')])

    assert not has_gradients(cook_estimator('GP', space=space))


@pytest.mark.fast_test
def test_normalize_dimensions_all_categorical():
    dimensions = (['a', 'b', 'c'], ['1', '2', '3'])
    space = normalize_dimensions(dimensions)
    assert space.is_categorical


@pytest.mark.fast_test
def test_categoricals_mixed_types():
    domain = [[1, 2, 3, 4], ['a', 'b', 'c'], [True, False]]
    x = [1, 'a', True]
    space = normalize_dimensions(domain)
    assert (space.inverse_transform(space.transform([x])) == [x])


@pytest.mark.fast_test
@pytest.mark.parametrize("dimensions, normalizations",
                         [(((1, 3), (1., 3.)),
                           ('normalize', 'normalize')
                           ),
                          (((1, 3), ('a', 'b', 'c')),
                           ('normalize', 'onehot')
                           ),
                          ])
def test_normalize_dimensions(dimensions, normalizations):
    space = normalize_dimensions(dimensions)
    for dimension, normalization in zip(space, normalizations):
        assert dimension.transform_ == normalization


@pytest.mark.fast_test
@pytest.mark.parametrize("dimension, name",
                         [(Real(1, 2, name="learning rate"), "learning rate"),
                          (Integer(1, 100, name="no of trees"), "no of trees"),
                          (Categorical(["red, blue"], name="colors"), "colors")])
def test_normalize_dimensions(dimension, name):
    space = normalize_dimensions([dimension])
    assert space.dimensions[0].name == name


@pytest.mark.fast_test
def test_use_named_args():
    """
    Test the function wrapper @use_named_args which is used
    for wrapping an objective function with named args so it
    can be called by the optimizers which only pass a single
    list as the arg.

    This test does not actually use the optimizers but merely
    simulates how they would call the function.
    """

    # Define the search-space dimensions. They must all have names!
    dim1 = Real(name='foo', low=0.0, high=1.0)
    dim2 = Real(name='bar', low=0.0, high=1.0)
    dim3 = Real(name='baz', low=0.0, high=1.0)

    # Gather the search-space dimensions in a list.
    dimensions = [dim1, dim2, dim3]

    # Parameters that will be passed to the objective function.
    default_parameters = [0.5, 0.6, 0.8]

    # Define the objective function with named arguments
    # and use this function-decorator to specify the search-space dimensions.
    @use_named_args(dimensions=dimensions)
    def func(foo, bar, baz):
        # Assert that all the named args are indeed correct.
        assert foo == default_parameters[0]
        assert bar == default_parameters[1]
        assert baz == default_parameters[2]

        # Return some objective value.
        return foo ** 2 + bar ** 4 + baz ** 8

    # Ensure the objective function can be called with a single
    # argument named x.
    res = func(x=default_parameters)
    assert (isinstance(res, float))

    # Ensure the objective function can be called with a single
    # argument that is unnamed.
    res = func(default_parameters)
    assert (isinstance(res, float))

    # Ensure the objective function can be called with a single
    # argument that is a numpy array named x.
    res = func(x=np.array(default_parameters))
    assert (isinstance(res, float))

    # Ensure the objective function can be called with a single
    # argument that is an unnamed numpy array.
    res = func(np.array(default_parameters))
    assert (isinstance(res, float))


@pytest.mark.fast_test
def test_space_names_in_use_named_args():
    space = [Integer(250, 2000, name='n_estimators')]

    @use_named_args(space)
    def objective(n_estimators):
        return n_estimators

    res = gp_minimize(objective, space, n_calls=10, random_state=0)
    best_params = dict(zip((s.name for s in res.space), res.x))
    assert 'n_estimators' in best_params
    assert res.space.dimensions[0].name == 'n_estimators'


@pytest.mark.fast_test
def test_check_dimension_names():
    # Define the search-space dimensions. They must all have names!
    dim1 = Real(name='foo', low=0.0, high=1.0)
    dim2 = Real(name='bar', low=0.0, high=1.0)
    dim3 = Real(name='baz', low=0.0, high=1.0)

    # Gather the search-space dimensions in a list.
    dimensions = [dim1, dim2, dim3]
    check_dimension_names(dimensions)
    dimensions = [dim1, dim2, dim3, Real(-1, 1)]
    assert_raises(ValueError, check_dimension_names, dimensions)


@pytest.mark.fast_test
def test_check_list_types():
    # Define the search-space dimensions. They must all have names!
    dim1 = Real(name='foo', low=0.0, high=1.0)
    dim2 = Real(name='bar', low=0.0, high=1.0)
    dim3 = Real(name='baz', low=0.0, high=1.0)

    # Gather the search-space dimensions in a list.
    dimensions = [dim1, dim2, dim3]
    check_list_types(dimensions, Dimension)
    dimensions = [dim1, dim2, dim3, "test"]
    assert_raises(ValueError, check_list_types, dimensions, Dimension)
