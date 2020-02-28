"""Scikit-optimize plotting tests."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal
from skopt.space import Integer, Categorical
from skopt import plots, gp_minimize
import matplotlib.pyplot as plt
from skopt.benchmarks import bench3
from skopt import expected_minimum, expected_minimum_random_sampling
from skopt.plots import _evaluate_min_params, partial_dependence
from skopt.plots import partial_dependence_1D, partial_dependence_2D
from skopt import Optimizer


def save_axes(ax, filename):
    """Save matplotlib axes `ax` to an image `filename`."""
    fig = plt.gcf()
    fig.add_axes(ax)
    fig.savefig(filename)


@pytest.mark.slow_test
def test_plots_work():
    """Basic smoke tests to make sure plotting doesn't crash."""
    SPACE = [
        Integer(1, 20, name='max_depth'),
        Integer(2, 100, name='min_samples_split'),
        Integer(5, 30, name='min_samples_leaf'),
        Integer(1, 30, name='max_features'),
        Categorical(['gini', 'entropy'], name='criterion'),
        Categorical(list('abcdefghij'), name='dummy'),
    ]

    def objective(params):
        clf = DecisionTreeClassifier(random_state=3,
                                     **{dim.name: val
                                        for dim, val in zip(SPACE, params)
                                        if dim.name != 'dummy'})
        return -np.mean(cross_val_score(clf, *load_breast_cancer(True)))

    res = gp_minimize(objective, SPACE, n_calls=10, random_state=3)

    x = [[11, 52, 8, 14, 'entropy', 'f'],
         [14, 90, 10, 2, 'gini', 'a'],
         [7, 90, 6, 14, 'entropy', 'f']]
    samples = res.space.transform(x)
    xi_ = [1., 10.5, 20.]
    yi_ = [-0.9240883492576596, -0.9240745890422687, -0.9240586402439884]
    xi, yi = partial_dependence_1D(res.space, res.models[-1], 0,
                                   samples, n_points=3)
    assert_array_almost_equal(xi, xi_)
    assert_array_almost_equal(yi, yi_, 1e-3)

    xi_ = [0, 1]
    yi_ = [-0.9241087603770617, -0.9240188905968352]
    xi, yi = partial_dependence_1D(res.space, res.models[-1], 4,
                                   samples, n_points=3)
    assert_array_almost_equal(xi, xi_)
    assert_array_almost_equal(yi, yi_, 1e-3)

    xi_ = [0, 1]
    yi_ = [1., 10.5, 20.]
    zi_ = [[-0.92412562, -0.92403575],
           [-0.92411186, -0.92402199],
           [-0.92409591, -0.92400604]]
    xi, yi, zi = partial_dependence_2D(res.space, res.models[-1], 0, 4,
                                       samples, n_points=3)
    assert_array_almost_equal(xi, xi_)
    assert_array_almost_equal(yi, yi_)
    assert_array_almost_equal(zi, zi_, 1e-3)

    x_min, f_min = expected_minimum_random_sampling(res, random_state=1)
    x_min2, f_min2 = expected_minimum(res, random_state=1)

    x_min, f_min = expected_minimum_random_sampling(res, random_state=1)
    x_min2, f_min2 = expected_minimum(res, random_state=1)

    assert x_min == x_min2
    assert f_min == f_min2

    plots.plot_convergence(res)
    plots.plot_evaluations(res)
    plots.plot_objective(res)
    plots.plot_objective(res, dimensions=["a", "b", "c", "d", "e", "f"])
    plots.plot_objective(res,
                         minimum='expected_minimum_random')
    plots.plot_objective(res,
                         sample_source='expected_minimum_random',
                         n_minimum_search=10000)
    plots.plot_objective(res,
                         sample_source='result')
    plots.plot_regret(res)
    plots.plot_objective_2D(res, 0, 4)
    plots.plot_histogram(res, 0, 4)

    # TODO: Compare plots to known good results?
    # Look into how matplotlib does this.


@pytest.mark.slow_test
def test_plots_work_without_cat():
    """Basic smoke tests to make sure plotting doesn't crash."""
    SPACE = [
        Integer(1, 20, name='max_depth'),
        Integer(2, 100, name='min_samples_split'),
        Integer(5, 30, name='min_samples_leaf'),
        Integer(1, 30, name='max_features'),
    ]

    def objective(params):
        clf = DecisionTreeClassifier(random_state=3,
                                     **{dim.name: val
                                        for dim, val in zip(SPACE, params)
                                        if dim.name != 'dummy'})
        return -np.mean(cross_val_score(clf, *load_breast_cancer(True)))

    res = gp_minimize(objective, SPACE, n_calls=10, random_state=3)
    plots.plot_convergence(res)
    plots.plot_evaluations(res)
    plots.plot_objective(res)
    plots.plot_objective(res,
                         minimum='expected_minimum')
    plots.plot_objective(res,
                         sample_source='expected_minimum',
                         n_minimum_search=10)
    plots.plot_objective(res, sample_source='result')
    plots.plot_regret(res)

    # TODO: Compare plots to known good results?
    # Look into how matplotlib does this.


@pytest.mark.fast_test
def test_evaluate_min_params():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      noise=1e-8,
                      n_calls=8,
                      n_random_starts=3,
                      random_state=1)

    x_min, f_min = expected_minimum(res, random_state=1)
    x_min2, f_min2 = expected_minimum_random_sampling(res,
                                                      n_random_starts=1000,
                                                      random_state=1)
    plots.plot_gaussian_process(res)
    assert _evaluate_min_params(res, params='result') == res.x
    assert _evaluate_min_params(res, params=[1.]) == [1.]
    assert _evaluate_min_params(res, params='expected_minimum',
                                random_state=1) == x_min
    assert _evaluate_min_params(res, params='expected_minimum',
                                n_minimum_search=20,
                                random_state=1) == x_min
    assert _evaluate_min_params(res, params='expected_minimum_random',
                                n_minimum_search=1000,
                                random_state=1) == x_min2


def test_names_dimensions():
    # Define objective
    def objective(x, noise_level=0.1):
        return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +\
               np.random.randn() * noise_level

    # Initialize Optimizer
    opt = Optimizer([(-2.0, 2.0)], n_initial_points=1)

    # Optimize
    for i in range(2):
        next_x = opt.ask()
        f_val = objective(next_x)
        res = opt.tell(next_x, f_val)

    # Plot results
    plots.plot_objective(res)
