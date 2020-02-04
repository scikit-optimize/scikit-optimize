"""Scikit-optimize plotting tests."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from skopt.space import Integer, Categorical
from skopt import plots, gp_minimize
import matplotlib.pyplot as plt


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
    plots.plot_convergence(res)
    plots.plot_evaluations(res)
    plots.plot_objective(res)
    plots.plot_objective(res,
                         eval_min_params='expected_minimum_random')
    plots.plot_objective(res,
                         eval_min_params='expected_minimum')
    plots.plot_objective(res,
                         usepartialdependence=True)
    plots.plot_regret(res)

    # TODO: Compare plots to known good results?
    # Look into how matplotlib does this.
