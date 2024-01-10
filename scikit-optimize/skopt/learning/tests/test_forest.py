import numpy as np
import pytest

from scipy import stats

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

from skopt.learning import ExtraTreesRegressor, RandomForestRegressor


def truth(X):
    return 0.5 * np.sin(1.75*X[:, 0])


@pytest.mark.fast_test
def test_random_forest():
    # toy sample
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]
    T = [[-1, -1], [2, 2], [3, 2]]
    true_result = [-1, 1, 1]

    clf = RandomForestRegressor(n_estimators=10, random_state=1)
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = RandomForestRegressor(n_estimators=10, min_impurity_decrease=0.1,
                                random_state=1)
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = RandomForestRegressor(n_estimators=10, criterion="mse",
                                max_depth=None, min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.,
                                max_features="auto", max_leaf_nodes=None,
                                min_impurity_decrease=0., bootstrap=True,
                                oob_score=False,
                                n_jobs=1, random_state=1,
                                verbose=0, warm_start=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = RandomForestRegressor(n_estimators=10, max_features=1,
                                random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    # also test apply
    leaf_indices = clf.apply(X)
    assert leaf_indices.shape == (len(X), clf.n_estimators)


@pytest.mark.fast_test
def test_extra_forest():
    # toy sample
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]
    T = [[-1, -1], [2, 2], [3, 2]]
    true_result = [-1, 1, 1]

    clf = ExtraTreesRegressor(n_estimators=10, random_state=1)
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = ExtraTreesRegressor(n_estimators=10, min_impurity_decrease=0.1,
                              random_state=1)
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = ExtraTreesRegressor(n_estimators=10, criterion="mse",
                              max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.,
                              max_features="auto", max_leaf_nodes=None,
                              min_impurity_decrease=0., bootstrap=False,
                              oob_score=False,
                              n_jobs=1, random_state=1,
                              verbose=0, warm_start=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = ExtraTreesRegressor(n_estimators=10, max_features=1, random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    # also test apply
    leaf_indices = clf.apply(X)
    assert leaf_indices.shape == (len(X), clf.n_estimators)
