"""
==========================================
Scikit-learn hyperparameter search wrapper
==========================================

Iaroslav Shcherbatyi, Tim Head and Gilles Louppe. June 2017.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

Introduction
============

This example assumes basic familiarity with
`scikit-learn <http://scikit-learn.org/stable/index.html>`_.

Search for parameters of machine learning models that result in best
cross-validation performance is necessary in almost all practical
cases to get a model with best generalization estimate. A standard
approach in scikit-learn is using :obj:`sklearn.model_selection.GridSearchCV` class, which takes
a set of values for every parameter to try, and simply enumerates all
combinations of parameter values. The complexity of such search grows
exponentially with the addition of new parameters. A more scalable
approach is using :obj:`sklearn.model_selection.RandomizedSearchCV`, which however does not take
advantage of the structure of a search space.

Scikit-optimize provides a drop-in replacement for :obj:`sklearn.model_selection.GridSearchCV`,
which utilizes Bayesian Optimization where a predictive model referred
to as "surrogate" is used to model the search space and utilized to
arrive at good parameter values combination as soon as possible.

Note: for a manual hyperparameter optimization example, see
"Hyperparameter Optimization" notebook.

"""
print(__doc__)
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

#############################################################################
# Minimal example
# ===============
#
# A minimal example of optimizing hyperparameters of SVC (Support Vector machine Classifier) is given below.

from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=32,
    cv=3
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))

#############################################################################
# Advanced example
# ================
#
# In practice, one wants to enumerate over multiple predictive model classes,
# with different search spaces and number of evaluations per class. An
# example of such search over parameters of Linear SVM, Kernel SVM, and
# decision trees is given below.

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# pipeline class is used as estimator to enable
# search over different model types
pipe = Pipeline([
    ('model', SVC())
])

# single categorical value of 'model' parameter is
# sets the model class
# We will get ConvergenceWarnings because the problem is not well-conditioned.
# But that's fine, this is just an example.
linsvc_search = {
    'model': [LinearSVC(max_iter=1000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}

# explicit dimension classes can be specified like this
svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(svc_search, 40), (linsvc_search, 16)],
    cv=3
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print("best params: %s" % str(opt.best_params_))

#############################################################################
# Partial Dependence plot of the objective function for SVC
#
_ = plot_objective(opt.optimizer_results_[0],
                   dimensions=["C", "degree", "gamma", "kernel"],
                   n_minimum_search=int(1e8))
plt.show()

#############################################################################
# Plot of the histogram for LinearSVC
#
_ = plot_histogram(opt.optimizer_results_[1], 1)
plt.show()

#############################################################################
# Progress monitoring and control using `callback` argument of `fit` method
# =========================================================================
#
# It is possible to monitor the progress of :class:`BayesSearchCV` with an event
# handler that is called on every step of subspace exploration. For single job
# mode, this is called on every evaluation of model configuration, and for
# parallel mode, this is called when n_jobs model configurations are evaluated
# in parallel.
#
# Additionally, exploration can be stopped if the callback returns `True`.
# This can be used to stop the exploration early, for instance when the
# accuracy that you get is sufficiently high.
#
# An example usage is shown below.

from skopt import BayesSearchCV

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)

searchcv = BayesSearchCV(
    SVC(gamma='scale'),
    search_spaces={'C': (0.01, 100.0, 'log-uniform')},
    n_iter=10,
    cv=3
)

# callback handler
def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True


searchcv.fit(X, y, callback=on_step)

#############################################################################
# Counting total iterations that will be used to explore all subspaces
# ====================================================================
#
# Subspaces in previous examples can further increase in complexity if you add
# new model subspaces or dimensions for feature extraction pipelines. For
# monitoring of progress, you would like to know the total number of
# iterations it will take to explore all subspaces. This can be
# calculated with `total_iterations` property, as in the code below.

from skopt import BayesSearchCV

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)

searchcv = BayesSearchCV(
    SVC(),
    search_spaces=[
        ({'C': (0.1, 1.0)}, 19),  # 19 iterations for this subspace
        {'gamma':(0.1, 1.0)}
    ],
    n_iter=23
)

print(searchcv.total_iterations)
