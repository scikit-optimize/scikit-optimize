# The bokeh plotting method needs a .pickle file.
# We create he file here so we have something to test with


from ProcessOptimizer.benchmarks import branin as branin
from ProcessOptimizer.benchmarks import hart6 as hart6_
from ProcessOptimizer.plots import plot_objective
from ProcessOptimizer import gp_minimize, forest_minimize, dummy_minimize
from ProcessOptimizer import plots
from ProcessOptimizer.space import Integer, Categorical
import pickle
import numpy as np
# Here we define a function that we evaluate.
def funny_func(x):
    s = 0
    for i in range(len(x)):
        s += (x[i]*i)**2
    return s

# we use different bounds to test if bokeh plottig works properly
bounds = [(-5, 4),(-3, 3),(-2, 2),(-1, 1),(-1, 1),(-1, 1)]
n_calls = 150

result = gp_minimize(funny_func, bounds, n_calls=300, base_estimator="ET",
                             random_state=100,n_random_starts=250)

# we delete this pointer to the function to make pickle work
del result.specs['args']['func']

pickle_out = open('result.pickle', 'wb')
pickle.dump(result,pickle_out)


#we also save one with categorical values

SPACE = [
    Integer(1, 20, name='max_depth'),
    Integer(2, 100, name='min_samples_split'),
    Integer(5, 30, name='min_samples_leaf'),
    Integer(1, 30, name='max_features'),
    Categorical(list('abc'), name='dummy'),
    Categorical(['gini', 'entropy'], name='criterion'),
    Categorical(list('def'), name='dummy'),
]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
def objective(params):
    clf = DecisionTreeClassifier(**{dim.name: val for dim, val in zip(SPACE, params) if dim.name != 'dummy'})
    return -np.mean(cross_val_score(clf, *load_breast_cancer(True)))

result = gp_minimize(objective, SPACE, n_calls = 40, n_random_starts=20)

del result.specs['args']['func']

pickle_out = open('result_cat.pickle', 'wb')
pickle.dump(result,pickle_out)