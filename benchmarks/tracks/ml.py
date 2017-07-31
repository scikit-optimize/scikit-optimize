import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes

from skopt.space import Real, Integer, Categorical
from skopt.utils import dimensions_aslist, point_asdict, point_aslist

from misc.tinynet import ffnn_predict
from misc import simulators

import os
script_dir = os.path.dirname(os.path.realpath(__file__))

class ColumnSubset(BaseEstimator, TransformerMixin):
    def __init__(self, i1=0, i2=1):
        self.i1 = i1
        self.i2 = i2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, [self.i1, self.i2]]

# --- BENCHMARKS START HERE ---

class Select2Features:
    def __init__(self):
        X, y = load_diabetes(True)
        self.dataset = X, y
        self.search_space = {
            'features__i1': Integer(0, X.shape[1]-1),
            'features__i2': Integer(0, X.shape[1]-1),
            'model__n_estimators': Integer(1, 512),
            'model__learning_rate': Real(0.0001, 1.0, 'log-uniform'),
        }
        self.space = dimensions_aslist(self.search_space)

    def __call__(self, x):
        model = Pipeline([
            ('features', ColumnSubset()),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor()),
        ])
        x_dict = point_asdict(self.search_space, x)
        model.set_params(**x_dict)
        X, y = self.dataset
        scores = cross_val_score(model, X, y)
        return -np.mean(scores)


class Train4LayerNN:
    def __init__(self):
        # Best performing ensemble: ~0.77 (~1 MB size)
        # estimated R^2 of compact simulator: ~0.75 (~0.003 MB in size)
        # ... this will probably be gone soon and normal model will be used
        self.simulator = simulators.dnn_sim
        self.search_space = {
            'model__lr': Real(1e-6, 1.0, prior='log-uniform'),
            'model__mom': Real(0.01, 1.0, prior='log-uniform'),
            'model__l1': Integer(1, 16),
            'model__l2': Integer(1, 16),
            'model__l3': Integer(1, 16),
            'model__l4': Integer(1, 16),
            'model__batch_size': Integer(32, 256),
            'model__epochs': Integer(1, 128),
        }
        self.space = dimensions_aslist(self.search_space)

    def __call__(self, x):
        x_dict = point_asdict(self.search_space, x)
        x_dict['model__lr'] = np.log(x_dict['model__lr'])
        x_list = point_aslist(self.search_space, x_dict)

        score = ffnn_predict([x_list], self.simulator)[0][0]

        return -np.mean(score)


problems = [Select2Features, Train4LayerNN]

