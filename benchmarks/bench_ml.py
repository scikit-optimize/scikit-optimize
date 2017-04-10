
"""
This code implements benchmark for the black box optimization algorithms,
applied to a task of optimizing parameters of ML algorithms for the task
of supervised learning.

The code implements benchmark on 4 datasets where parameters for 6 classes
of supervised models are tuned to optimize performance on datasets. Supervised
learning models implementations are taken from sklearn.

Regression learning task is solved on 2 datasets, and classification on the
rest of datasets. 3 model classes are regression models, and rest are
classification models.
"""
from collections import defaultdict
from datetime import datetime
import json
import os
import sys
if sys.version_info.major == 2:
    # Python 2
    from urllib2 import HTTPError
    from urllib import urlretrieve
else:
    from urllib.error import HTTPError
    from urllib.request import urlretrieve

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from skopt import Optimizer
from skopt.learning import ExtraTreesRegressor
from skopt.learning import GaussianProcessRegressor
from skopt.learning import GradientBoostingQuantileRegressor
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real


MODEL_PARAMETERS = "model parameters"
MODEL_BACKEND = "model backend"

# functions below are used to apply non - linear maps to parameter values, eg
# -3.0 -> 0.001
def pow10map(x):
    return 10.0 ** x

def pow2intmap(x):
    return int(2.0 ** x)

def nop(x):
    return x

nnparams = {
    # up to 1024 neurons
    'hidden_layer_sizes': (Real(1.0, 10.0), pow2intmap),
    'activation': (Categorical(['identity', 'logistic', 'tanh', 'relu']), nop),
    'solver': (Categorical(['lbfgs', 'sgd', 'adam']), nop),
    'alpha': (Real(-5.0, -1), pow10map),
    'batch_size': (Real(5.0, 10.0), pow2intmap),
    'learning_rate': (Categorical(['constant', 'invscaling', 'adaptive']), nop),
    'max_iter': (Real(5.0, 8.0), pow2intmap),
    'learning_rate_init': (Real(-5.0, -1), pow10map),
    'power_t': (Real(0.01, 0.99), nop),
    'momentum': (Real(0.1, 0.98), nop),
    'nesterovs_momentum': (Categorical([True, False]), nop),
    'beta_1': (Real(0.1, 0.98), nop),
    'beta_2': (Real(0.1, 0.9999999), nop),
}


MODELS = {
    MLPRegressor(): nnparams,
    SVR(): {
            'C': (Real(-4.0, 4.0), pow10map),
            'epsilon': (Real(-4.0, 1.0), pow10map),
            'gamma': (Real(-4.0, 1.0), pow10map)},
    DecisionTreeRegressor(): {
            'max_depth': (Real(1.0, 4.0), pow2intmap),
            'min_samples_split': (Real(1.0, 8.0), pow2intmap)},
    MLPClassifier(): nnparams,
    SVC(): {
            'C': (Real(-4.0, 4.0), pow10map),
            'gamma': (Real(-4.0, 1.0), pow10map)},
    DecisionTreeClassifier(): {
            'max_depth': (Real(1.0, 4.0), pow2intmap),
            'min_samples_split': (Real(1.0, 8.0), pow2intmap)}
}

# every dataset should have have a mapping to the mixin that can handle it.
DATASETS = {
    "Boston": RegressorMixin,
    "Housing": RegressorMixin,
    "digits": ClassifierMixin,
    "Climate Model Crashes": ClassifierMixin,
}

# bunch of dataset preprocessing functions below
def split_normalize(X, y, random_state):
    """
    Splits data into training and validation parts.
    Test data is assumed to be used after optimization.

    Parameters
    ----------
    * `X` [array-like, shape = (n_samples, n_features)]:
        Training data.

    * `y`: [array-like, shape = (n_samples)]:
        Target values.

    Returns
    -------
    Split of data into training and validation sets.
    70% of data is used for training, rest for validation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)
    sc = StandardScaler()
    sc.fit(X_train, y_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    return X_train, y_train, X_test, y_test


# this is used to process the output of fetch_mldata
def load_data_target(name):
    """
    Loads data and target given the name of the dataset.
    """
    if name == "Boston":
        data = load_boston()
    elif name == "Housing":
        data = fetch_california_housing()
    elif name == "digits":
        data = load_digits()
    elif name == "Climate Model Crashes":
        try:
            data = fetch_mldata("climate-model-simulation-crashes")
        except HTTPError as e:
            #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
            url = "http://nrvis.com/data/mldata/pop_failures.csv"
            urlretrieve(url, "pop_failures.dat")
            data = dict()
            samples = np.loadtxt("pop_failures.dat", skiprows=1,delimiter=",")
            os.remove("pop_failures.dat")
            data["data"] = samples[:, :-1]
            data["target"] = np.array(samples[:, -1], dtype=np.int)
    else:
        raise ValueError("dataset not supported.")
    return data["data"], data["target"]


class MLBench(object):
    """
    A class which is used to perform benchmarking of black box optimization
    algorithms on various machine learning problems.
    On instantiation, the dataset is loaded that is used for experimentation
    and is kept in memory in order to avoid delays due to reloading of data.

    Parameters
    ----------
    * `model`: scikit-learn estimator
        An instance of a sklearn estimator.

    * `dataset`: str
        Name of the dataset.

    * `random_state`: seed
        Initialization for the random number generator in numpy.
    """
    def __init__(self, model, dataset, random_state):
        X, Y = load_data_target(dataset)
        self.X_train, self.y_train, self.X_test, self.y_test = split_normalize(
            X, Y, random_state)
        self.random_state = random_state
        self.model = model
        self.space = MODELS[model]

    def evaluate(self, point):
        """
        Fits model using the particular setting of hyperparameters and
        evaluates the model validation data.

        Parameters
        ----------
        * `point`: dict
            A mapping of parameter names to the corresponding values

        Returns
        -------
        * `score`: float
            Score (more is better!) for some specific point
        """
        X_train, y_train, X_test, y_test = (
            self.X_train, self.y_train, self.X_test, self.y_test)

        # apply transformation to model parameters, for example exp transformation
        point_mapped = {}
        for param, val in point.items():
            point_mapped[param] = self.space[param][1](val)

        try:
            params = {'random_state':self.random_state}
            params.update(point_mapped)
            self.model.set_params(**params)
        except TypeError as ex:
            self.model.set_params(**point_mapped)

        # Infeasible parameters are expected to raise an exception, thus the try
        # catch below, infeasible parameters yield assumed smallest objective.
        try:
            self.model.fit(X_train, y_train)
            r = self.model.score(X_test, y_test)
        except BaseException as ex:
            r = 0.0 # on error: return assumed smallest value of objective function

        # while negative values could be informative, they could be very large also,
        # which could mess up the optimization procedure. Suggestions are welcome.
        return max(r, 0.0)

# this is necessary to generate table for README in the end
table_template = """|Blackbox Function| Minimum | Best minimum |
------------------|------------|-----------|---------------------|
|ML_hypp_tune|"""

def calculate_performance(all_data):
    """
    Calculates the performance metrics as found in "benchmarks" folder of
    scikit-optimize and prints them in console.

    Parameters
    ----------
    * `all_data`: dict
        Traces data collected during run of algorithms. For more details, see
        'evaluate_optimizer' function.
    """
    best_for_algo = defaultdict(list)
    curr_func_algo = defaultdict(list)

    for model in all_data:
        for dataset in all_data[model]:
            for algorithm in all_data[model][dataset]:
                data = all_data[model][dataset][algorithm]

                # leave only best objective values at particular iteration
                best = [v[-1] for d in data for v in d]

                # here best is a 2d list, where first dimension corresponds to
                # particular repeat of experiment, and second dimension corresponds to index
                # of optimization step during optimization
                best_for_algo[algorithm].append(best)

    # calculate averages
    for algorithm in best_for_algo:
        mean_obj_vals = np.mean(best_for_algo[algorithm], axis=0)
        min_mean = np.mean(mean_obj_vals)
        min_std = np.std(mean_obj_vals)
        min_best = np.min(mean_obj_vals)

        def fmt(float_value):
            return ("%.3f" % float_value)

        output = "|".join([fmt(min_mean) + " +/- " + fmt(min_std), fmt(min_best)])
        result = table_template + output
        print("")
        print(algorithm)
        print(result)


def evaluate_optimizer(surrogate, model, dataset, n_calls, random_state):
    """
    Evaluates some estimator for the task of optimization of parameters of some
    model, given limited number of model evaluations.

    Parameters
    ----------
    * `surrogate`:
        Estimator to use for optimization.
    * `model`: scikit-learn estimator.
        sklearn estimator used for parameter tuning.
    * `dataset`: str
        Name of dataset to train ML model on.
    * `n_calls`: int
        Budget of evaluations
    * `random_state`: seed
        Set the random number generator in numpy.

    Returns
    -------
    * `trace`: list of tuples
        (p, f(p), best), where p is a dictionary of the form "param name":value,
        and f(p) is performance achieved by the model for configuration p
        and best is the best value till that index.
        Such a list contains history of execution of optimization.
    """
    # below seed is necessary for processes which fork at the same time
    # so that random numbers generated in processes are different
    np.random.seed(random_state)
    problem = MLBench(model, dataset, random_state)
    space = problem.space

    # initialization
    estimator = surrogate(random_state=random_state)
    dimensions_names = sorted(space)
    dimensions = [space[d][0] for d in dimensions_names]
    solver = Optimizer(dimensions, estimator, random_state=random_state)

    trace = []
    best_y = np.inf

    # optimization loop
    for i in range(n_calls):
        point_list = solver.ask()

        # convert list of dimension values to dictionary
        point_dct = dict(zip(dimensions_names, point_list))

        # the result of "evaluate" is accuracy / r^2, which is the more the better
        objective_at_point = -problem.evaluate(point_dct)

        if best_y > objective_at_point:
            best_y = objective_at_point

        # remember the point, objective pair
        trace.append((point_dct, objective_at_point, best_y))
        print("Evaluation no. " + str(i + 1))

        solver.tell(point_list, objective_at_point)
    return trace


def run(n_calls=32, n_runs=1, save_traces=True, n_jobs=1):
    """
    Main function used to run the experiments.

    Parameters
    ----------
    * `n_calls`: int
        Evaluation budget.

    * `n_runs`: int
        Number of times to repeat the optimization in order to average out noise.

    * `save_traces`: bool
        Whether or not to save data collected during optimization

    * `n_jobs`: int
        Number of different repeats of optimization to run in parallel.
    """
    surrogates = [GaussianProcessRegressor, ExtraTreesRegressor, GradientBoostingQuantileRegressor]
    selected_models = sorted(MODELS, key=lambda x: x.__class__.__name__)
    selected_datasets = sorted(DATASETS.keys())

    # all the parameter values and objectives collected during execution are stored in list below
    all_data = {}
    for model in selected_models:
        all_data[model] = {}

        for dataset in selected_datasets:
            if not isinstance(model, DATASETS[dataset]):
                continue

            all_data[model][dataset] = {}
            for surrogate in surrogates:
                print(surrogate.__name__, model.__class__.__name__, dataset)
                seeds = np.random.randint(0, 2**30, n_runs)
                raw_trace = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_optimizer)(
                        surrogate, model, dataset, n_calls, seed
                    ) for seed in seeds
                )
                all_data[model][dataset][surrogate.__name__] = raw_trace

    # dump the recorded objective values as json
    if save_traces:
        with open(datetime.now().strftime("%m_%Y_%d_%H_%m_%s")+'.json', 'w') as f:
            json.dump(all_data, f)
    calculate_performance(all_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_calls', nargs="?", default=50, type=int,
        help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=10, type=int,
        help="Number of re-runs of single algorithm on single instance of a "
        "problem, in order to average out the noise.")
    parser.add_argument(
        '--save_traces', nargs="?", default=False, type=bool,
        help="Whether to save pairs (point, objective, best_objective) obtained"
        " during experiments in a json file.")
    parser.add_argument(
        '--n_jobs', nargs="?", default=1, type=int,
        help="Number of worker processes used for the benchmark.")

    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.save_traces, args.n_jobs)
