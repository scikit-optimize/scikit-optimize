
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

# models used
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin, RegressorMixin

from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer

import numpy as np
import json
from datetime import datetime
from joblib import Parallel, delayed

OUTPUT_TYPE = "Output type"
OUTPUT_CONTINUOUS = "Number"
OUTPUT_CATEGORY = "Category"

MODEL_PARAMETERS = "model parameters"
MODEL_BACKEND = "model backend"
PARAM_MAP = "parameter map"

DATASET_LOADER = "Dataset loader"

# bunch of dataset preprocessing functions below

def split_normalize(X, y, random_state):

    """
    Splits data into training and validation parts.
    Test data is assumed to be used after optimization.

    :param X: matrix where every row is a training input
    :param y: vector where every entry is correpsonding output
    :return: split of data into training and validation sets.
            70% of data is used for training, rest for validation.
    """

    X, Xv, y, yv = train_test_split(X, y, train_size=0.7, random_state=random_state)


    sc = StandardScaler() # normalize data
    sc.fit(X, y)

    X, Xv = sc.transform(X), sc.transform(Xv)

    return X, y, Xv, yv

# functions below are used to apply non - linear maps to parameter values, eg
# -3.0 -> 0.001

def pow10map(x):
    return 10.0 ** x

def pow2intmap(x):
    return int(2.0 ** x)

def nop(x):
    return x

# this is used to process the output of fetch_mldata
def load_data_target(data):
    X = data['data']
    Y = data['target']
    return X, Y

class MLBench():

    """
    A class which is used to perform benchmarking of black box optimization algorithms on various
    machine learning problems.
    On __init__, the dataset is loaded that is used for experimentation, and it is kept in memory
    in order to avoid reloading data.
    """
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

    # every model should have fields:

    # MODEL_PARAMETERS: dictionary whose keys correspond to parameters that are passed to __init__ of python class
    #                   that implements model
    # MODEL_BACKEND:    python class that implements model. Should be a subclass of RegressorMixin or ClassifierMixin,
    #                   so that the type of output of the model could be determined easily.
    class_to_outp = {
        RegressorMixin: OUTPUT_CONTINUOUS,
        ClassifierMixin: OUTPUT_CATEGORY
    }

    models = {
        #
        #   regressors go here
        #
        MLPRegressor.__name__:{
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: MLPRegressor,
        },
        SVR.__name__:{
            MODEL_PARAMETERS: {
                'C': (Real(-4.0, 4.0), pow10map),
                'epsilon': (Real(-4.0, 1.0), pow10map),
                'gamma': (Real(-4.0, 1.0), pow10map),
            },
            MODEL_BACKEND: SVR,
        },
        DecisionTreeRegressor.__name__:{
            MODEL_PARAMETERS: {
                'max_depth': (Real(1.0, 4.0), pow2intmap),
                'min_samples_split': (Real(1.0, 8.0), pow2intmap),
            },
            MODEL_BACKEND: DecisionTreeRegressor,
        },
        #
        #   classifiers go here
        #
        MLPClassifier.__name__:{
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: MLPClassifier,
        },
        SVC.__name__:{
            MODEL_PARAMETERS: {
                'C': (Real(-4.0, 4.0), pow10map),
                'gamma': (Real(-4.0, 1.0), pow10map),
            },
            MODEL_BACKEND: SVC,
        },
        DecisionTreeClassifier.__name__: {
            MODEL_PARAMETERS: {
                'max_depth': (Real(1.0, 4.0), pow2intmap),
                'min_samples_split': (Real(1.0, 8.0), pow2intmap),
            },
            MODEL_BACKEND: DecisionTreeClassifier,
        },
    }

    # every dataset should have fields:
    # OUTPUT_TYPE:  OUTPUT_CONTINUOUS / OUTPUT_CATEGORY
    # DATASET_LOADER: callable that loads data into matrix X of inputs and vector y of outputs
    datasets={
        "Boston":{
            OUTPUT_TYPE: OUTPUT_CONTINUOUS,
            DATASET_LOADER: lambda : load_data_target(skd.load_boston())
        },
        "Housing":{
            OUTPUT_TYPE: OUTPUT_CONTINUOUS,
            DATASET_LOADER: lambda : load_data_target(skd.fetch_california_housing())
        },
        #
        #   classification datasets go here
        #
        "Leukemia":{
            OUTPUT_TYPE: OUTPUT_CATEGORY,
            DATASET_LOADER: lambda : load_data_target(skd.fetch_mldata('leukemia'))
        },
        "Climate Model Crashes":{
            OUTPUT_TYPE: OUTPUT_CATEGORY,
            DATASET_LOADER: lambda : load_data_target(skd.fetch_mldata('climate-model-simulation-crashes', target_name=-1))
        },
    }

    @staticmethod
    def supports(model, dataset):
        """
        Used to determine if model can be trained on particular dataset.
        :param model: string, name of model class
        :param dataset: string, name of dataset
        :return: bool
        """


        if not dataset in MLBench.datasets:
            return False, "unknown dataset"

        selected_dataset = MLBench.datasets[dataset]
        if not model in MLBench.models:
            return False, "unknown model"

        selection = MLBench.models[model]
        model_backend = selection[MODEL_BACKEND]

        # check the model output
        if not any([issubclass(model_backend, k) for k in MLBench.class_to_outp.keys()]):
            raise TypeError("The model is not a subclass of any supported estimator."
                             "Supported estimators are: " + str(MLBench.class_to_outp.keys()))

        for k in MLBench.class_to_outp.keys():
            if issubclass(model_backend,k):
                model_output = MLBench.class_to_outp[k]

        if not selected_dataset[OUTPUT_TYPE] == model_output:

            return False, "model cannot be applied to the dataset"

        return True, None

    def __init__(self, model, dataset, random_state):

        supports, reason_if_not = MLBench.supports(model, dataset)


        if not supports:
            raise BaseException(reason_if_not)

        X, Y = self.datasets[dataset][DATASET_LOADER]()

        self.dataset = split_normalize(X, Y, random_state)
        self.model_description = MLBench.models[model]


    def evaluate(self, point):
        """
        Evaluates model on loaded dataset for particular setting of hyperparameters.
        :param point: dict, pairs of parameter names and corresponding values
        :return: score (more is better!) for some specific point
        """

        X, Y, Xv, Yv = self.dataset

        # apply transformation to model parameters, for example exp transformation
        point_mapped = {}

        for param in point.keys():
            v = point[param]
            v = self.model_description[MODEL_PARAMETERS][param][1](v)
            point_mapped[param] = v

        model = self.model_description[MODEL_BACKEND](**point_mapped)

        try:
            model.fit(X, Y)
            r = model.score(Xv, Yv)
        except BaseException as ex:
            r = 0.0 # on error: return assumed smallest value of objective function

        # while negative values could be informative, they could be very large also,
        # which could mess up the optimization procedure. Suggestions are welcome.
        r = max(r, 0.0)

        return r

# this is necessary to generate table for README in the end
table_template = """|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|ML_hypp_tune|"""

def calculate_performance(all_data):
    """
    Calculates the performance metrics as found in "benchmarks" folder of scikit-optimize,
    and prints them in console.

    :param all_data: dict, traces data collected during run of algorithms. For more details, see
                    'evaluate_optimizer' function.
    :return: nothing
    """

    average_for_algo = {}

    for Model in all_data:
        for Dataset in all_data[Model]:
            for Algorithm in all_data[Model][Dataset]:
                data = all_data[Model][Dataset][Algorithm]

                if not Algorithm in average_for_algo:
                    average_for_algo[Algorithm] = []

                objs = [[v[-1] for v in d] for d in data] # leave only best objective values at particular iteration

                # here objs is a 2d list, where first dimension corresponds to
                # particular repeat of experiment, and second dimension corresponds to index
                # of optimization step during optimization
                average_for_algo[Algorithm].append(objs)

    # calculate averages
    for Algorithm in average_for_algo:
        mean_obj_vals = np.mean(average_for_algo[Algorithm], axis=0)
        minimums = np.min(mean_obj_vals, axis=1)
        f_calls = np.argmin(mean_obj_vals, axis=1)

        min_mean = np.mean(minimums)
        min_stdd = np.std(minimums)
        min_best = np.min(minimums)

        f_mean = np.mean(f_calls)
        f_stdd = np.std(f_calls)
        f_best = np.min(f_calls)

        def fmt(float_value):
            return ("%.3f" % float_value)

        output = "|".join([fmt(min_mean) + " +/- " + fmt(min_stdd)] + [fmt(v) for v in [ min_best, f_mean, f_stdd, f_best]])
        result = table_template + output
        print("")
        print(Algorithm)
        print(result)


def evaluate_optimizer(surrogate, model, dataset, n_calls, random_state):
    """
    Evaluates some estimator for the task of optimization of parameters of some
    model, given limited number of model evaluations.
    Parentheses on parameters are used to be able to run function with pool.map method.

    :param surrogate: Estimator to use for optimization.
    :param model: str, name of the ML model class to be used for parameter tuning
    :param dataset: str, name of dataset to train ML model on
    :param n_calls: a budget of evaluations
    :param random_state: random seed, used to set the random number generator in numpy
    :return: a list of paris (p, f(p)), where p is a dictionary of the form "param name":value,
            and f(p) is performance measure value achieved by the model for configuration p.
            Such list contains history of execution of optimization.
    """


    # below seed is necessary for processes which fork at the same time
    # so that random numbers generated in processes are different

    np.random.seed(random_state)

    problem = MLBench(model, dataset, random_state=random_state)
    space = problem.model_description[MODEL_PARAMETERS]

    # initialization
    estimator = surrogate(random_state=random_state)
    dimensions_names = space.keys()

    dimensions = [space[k][0] for k in dimensions_names]
    solver = Optimizer(dimensions, estimator, random_state=random_state)

    trace = []
    best_y = np.inf
    best_x = None

    # optimization loop
    for i in range(n_calls):
        point_list = solver.ask()

        point_dct = {k: v for k, v in zip(dimensions_names, point_list)}  # convert list of dimension values to dictionary
        objective_at_point = -problem.evaluate(point_dct) # the result of "evaluate" is accuracy / r^2, which is the more the better

        if best_y > objective_at_point:
            best_y = objective_at_point

        trace.append((point_dct, objective_at_point, best_y)) # remember the point, objective pair
        print("Eval. #"+str(i))

        solver.tell(point_list, objective_at_point)

    return trace

# the function below is necessary as a workaround for the pool.map method requiring map function to take only a single argument
def _evaluate_optimizer(params):
    return evaluate_optimizer(*params)


def run(n_calls = 32,
        n_runs = 1,
        save_traces = True,
        n_jobs = -1):

    """
    Main function used to run the experiments.

    :param n_calls: evaluation budget
    :param n_runs: how many times to repeat the optimization in order to average out noise
    :param save_traces: whether to save data collected during optimization

    :param n_jobs: number of different repeats of optimization to run in parallel
    :return: None
    """

    surrogates = [GaussianProcessRegressor]
    selected_models = MLBench.models.keys()
    selected_datasets = MLBench.datasets.keys()

    # all the parameter values and objectives collected during execution are stored in list below
    all_data = {}

    for model in selected_models:
        all_data[model] = {}

        for dataset in selected_datasets:

            supports, _ = MLBench.supports(model, dataset)
            if not supports:
                continue

            all_data[model][dataset] = {}
            for surrogate in surrogates:
                print(surrogate.__name__, model, dataset)

                seeds = np.random.randint(0, 2 ** 30, n_runs)

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
        '--n_calls', nargs="?", default=64, type=int,
        help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=5, type=int,
        help="Number of re-runs of single algorithm on single instance of a problem, in order to average out the noise.")
    parser.add_argument(
        '--save_traces', nargs="?", default=False, type=bool,
        help="Whether to save pairs (point, objective, best_objective) obtained during experiments in a json file.")
    parser.add_argument(
        '--n_jobs', nargs="?", default=-1, type=bool,
        help="Number of worker processes used for the benchmark.")

    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.save_traces, args.n_jobs)

