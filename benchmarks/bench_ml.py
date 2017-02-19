import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
import json
from datetime import datetime

from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import datasets as skd

from skopt import Optimizer
from skopt.space import Real, Categorical, Integer

OUTPUT_TYPE = "Output type"
FMT_NUMBER = "Number"
FMT_CATEGORY = "Category"

MODEL_PARAMETERS = "model parameters"
MODEL_BACKEND = "model backend"
PARAM_MAP = "parameter map"

CATEGORY_VARIABLE = "Categorical"
REAL_VARIABLE = "Real"
INTEGER_VARIABLE = "Integer"

VARIABLE_TYPE = "type"
VARIABLE_RANGE = "range"

DATASET_LOADER = "Dataset loader"

# bunch of dataset preprocessing functions below

def split_normalize(X, y):
    """
    Splits data into training and validation parts.
    Test data is assumed to be used after optimization.

    :param X: matrix where every row is a training input
    :param y: vector where every entry is correpsonding output
    :return: split of data into training and validation sets.
            70% of data is used for training, rest for validation.
    """

    tr_fold = int(len(X) * 0.7) # split data
    X, Xv = X[:tr_fold], X[tr_fold:]
    y, yv = y[:tr_fold], y[tr_fold:]

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

class TesterClass():
    """
    A class which is used to perform benchmarking of black box optimization algorithms on various
    machine learning problems.
    On __init__, the dataset is loaded that is used for experimentation, and it is kept in memory
    in order to avoid reloading data.
    """
    nnparams = {
        # up to 1024 neurons
        'hidden_layer_sizes': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 10.0], PARAM_MAP: pow2intmap},
        'activation': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: ['identity', 'logistic', 'tanh', 'relu'], PARAM_MAP: nop},
        'solver': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: ['lbfgs', 'sgd', 'adam'], PARAM_MAP: nop},
        'alpha': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-5.0, -1], PARAM_MAP: pow10map},
        'batch_size': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [5.0, 10.0], PARAM_MAP: pow2intmap},
        'learning_rate': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: ['constant', 'invscaling', 'adaptive'], PARAM_MAP: nop},
        'max_iter': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [5.0, 8.0], PARAM_MAP: pow2intmap},
        'learning_rate_init': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-5.0, -1], PARAM_MAP: pow10map},
        'power_t': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.01, 0.99], PARAM_MAP: nop},
        'momentum': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.1, 0.98], PARAM_MAP: nop},
        'nesterovs_momentum': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: [True, False], PARAM_MAP: nop},
        'beta_1': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.1, 0.98], PARAM_MAP: nop},
        'beta_2': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.1, 0.9999999], PARAM_MAP: nop},
    }

    # every model should have fields:
    # OUTPUT_TYPE:  FMT_NUMBER / FMT_CATEGORY
    # MODEL_PARAMETERS: dictionary whose keys correspond to parameters that are passed to __init__ of python class
    #                   that implements model
    # MODEL_BACKEND: python class that implements model
    models = {
        #
        #   regressors go here
        #
        MLPRegressor.__name__:{
            OUTPUT_TYPE: FMT_NUMBER,
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: MLPRegressor,
        },
        SVR.__name__:{
            OUTPUT_TYPE: FMT_NUMBER,
            MODEL_PARAMETERS: {
                'C': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-4.0, 4.0], PARAM_MAP: pow10map},
                'epsilon': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-4.0, 1.0], PARAM_MAP: pow10map},
                'gamma': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-4.0, 1.0], PARAM_MAP: pow10map},
            },
            MODEL_BACKEND: SVR,
        },
        DecisionTreeRegressor.__name__:{
            OUTPUT_TYPE: FMT_NUMBER,
            MODEL_PARAMETERS: {
                'max_depth': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 4.0], PARAM_MAP: pow2intmap},
                'min_samples_split': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 8.0], PARAM_MAP: pow2intmap},
            },
            MODEL_BACKEND: DecisionTreeRegressor,
        },
        #
        #   classifiers go here
        #
        MLPClassifier.__name__:{
            OUTPUT_TYPE: FMT_CATEGORY,
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: MLPClassifier,
        },
        SVC.__name__:{
            OUTPUT_TYPE: FMT_CATEGORY,
            MODEL_PARAMETERS: {
                'C': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-4.0, 4.0], PARAM_MAP: pow10map},
                'gamma': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-4.0, 1.0], PARAM_MAP: pow10map},
            },
            MODEL_BACKEND: SVC,
        },
        DecisionTreeClassifier.__name__: {
            OUTPUT_TYPE: FMT_CATEGORY,
            MODEL_PARAMETERS: {
                'max_depth': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 4.0], PARAM_MAP: pow2intmap},
                'min_samples_split': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 8.0], PARAM_MAP: pow2intmap},
            },
            MODEL_BACKEND: DecisionTreeClassifier,
        },
    }

    # every dataset should have fields:
    # OUTPUT_TYPE:  FMT_NUMBER / FMT_CATEGORY
    # DATASET_LOADER: callable that loads data into matrix X of inputs and vector y of outputs
    datasets={
        "Boston":{
            OUTPUT_TYPE: FMT_NUMBER,
            DATASET_LOADER: lambda : load_data_target(skd.load_boston())
        },
        "Housing":{
            OUTPUT_TYPE: FMT_NUMBER,
            DATASET_LOADER: lambda : load_data_target(skd.fetch_california_housing())
        },
        #
        #   classification datasets go here
        #
        "Leukemia":{
            OUTPUT_TYPE: FMT_CATEGORY,
            DATASET_LOADER: lambda : load_data_target(skd.fetch_mldata('leukemia'))
        },
        "Climate Model Crashes":{
            OUTPUT_TYPE: FMT_CATEGORY,
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

        if not dataset in TesterClass.datasets:
            return False, "unknown dataset"

        dst = TesterClass.datasets[dataset]

        if not model in TesterClass.models:
            return False, "unknown model"

        mod = TesterClass.models[model]

        if not dst[OUTPUT_TYPE] == mod[OUTPUT_TYPE]:
            return False, "model cannot be applied to the dataset"

        return True, None

    def __init__(self, model, dataset):
        supports, reason_if_not = TesterClass.supports(model, dataset)

        if not supports:
            raise BaseException(reason_if_not)

        X, Y = self.datasets[dataset][DATASET_LOADER]()

        self.dataset = split_normalize(X, Y)
        self.model_class = TesterClass.models[model]

    def evaluate(self, point):
        """
        Evaluates model on loaded dataset for particular setting of hyperparameters.
        :param point: dict, pairs of parameter names and corresponding values
        :return: score (more is better!) for some specific point
        """

        X, Y, Xv, Yv = self.dataset
        cls = self.model_class[MODEL_BACKEND]

        # apply transformation to model parameters, for example exp transformation
        point_mapped = {}

        for param, v in point.iteritems():
            v = self.model_class[MODEL_PARAMETERS][param][PARAM_MAP](v)
            point_mapped[param] = v

        model = cls(**point_mapped)

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

                objs = [[v[1] for v in d] for d in data] # leave only objective values
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


def evaluate_optimizer((base_estimator, model, dataset, max_iter, seed_value)):
    """
    Evaluates some estimator for the task of optimization of parameters of some
    model, given limited number of model evaluations.
    Parentheses on parameters are used to be able to run function with pool.map method.

    :param base_estimator: Estimator to use for optimization.
    :param model: str, name of the ML model class to be used for parameter tuning
    :param dataset: str, name of dataset to train ML model on
    :param max_iter: a budget of evaluations
    :param seed_value: random seed, used to set the random number generator in numpy
    :return: a list of paris (p, f(p)), where p is a dictionary of the form "param name":value,
            and f(p) is performance measure value achieved by the model for configuration p.
            Such list contains history of execution of optimization.
    """


    # below seed is necessary for processes which fork at the same time
    # so that random numbers generated in processes are different
    np.random.seed(seed_value)

    problem = TesterClass(model, dataset)
    space = problem.model_class[MODEL_PARAMETERS]

    dimensions = []

    # convert space dictionary to dimensions
    for k,v in space.iteritems():

        if v[VARIABLE_TYPE] == REAL_VARIABLE: # if real variable ...
            dim_range = v[VARIABLE_RANGE]
            dim = Real(dim_range[0], dim_range[1])
        elif v[VARIABLE_TYPE] == INTEGER_VARIABLE: # if integer ...
            dim_range = v[VARIABLE_RANGE]
            dim = Integer(dim_range[0], dim_range[1])
        elif v[VARIABLE_TYPE] == CATEGORY_VARIABLE: # if category ...
            dim_range = v[VARIABLE_RANGE]
            dim = Categorical(dim_range)
        else:
            raise BaseException("Unknown type of variable!")

        dimensions.append((dim, k)) # need to remember names of dimensions for evaluation code

    # initialization
    estimator = base_estimator(random_state=seed_value)
    solver = Optimizer([d[0] for d in dimensions], estimator, acq_optimizer="sampling", random_state=seed_value)

    trace = []
    best_y = np.inf
    best_x = None

    # optimization loop
    for i in range(max_iter):
        p = solver.ask()

        point = {d[1]:v for d,v in zip(dimensions, p)} # convert list of dimension values to dictionary

        v = -problem.evaluate(point) # the result of "evaluate" is accuracy / r^2, which is the more the better

        solver.tell(p, v)

        if best_y > v:
            best_y = v
            best_x = point

        trace.append((best_x, best_y)) # remember the point, objective pair
        print("Eval. #"+str(i))

    return trace

def run(n_calls = 32,
        n_runs = 1,
        save_traces = True,
        run_parallel = False):
    """
    Main function used to run the experiments.

    :param n_calls: evaluation budget
    :param n_runs: how many times to repeat the optimization in order to average out noise
    :param save_traces: whether to save data collected during optimization
    :param run_parallel: whether to run different repeats of optimization in parallel
    :return: None
    """

    selected_regressors = [GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor]
    selected_models = TesterClass.models.keys()
    selected_datasets = TesterClass.datasets.keys()

    pool = Pool()

    # all the parameter values and objectives collected during execution are stored in list below
    all_data = {}

    for Model in selected_models:
        all_data[Model] = {}

        for Dataset in selected_datasets:
            Supports, Message = TesterClass.supports(Model, Dataset)

            if not Supports:
                continue

            all_data[Model][Dataset] = {}

            for Regressor in selected_regressors:
                print(Regressor.__name__, Model, Dataset)

                params = [(Regressor, Model, Dataset, n_calls, np.random.randint(2 ** 30)) for rep in range(n_runs)]

                if run_parallel:
                    raw_trace = pool.map(evaluate_optimizer, params)
                else:
                    raw_trace = [evaluate_optimizer(p) for p in params]

                all_data[Model][Dataset][Regressor.__name__] = raw_trace

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
        help="Maximum number of allowed function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=5, type=int,
        help="Number of re-runs of single algorithm on single instance of a problem.")
    parser.add_argument(
        '--save_traces', nargs="?", default=False, type=bool,
        help="Whether to save pairs (point, objective) obtained during experiments in a json file.")
    parser.add_argument(
        '--run_parallel', nargs="?", default=True, type=bool,
        help="Whether to run in parallel or sequential mode.")

    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.save_traces, args.run_parallel)