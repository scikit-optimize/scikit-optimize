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

### bunch of dataset preprocessing functions ###

def split_normalize(X, Y):
    # only use training and validation - test is assumed to be evaluated after optimization
    tr_fold = int(len(X) * 0.7)

    X, Xv = X[:tr_fold], X[tr_fold:]
    Y, Yv = Y[:tr_fold], Y[tr_fold:]

    # normalize

    sc = StandardScaler()
    sc.fit(X, Y)

    X = sc.transform(X)
    Xv = sc.transform(Xv)

    return X, Y, Xv, Yv

def pow10map(x):
    return 10.0 ** x

def pow2intmap(x):
    return int(2.0 ** x)

def nop(x):
    return x

def load_data_target(data):
    X = data['data']
    Y = data['target']
    return X, Y

class TesterClass():

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
        # generate them data

        supp, reason = TesterClass.supports(model, dataset)

        if not supp:
            raise BaseException(reason)

        X, Y = self.datasets[dataset][DATASET_LOADER]()

        self.dataset = split_normalize(X, Y)
        self.model_class = TesterClass.models[model]

    def get_space(self):
        return self.model_class[MODEL_PARAMETERS]

    def evaluate(self, point):
        """
        :param point: configuration for some model
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
            r = 0.0 # todo: maybe use here negative value

        r = max(r, 0.0)

        return r



def visualize_traces(all_data):
    import matplotlib.pyplot as plt
    plt.close('all')

    average_for_algo = {}

    for Model in all_data:
        for Dataset in all_data[Model]:
            for Algorithm in all_data[Model][Dataset]:

                data = all_data[Model][Dataset][Algorithm]


                if not Algorithm in average_for_algo:
                    average_for_algo[Algorithm] = []

                # leave only objective values
                data = [[v[1] for v in d] for d in data]
                average_for_algo[Algorithm].extend(data)

    # calculate averages
    for Algorithm in average_for_algo:
        average_for_algo[Algorithm] = np.mean(average_for_algo[Algorithm], axis=0)

    to_render = {'Average objective': average_for_algo}

    def render_data(all_traces, name):
        plt.figure()
        idx = 0
        colors = ['r','g','b','v','o']

        for k, v in all_traces.iteritems():
            T = np.array(v)
            plt.scatter(range(len(T)), T, label=k, c=colors[idx])
            idx += 1

        plt.xlabel("Iteration")
        plt.grid()
        plt.title(name)
        plt.legend()

    for k, p in to_render.iteritems():
        render_data(p, k)

    plt.show()

def evaluate_optimizer((base_estimator, model, dataset, max_iter, seed_value)):
    # below seed is necessary for processes which fork at the same time
    # so that random numbers generated in processes are different
    np.random.seed(seed_value)

    problem = TesterClass(model, dataset)
    space = problem.get_space()

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

    estimator = base_estimator(random_state=seed_value)
    solver = Optimizer([d[0] for d in dimensions], estimator, acq_optimizer="sampling", random_state=seed_value)

    trace = []
    best_y = np.inf
    best_x = None

    for i in range(max_iter):
        p = solver.ask()

        point = {d[1]:v for d,v in zip(dimensions, p)}

        v = -problem.evaluate(point)

        solver.tell(p, v)

        if best_y > v:
            best_y = v
            best_x = point

        trace.append((best_x, best_y))
        print("Eval. #"+str(i))

    return trace

if __name__ == "__main__":

    max_iter = 32
    repetitions = 1
    save_traces = True
    run_parallel = False

    selected_algorithms = [ GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor]

    # For the full list of models see: Test_Model_On_Dataset.models
    selected_models = ['DecisionTreeClassifier', 'DecisionTreeRegressor'] #TesterClass.models.keys()
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

            for Algorithm in selected_algorithms:
                print(Algorithm.__name__, Model, Dataset)

                params = [(Algorithm, Model, Dataset, max_iter, np.random.randint(2**30)) for rep in range(repetitions)]

                if run_parallel:
                    raw_trace = pool.map(evaluate_optimizer, params)
                else:
                    raw_trace = [evaluate_optimizer(p) for p in params] # used for debugging

                all_data[Model][Dataset][Algorithm.__name__] = deepcopy(raw_trace)

    # dump the recorded objective values as json
    if save_traces:
        with open(datetime.now().strftime("%M_%Y_%d_%h_%m_%s")+'.json', 'w') as f:
            json.dump(all_data, f)

    visualize_traces(all_data)

