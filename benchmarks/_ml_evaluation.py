
import _ml_datasets as dts
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from _sk_keras import KerasRegressor, KerasClassifier

from skopt import Optimizer
from skopt.space import Real, Categorical, Integer

### bunch of constants ###

FMT_NUMBER = dts.NUMERIC_OUTPUT
FMT_CATEGORY = dts.CATEGORY_OUTPUT
FMT_SKIP = "Skip"

OUTPUT_COLUMN = dts.OUTPUT_COLUMN
OUTPUT_TYPE = dts.OUTPUT_TYPE

MODEL_PARAMETERS = "model parameters"
MODEL_BACKEND = "model backend"
PARAM_MAP = "parameter map"

CATEGORY_VARIABLE = "Categorical"
REAL_VARIABLE = "Real"
INTEGER_VARIABLE = "Integer"

VARIABLE_TYPE = "type"
VARIABLE_RANGE = "range"

### bunch of dataset preprocessing functions ###

def csv_to_numbers(D):
    """
    Converts a csv table to a matrix which could be used for further machine learning.
    :param D: Dataset csv as numpy matrix
    :return: data matrix
    """

    R = []

    for column in D.T:

        # test if column consists mostly of numbers
        count = 0

        for v in column:
            try:
                float(v)
                count += 1
            except BaseException as ex:
                pass

        fraction = count / (len(column) * 1.0)

        # test if mostly numbers
        mostly_numbers = fraction > 0.8

        # test categorical values
        sparcity = len(np.unique(column))
        sparse_values = sparcity < 32

        if mostly_numbers and not sparse_values:
            f = FMT_NUMBER
        elif sparse_values:
            f = FMT_CATEGORY
        else:
            f = FMT_SKIP

        if f == FMT_NUMBER:
            V = np.genfromtxt(column.astype('unicode'))
            I = np.isnan(V)

            if np.any(I):
                V[I] = np.mean(V[~I]) # impute with mean missing values
                I = I * 1.0 # add additional column which has values of 1 if there was imputation
                R.append(V)
                R.append(I)
            else:
                R.append(V) # no missing values - append column as is

        elif f == FMT_CATEGORY:
            # convert column to integer categories
            U = np.unique(column)
            I = {u: i for i, u in enumerate(U)}
            Z = []

            for v in column:
                z = np.zeros(len(U))
                i = I[v]
                z[i] = 1.0
                Z.append(z)

            Z = np.row_stack(Z)
            R.append(Z)
        elif f == FMT_SKIP:
            continue

    R = np.column_stack(R)

    return R

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

class Test_Model_On_Dataset():

    nnparams = {
        # up to 1024 neurons
        'n_neurons': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 10.0], PARAM_MAP: pow2intmap},
        'n_layers': {VARIABLE_TYPE: INTEGER_VARIABLE, VARIABLE_RANGE: [1, 10], PARAM_MAP: nop},
        'neuron_type': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: ['relu', 'tanh'], PARAM_MAP: nop},
        # learning rate: 0.1 .. 0.00001
        'opt_par_1': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-5.0, -1], PARAM_MAP: pow10map},
        'opt_par_2': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.1, 0.98], PARAM_MAP: nop},
        'opt_par_3': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.1, 0.98], PARAM_MAP: nop},
        'optim_type': {VARIABLE_TYPE: CATEGORY_VARIABLE, VARIABLE_RANGE: ['adam', 'rmsprop'], PARAM_MAP: nop},
        # batch size starts from 32 - smaller batches might slow down the computations
        'batch_size': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [5.0, 10.0], PARAM_MAP: pow2intmap},
        'dropout': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [0.0, 0.9], PARAM_MAP: nop},
        # l2 regularization: 0.1 .. 0.000001.
        'l2_regularization': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [-5.0, -1], PARAM_MAP: pow10map},
    }

    models = {
        #
        #   regressors go here
        #
        KerasRegressor.__name__:{
            OUTPUT_TYPE: FMT_NUMBER,
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: KerasRegressor,
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
        KerasClassifier.__name__:{
            OUTPUT_TYPE: FMT_CATEGORY,
            MODEL_PARAMETERS: nnparams,
            MODEL_BACKEND: KerasClassifier,
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
                'max_depth': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 4.0],
                              PARAM_MAP: pow2intmap},
                'min_samples_split': {VARIABLE_TYPE: REAL_VARIABLE, VARIABLE_RANGE: [1.0, 8.0],
                                      PARAM_MAP: pow2intmap},
            },
            MODEL_BACKEND: DecisionTreeClassifier,
        },
    }

    @staticmethod
    def supports(model, dataset):
        if not dataset in dts.datasets:
            return False, "unknown dataset"

        dst = dts.datasets[dataset]

        if not model in Test_Model_On_Dataset.models:
            return False, "unknown model"

        mod = Test_Model_On_Dataset.models[model]

        if not dst[OUTPUT_TYPE] == mod[OUTPUT_TYPE]:
            return False, "model cannot be applied to the dataset"

        return True, None

    def get_dataset(self, name):

        dataset = dts.datasets[name]
        otp_idx = dataset[OUTPUT_COLUMN]
        otp_type = dataset[OUTPUT_TYPE]
        dataset_fmt = dataset[dts.DATASET_FORMAT]

        if dataset_fmt == dts.DATASET_TABLE:
            D = dts.data(name).as_matrix()

            # shuffle rows of matrix to avoid any funny distributions
            #np.random.shuffle(D)

            Y = D[:, otp_idx]  # get the output column

            # select all columns except for the output one
            I = np.array([True] * D.shape[1])
            I[otp_idx] = False

            X = D[:, I]
            # convert feature columns to matrix
            X = csv_to_numbers(X)

            if otp_type == FMT_CATEGORY:
                # convert to integer representation
                I = {u: i for i, u in enumerate(np.unique(Y))}
                Y = np.vectorize(lambda v: I[v])(Y)
            elif otp_type == FMT_NUMBER:
                Y = Y.astype('float')
        else:
            raise BaseException('Unknown type of dataset')

        return X, Y

    def __init__(self, model, dataset):
        # generate them data

        supp, reason = Test_Model_On_Dataset.supports(model, dataset)

        if not supp:
            raise BaseException(reason)

        X, Y = self.get_dataset(dataset)

        self.dataset = split_normalize(X, Y)
        self.model_class = Test_Model_On_Dataset.models[model]

    def get_space(self):
        return self.model_class[MODEL_PARAMETERS]

    def evaluate(self, point):
        """
        :param point: configuration for some model
        :return: score (more is better!) for some specific point
        """

        X, Y, Xv, Yv = self.dataset

        cls = self.model_class[MODEL_BACKEND]

        # map the model parameters
        point_mapped = {}

        for param, v in point.iteritems():
            v = self.model_class[MODEL_PARAMETERS][param][PARAM_MAP](v)
            point_mapped[param] = v

        model = cls(**point_mapped)
        model.fit(X, Y)
        r = model.score(Xv, Yv)

        return r



def visualize_traces(all_data):
    from copy import deepcopy
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

    problem = Test_Model_On_Dataset(model, dataset)
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
            best_x = p

        trace.append((best_x, best_y))
        print("Eval. #"+str(i))

    return trace